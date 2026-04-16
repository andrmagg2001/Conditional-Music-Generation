import re
import torch
import argparse
import pathlib
import miditoolkit
import warnings
import sys

import json as js

from feature_extraction import classify_audio_file
from train_baseline import BaselineTransformerLM


# Legacy regex for old monolithic token format (kept for reference)
EVENT_RE = re.compile(r"^([DBH]):s=(\d+),e=(\d+),p=(\d+),v=(\d+)$")

# Sub-token prefixes used by the new tokenization scheme
_SUBTOKEN_PREFIXES = ("<R_", "<S_", "<E_", "<P_", "<V_")


def load_meta(meta_path: pathlib.Path) -> dict:
    """Load optional generation metadata (BPM/GENRE/CHORDS) from JSON file."""
    if not meta_path.is_file():
        return {"bpm": None, "genre": None, "chords": []}

    with meta_path.open("r", encoding="utf-8") as f:
        payload = js.load(f)

    bpm = payload.get("BPM")
    try:
        bpm = float(bpm) if bpm is not None else None
    except Exception:
        bpm = None

    genre = payload.get("GENRE")
    genre = str(genre).strip() if genre is not None else None

    chords = payload.get("CHORDS", [])
    if not isinstance(chords, list):
        chords = []

    return {
        "bpm": bpm,
        "genre": genre,
        "chords": chords,
    }


def _pick_existing_token(candidates: list[str], token_to_id: dict[str, int]) -> str | None:
    for c in candidates:
        if c in token_to_id:
            return c
    return None


def _sample_next_id(logits: torch.Tensor, top_k: int, temperature: float) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    scaled = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(int(top_k), int(scaled.numel()))
        values, indices = torch.topk(scaled, k=k)
        probs = torch.softmax(values, dim=-1)
        sampled_local = torch.multinomial(probs, num_samples=1)
        return int(indices[sampled_local].item())

    probs = torch.softmax(scaled, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    return int(sampled.item())


def _pick_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _extract_event_tokens(generated_ids: list[int], id_to_token: dict[int, str]) -> list[str]:
    """Extract sub-tokens between <LOOP> and </LOOP> from generated ids."""
    tokens = [id_to_token.get(i, "<UNK>") for i in generated_ids]
    try:
        loop_start = tokens.index("<LOOP>") + 1
    except ValueError:
        return []

    end_idx = len(tokens)
    for i in range(loop_start, len(tokens)):
        if tokens[i] in {"</LOOP>", "<EOS>"}:
            end_idx = i
            break

    # Keep only sub-tokens (filter out any non-event tokens)
    return [t for t in tokens[loop_start:end_idx] if t.startswith(_SUBTOKEN_PREFIXES)]


def load_vocab(vocab_path: pathlib.Path) -> tuple[dict[str, int], dict[int, str]]:
    """
    Load and validate vocabulary mappings from a JSON file.

    The function expects a JSON payload containing at least `token_to_id`.
    If `id_to_token` is missing or empty, it is rebuilt from `token_to_id`.
    Output mappings are normalized to consistent Python types:
    - token_to_id: dict[str, int]
    - id_to_token: dict[int, str]

    Required special tokens are checked to ensure the generation pipeline
    can build prompts and stop conditions correctly.

    Parameters
    ----------
    vocab_path : pathlib.Path
        Path to the vocabulary JSON file (typically `data/json/vocab.json`).

    Returns
    -------
    tuple[dict[str, int], dict[int, str]]
        A tuple containing:
        1) token_to_id mapping (token string -> integer id)
        2) id_to_token mapping (integer id -> token string)

    Raises
    ------
    ValueError
        If `token_to_id` is missing, not a dictionary, or empty.
    KeyError
        If one or more required special tokens are missing
        (`<BOS>`, `<EOS>`, `<LOOP>`, `</LOOP>`, `<UNK>`, `<PAD>`).
    json.JSONDecodeError
        If the file is not valid JSON.
    OSError
        If the file cannot be opened/read.
    """
    with vocab_path.open("r", encoding="utf-8") as f:
        payload = js.load(f)
    
    token_to_id = payload.get("token_to_id", {})
    id_to_token = payload.get("id_to_token", {})

    if not isinstance(token_to_id, dict) or len(token_to_id) == 0:
        raise ValueError("token_to_id missing or empty in vocab.json")
    
    if not isinstance(id_to_token, dict) or len(id_to_token) == 0:
        id_to_token = {str(v): k for k, v in token_to_id.items()}

    token_to_id = {str(k): int(v) for k, v in token_to_id.items()}
    id_to_token_int = {int(k): str(v) for k, v in id_to_token.items()}

    required = ["<BOS>", "<EOS>", "<LOOP>", "</LOOP>", "<UNK>", "<PAD>"]
    missing = [tok for tok in required if tok not in token_to_id]
    if missing:
        raise KeyError(f"Missing required vocab tokens: {missing}")
    
    return token_to_id, id_to_token_int




def load_model_from_ckpt(ckpt_path: pathlib.Path, vocab_size: int, device: torch.device):
    """
    Load a trained Transformer checkpoint and rebuild the model for inference.

    The checkpoint must contain:
    - `model`: state_dict with learned weights
    - `config`: training configuration used to instantiate the architecture

    The function recreates `BaselineTransformerLM` from checkpoint config,
    loads weights with strict matching, moves the model to the target device,
    and switches it to eval mode.

    Parameters
    ----------
    ckpt_path : pathlib.Path
        Path to the `.pth` checkpoint file.
    vocab_size : int
        Vocabulary size used to build embedding/output layers.
    device : torch.device
        Target device for inference (`cpu`, `cuda`, or `mps`).

    Returns
    -------
    tuple[BaselineTransformerLM, dict]
        A tuple containing:
        1) the loaded model in eval mode on `device`
        2) checkpoint config dictionary (`cfg`)

    Raises
    ------
    FileNotFoundError
        If the checkpoint path does not exist.
    KeyError
        If required keys (`model`, `config`, or config fields) are missing.
    RuntimeError
        If state_dict loading fails (shape mismatch / incompatible architecture).
    """
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" not in ckpt or "config" not in ckpt:
        raise KeyError("Checkpoint must contain keys: 'model' and 'config'")
    
    cfg = ckpt["config"]
    state = ckpt["model"]

    required_cfg = ["max_len", "d_model", "n_heads", "n_layers", "d_ff", "dropout"]
    missing = [k for k in required_cfg if k not in cfg]

    if missing: raise KeyError(f"Missing config fields in checkpoint: {missing}")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="enable_nested_tensor is True, but self.use_nested_tensor is False.*",
            category=UserWarning,
        )
        model = BaselineTransformerLM(
            vocab_size=vocab_size,
            pad_id=0,
            max_len=int(cfg["max_len"]),
            d_model=int(cfg["d_model"]),
            n_heads=int(cfg["n_heads"]),
            n_layers=int(cfg["n_layers"]),
            d_ff=int(cfg["d_ff"]),
            dropout=float(cfg["dropout"]),
        ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg
    


def _bucket_bpm(bpm: float) -> int:
    """Round BPM to nearest 10, clamp to [60, 200]. Values >200 are halved first."""
    b = float(bpm)
    if b > 200:
        b = b / 2.0
    b = max(60.0, min(200.0, b))
    return int(round(b / 10.0) * 10)


def build_prompt(
    genre: str,
    bpm: float,
    token_to_id: dict[str, int],
    len_token: str = "<LEN_128>",
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Build a conditioned prompt token sequence matching the training format.

    The output token order is:
        <GENRE_X> <BPM_Y> <LEN_Z> Generate one loop. Output only tokens. <LOOP>

    Parameters
    ----------
    genre : str
        Genre label (e.g. "rock", "jazz", "unknown").
    bpm : float
        Raw BPM value — will be bucketed to nearest 10.
    token_to_id : dict
        Vocabulary mapping.
    len_token : str
        Loop length token, default "<LEN_128>".
    strict : bool
        If True, raise on missing tokens; otherwise warn and continue.

    Returns
    -------
    tuple of (prompt_tokens, warnings)
    """
    warnings = []

    if not isinstance(token_to_id, dict) or len(token_to_id) == 0:
        raise ValueError("token_to_id empty or not valid")

    # ── Validate critical tokens ────────────────
    critical = ["<LOOP>", "</LOOP>", "<BOS>", "<EOS>", "<UNK>", "<PAD>"]
    missing_critical = [t for t in critical if t not in token_to_id]
    if missing_critical:
        raise KeyError(f"Critical token not in vocab: {missing_critical}")

    # ── Genre token ─────────────────────────────
    genre_token = f"<GENRE_{genre}>"
    if genre_token not in token_to_id:
        warnings.append(f"Genre token {genre_token} not in vocab, falling back to <GENRE_unknown>")
        genre_token = "<GENRE_unknown>"
        if genre_token not in token_to_id and strict:
            raise KeyError("No <GENRE_unknown> token in vocab")

    # ── BPM token ───────────────────────────────
    bpm_bucket = _bucket_bpm(bpm)
    bpm_token = f"<BPM_{bpm_bucket}>"
    if bpm_token not in token_to_id:
        warnings.append(f"BPM token {bpm_token} not in vocab, trying nearest")
        bpm_candidates = sorted(
            [t for t in token_to_id if t.startswith("<BPM_") and t.endswith(">")]
        )
        if bpm_candidates:
            bpm_token = min(bpm_candidates, key=lambda t: abs(int(t[5:-1]) - bpm_bucket))
            warnings.append(f"Using nearest BPM token: {bpm_token}")
        elif strict:
            raise KeyError("No <BPM_*> token in vocab")

    # ── LEN token ───────────────────────────────
    if not len_token or len_token not in token_to_id:
        if "<LEN_128>" in token_to_id:
            warnings.append(f"len_token not valid: using <LEN_128> over {len_token}")
            len_token = "<LEN_128>"
        else:
            len_candidates = [t for t in token_to_id if t.startswith("<LEN_") and t.endswith(">")]
            if len_candidates:
                len_token = sorted(len_candidates)[0]
                warnings.append(f"<LEN_128> not found: using fallback {len_token}")
            elif strict:
                raise KeyError("No <LEN_*> token in vocab")
            else:
                warnings.append("No LEN token available: continuing without it")
                len_token = None

    # ── Instruction tokens ──────────────────────
    tok_generate = _pick_existing_token(["Generate", "generate"], token_to_id)
    tok_one = _pick_existing_token(["one", "One"], token_to_id)
    tok_loop = _pick_existing_token(["loop.", "loop"], token_to_id)
    tok_output = _pick_existing_token(["Output", "output"], token_to_id)
    tok_only = _pick_existing_token(["only", "Only"], token_to_id)
    tok_tokens = _pick_existing_token(["tokens.", "tokens"], token_to_id)

    instruction_parts = [tok_generate, tok_one, tok_loop, tok_output, tok_only, tok_tokens]
    if any(x is None for x in instruction_parts):
        if strict:
            raise KeyError("Instruction tokens not in vocab for base prompt")
        warnings.append("Missing instruction tokens: using minimum fallback")
        instruction_parts = [x for x in instruction_parts if x is not None]

    # ── Assemble prompt: <GENRE> <BPM> <LEN> instruction <LOOP> ──
    prompt_tokens = [genre_token, bpm_token]
    if len_token is not None:
        prompt_tokens.append(len_token)
    prompt_tokens.extend(instruction_parts)
    prompt_tokens.append("<LOOP>")

    warnings.append(f"conditioning: genre={genre}, bpm_bucket={bpm_bucket}")

    out_of_vocab = [t for t in prompt_tokens if t not in token_to_id]
    if out_of_vocab:
        if strict:
            raise KeyError(f"Final prompt contains OOV tokens: {out_of_vocab}")
        warnings.append(f"OOV tokens in prompt: {out_of_vocab}")

    return prompt_tokens, warnings


def generate_tokens(
    model,
    prompt_ids: list[int],
    eos_id: int,
    loop_end_id: int,
    max_new_tokens: int,
    min_new_tokens: int,
    top_k: int,
    temperature: float,
    device: torch.device,
    ) -> list[int]:
    """
    Autoregressive generation with safe stopping conditions.

    Stops when:
    - </LOOP> is generated
    - <EOS> is generated
    - max_new_tokens is reached
    """
    if not prompt_ids: raise ValueError("prompt_ids is empty")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if min_new_tokens < 0:
        raise ValueError("min_new_tokens must be >= 0")

    model.eval()
    generated = list(prompt_ids)
    max_ctx = int(getattr(model, "max_len", 1024))

    with torch.no_grad():
        for step_i in range(max_new_tokens):
            ctx = generated[-max_ctx:]
            input_ids = torch.tensor([ctx], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

            logits = model(input_ids, attention_mask)[:, -1, :].squeeze(0)
            # Do not allow early stop tokens until we have generated enough tokens.
            if step_i + 1 < min_new_tokens:
                logits = logits.clone()
                logits[eos_id] = float("-inf")
                logits[loop_end_id] = float("-inf")

            next_id = _sample_next_id(logits, top_k=top_k, temperature=temperature)

            generated.append(next_id)

            if next_id == loop_end_id or next_id == eos_id:
                break

    return generated


def _parse_subtoken_value(tok: str) -> int | None:
    """Extract integer value from a sub-token like '<S_42>' -> 42."""
    try:
        return int(tok[tok.index('_') + 1 : -1])
    except (ValueError, IndexError):
        return None


def tokens_to_midi(event_tokens: list[str], out_path: pathlib.Path,
                   step_ticks: int = 120, tpb: int = 480, bpm: float = 120.0) -> int:
    """
    Convert generated sub-tokens to a MIDI file.

    Expects tokens in groups of 5: <R_X> <S_x> <E_x> <P_x> <V_x>
    Returns the number of valid notes written.
    """
    midi = miditoolkit.MidiFile(ticks_per_beat=tpb)
    midi.tempo_changes = [miditoolkit.TempoChange(tempo=float(bpm), time=0)]
    midi.time_signature_changes = [miditoolkit.TimeSignature(4, 4, 0)]

    inst_map = {
        "D": miditoolkit.Instrument(program=0, is_drum=True, name="drums"),
        "B": miditoolkit.Instrument(program=34, is_drum=False, name="bass"),
        "H": miditoolkit.Instrument(program=0, is_drum=False, name="harmony"),
    }

    note_count = 0
    i = 0
    while i + 4 < len(event_tokens):
        r_tok = event_tokens[i]
        s_tok = event_tokens[i + 1]
        e_tok = event_tokens[i + 2]
        p_tok = event_tokens[i + 3]
        v_tok = event_tokens[i + 4]

        # Validate sub-token sequence starts with <R_>
        if not r_tok.startswith("<R_"):
            i += 1  # skip and try to resync
            continue

        role = r_tok[3:-1]  # "D", "B", or "H"
        if role not in inst_map:
            i += 1
            continue

        s_i = _parse_subtoken_value(s_tok)
        e_i = _parse_subtoken_value(e_tok)
        p_i = _parse_subtoken_value(p_tok)
        v_i = _parse_subtoken_value(v_tok)

        i += 5  # advance to next group

        if any(x is None for x in (s_i, e_i, p_i, v_i)):
            continue
        if e_i <= s_i:
            continue
        if not (0 <= p_i <= 127 and 1 <= v_i <= 127):
            continue

        note = miditoolkit.Note(
            velocity=v_i,
            pitch=p_i,
            start=s_i * step_ticks,
            end=e_i * step_ticks,
        )
        inst_map[role].notes.append(note)
        note_count += 1

    for role in ["D", "B", "H"]:
        inst = inst_map[role]
        if inst.notes:
            inst.notes.sort(key=lambda n: (n.start, n.end, n.pitch, n.velocity))
            midi.instruments.append(inst)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(out_path))
    return note_count


def validate_midi(midi_path: pathlib.Path) -> bool:
    """Return True when MIDI can be parsed and contains valid note durations."""
    try:
        parsed = miditoolkit.MidiFile(str(midi_path))
    except Exception:
        return False

    total_notes = 0
    for inst in parsed.instruments:
        total_notes += len(inst.notes)
        for note in inst.notes:
            if note.end <= note.start:
                return False

    return total_notes > 0


def score_sample(note_count: int, event_count: int, valid: bool) -> float:
    """Score generated candidates; invalid samples always rank last."""
    if not valid:
        return -1e9
    return float(event_count) + 0.5 * float(note_count)


def prompt_bpm_confirmation(bpm_value: float) -> float:
    """Ask user to confirm or override BPM in interactive terminal sessions."""
    if not sys.stdin.isatty():
        return bpm_value

    prompt = (
        f"I understand that BPM is {bpm_value:.2f}. Is this correct? "
        "[Press Enter to confirm, type the correct number to override]: "
    )

    while True:
        raw = input(prompt).strip()
        if raw == "":
            return bpm_value
        try:
            new_bpm = float(raw)
        except ValueError:
            print("Invalid input. Please press Enter or type a numeric BPM value.")
            continue
        if new_bpm <= 0:
            print("BPM must be greater than 0.")
            continue
        return new_bpm


def main():
    parser = argparse.ArgumentParser(description="Classify audio and generate MIDI with baseline Transformer")
    parser.add_argument("--audio", type=str, required=True, help="Input audio file path")
    parser.add_argument("--meta", type=str, default="data/json/meta.json")
    parser.add_argument("--ckpt", type=str, default="data/checkpoints/baseline_transformer/latest.pth")
    parser.add_argument("--vocab", type=str, default="data/json/vocab.json")
    parser.add_argument("--out-dir", type=str, default="data/samples/pipeline")
    parser.add_argument("--len-token", type=str, default="<LEN_128>")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=420)
    parser.add_argument("--min-new-tokens", type=int, default=96)
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--min-notes", type=int, default=16)
    parser.add_argument("--step-ticks", type=int, default=120)
    parser.add_argument("--tpb", type=int, default=480)
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    device = _pick_device(args.device)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(pathlib.Path(args.meta))

    tok2id, id2tok = load_vocab(pathlib.Path(args.vocab))
    model, _ = load_model_from_ckpt(pathlib.Path(args.ckpt), vocab_size=len(tok2id), device=device)

    style_label = meta["genre"] or classify_audio_file(pathlib.Path(args.audio))

    bpm_value = float(meta["bpm"]) if meta["bpm"] is not None else float(args.bpm)
    bpm_value = prompt_bpm_confirmation(bpm_value)

    prompt_tokens, prompt_warnings = build_prompt(style_label, bpm_value, tok2id, args.len_token, strict=True)
    prompt_warnings.append(f"meta_chords_count={len(meta['chords'])}")

    bos_id = tok2id["<BOS>"]
    eos_id = tok2id["<EOS>"]
    loop_end_id = tok2id["</LOOP>"]
    prompt_ids = [bos_id] + [tok2id[t] for t in prompt_tokens]

    report_path = out_dir / "report.jsonl"
    valid_count = 0
    best_idx = None
    best_score = -1e9

    with report_path.open("w", encoding="utf-8") as rep:
        for idx in range(1, args.num_candidates + 1):
            gen_ids = generate_tokens(
                model=model,
                prompt_ids=prompt_ids,
                eos_id=eos_id,
                loop_end_id=loop_end_id,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                top_k=args.top_k,
                temperature=args.temperature,
                device=device,
            )

            event_tokens = _extract_event_tokens(gen_ids, id2tok)
            token_file = out_dir / f"sample_{idx:03d}.txt"
            midi_file = out_dir / f"sample_{idx:03d}.mid"
            token_file.write_text(" ".join(event_tokens), encoding="utf-8")

            note_count = tokens_to_midi(
                event_tokens=event_tokens,
                out_path=midi_file,
                step_ticks=args.step_ticks,
                tpb=args.tpb,
                bpm=bpm_value,
            )
            ok = note_count >= args.min_notes and validate_midi(midi_file)
            if ok:
                valid_count += 1
                sample_score = score_sample(note_count, len(event_tokens), ok)
                if sample_score > best_score:
                    best_score = sample_score
                    best_idx = idx

            row = {
                "sample": idx,
                "valid": ok,
                "event_count": len(event_tokens),
                "note_count": note_count,
                "score": score_sample(note_count, len(event_tokens), ok),
                "midi": str(midi_file),
            }
            rep.write(js.dumps(row) + "\n")

            print(
                f"[SAMPLE {idx}/{args.num_candidates}] valid={ok} "
                f"events={len(event_tokens)} notes={note_count}"
            )

    summary = {
        "audio": str(pathlib.Path(args.audio)),
        "style_label": style_label,
        "meta_genre": meta["genre"],
        "meta_bpm": bpm_value,
        "meta_chords_count": len(meta["chords"]),
        "prompt": prompt_tokens,
        "warnings": prompt_warnings,
        "valid_count": valid_count,
        "num_candidates": args.num_candidates,
        "min_notes": args.min_notes,
        "best_sample": best_idx,
        "device": str(device),
    }
    (out_dir / "summary.json").write_text(js.dumps(summary, indent=2), encoding="utf-8")

    print("[DONE]", js.dumps(summary, indent=2))


if __name__ == "__main__":
    main()