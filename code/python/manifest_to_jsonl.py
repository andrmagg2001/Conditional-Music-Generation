import os
import yaml
import miditoolkit
import json as js



INSTRUCTION     = "Generate one loop. Output only tokens."
YAML_PATH       = "code/python/preprocess.yaml"
JSON_MANIFEST   = ""
ENRICHED_MANIFEST = "data/json/manifest_enriched.jsonl"
JSON_OUTPUT_DIR = ""
JSON_LOG_DIR    = ""
QUANT_GRID      = ""
TARGET_TPB      = 0
STEPS_PER_BEAT  = 0
STEP_TICKS      = 0
LOOP_STEPS_CFG  = None
MAX_HARMONY_POLY = None 


def bump_reason(summary: dict, key: str, inc: int = 1):
    summary["reasons"][key] = summary["reasons"].get(key, 0) + inc


def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


with open(YAML_PATH, "r", encoding="utf-8") as stream:
    temp = yaml.safe_load(stream)


if os.path.isfile(ENRICHED_MANIFEST):
    JSON_MANIFEST = ENRICHED_MANIFEST
    print(f"[INFO] Using enriched manifest: {ENRICHED_MANIFEST}")
else:
    JSON_MANIFEST = temp["paths"]["manifest_path"]
    print(f"[WARN] Enriched manifest not found, using base: {JSON_MANIFEST}")
JSON_OUTPUT_DIR = temp["paths"]["output_json_dir"]
JSON_LOG_DIR    = temp["paths"]["report_dir"]

TARGET_TPB      = int(temp["normalize"]["target_tpb"])
STEPS_PER_BEAT  = int(temp["quantization"]["steps_per_beat"])
QUANT_GRID      = str(temp["quantization"]["grid"])

LOOP_STEPS_CFG  = temp.get("loop", {}).get("steps", None)
if LOOP_STEPS_CFG is not None:
    LOOP_STEPS_CFG = int(LOOP_STEPS_CFG)

MAX_HARMONY_POLY = temp.get("loop", {}).get("max_harmony_polyphony", None)
if MAX_HARMONY_POLY is not None:
    MAX_HARMONY_POLY = int(MAX_HARMONY_POLY)
    if MAX_HARMONY_POLY <= 0:
        MAX_HARMONY_POLY = None

STEP_TICKS = TARGET_TPB // STEPS_PER_BEAT
if STEP_TICKS <= 0:
    raise ValueError(f"Invalid STEP_TICKS={STEP_TICKS}. Check target_tpb and steps_per_beat.")


manifest_dir            = os.path.dirname(JSON_MANIFEST)
processed_jsonl_path    = os.path.join(JSON_OUTPUT_DIR, "processed.jsonl")
errors_jsonl_path       = os.path.join(JSON_LOG_DIR, "errors.jsonl")
discarded_jsonl_path    = os.path.join(JSON_LOG_DIR, "discarded.jsonl")
summary_json_path       = os.path.join(JSON_LOG_DIR, "summary.json")

os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
os.makedirs(JSON_LOG_DIR,    exist_ok=True)

for p in [processed_jsonl_path, errors_jsonl_path, discarded_jsonl_path]:
    open(p, "w", encoding="utf-8").close()

summary = {
    "total": 0,
    "ok": 0,
    "errors": 0,
    "discarded": 0,
    "reasons": {}
}

def quantize_velocity(v: int, step: int = 8) -> int:
    """Quantize velocity to nearest multiple of `step`, clamped to [1, 127]."""
    v = max(1, min(127, int(v)))
    q = int(round(v / step) * step)
    return max(1, min(127, q))


def event_to_subtokens(role: str, ev: dict) -> list[str]:
    """
    Convert a single MIDI event into 5 sub-tokens:
        <R_D> <S_0> <E_1> <P_42> <V_112>

    This reduces vocabulary from ~419K combinatorial tokens to ~450 atomic tokens.
    """
    r = role[0].upper()  # D, B, H
    s = int(ev["start_step"])
    e = int(ev["end_step"])
    p = int(ev["pitch"])
    v = quantize_velocity(int(ev["velocity"]))
    return [f"<R_{r}>", f"<S_{s}>", f"<E_{e}>", f"<P_{p}>", f"<V_{v}>"]


def roles_to_tokens(roles: dict) -> str:
    """Serialize all roles into a flat string of sub-tokens."""
    tokens = []
    for role_name in ["drums", "bass", "harmony"]:
        evs = roles.get(role_name, [])
        evs = sorted(evs, key=lambda x: (x["start_step"], x["end_step"], x["pitch"], x["velocity"]))
        for ev in evs:
            tokens.extend(event_to_subtokens(role_name, ev))
    return " ".join(tokens)


_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_NOTE_NAMES = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]


def detect_key_from_events(bass_events: list, harmony_events: list) -> str:
    """
    Detect the musical key from bass + harmony MIDI events using
    the Krumhansl-Schmuckler algorithm (pitch class correlation).

    Returns a key label like 'C', 'Am', 'Fs', 'Dsm' (lowercase m = minor).
    Falls back to 'C' if no pitched events are available.
    """
    pc_hist = [0.0] * 12
    for ev in bass_events + harmony_events:
        pc = int(ev["pitch"]) % 12
        duration = max(1, int(ev["end_step"]) - int(ev["start_step"]))
        pc_hist[pc] += duration

    total = sum(pc_hist)
    if total == 0:
        return "C"

    pc_hist = [x / total for x in pc_hist]

    best_key = "C"
    best_corr = -999.0

    for shift in range(12):
        major_rot = [_MAJOR_PROFILE[(i - shift) % 12] for i in range(12)]
        minor_rot = [_MINOR_PROFILE[(i - shift) % 12] for i in range(12)]

        for profile, suffix in [(major_rot, ""), (minor_rot, "m")]:
            mean_p = sum(profile) / 12
            mean_h = sum(pc_hist) / 12
            num = sum((pc_hist[i] - mean_h) * (profile[i] - mean_p) for i in range(12))
            den_h = sum((pc_hist[i] - mean_h) ** 2 for i in range(12)) ** 0.5
            den_p = sum((profile[i] - mean_p) ** 2 for i in range(12)) ** 0.5
            if den_h * den_p == 0:
                continue
            corr = num / (den_h * den_p)
            if corr > best_corr:
                best_corr = corr
                best_key = _NOTE_NAMES[shift] + suffix

    return best_key


def clip_events_to_loop(evs: list, loop_steps: int) -> list:
    """
    Keep only events that intersect [0, loop_steps).
    Clip end to loop_steps and discard empty/invalid events.
    """
    out = []
    for ev in evs:
        s = int(ev["start_step"])
        e = int(ev["end_step"])

        if e <= 0 or s >= loop_steps:
            continue

        s2 = max(0, s)
        e2 = min(loop_steps, e)

        if e2 <= s2:
            continue

        out.append({
            "start_step": s2,
            "end_step": e2,
            "pitch": int(ev["pitch"]),
            "velocity": int(ev["velocity"]),
        })
    return out


def shift_events_to_zero(evs: list) -> list:
    """Shift events so that the earliest start_step becomes 0."""
    if not evs:
        return evs
    min_s = min(int(ev["start_step"]) for ev in evs)
    if min_s <= 0:
        return evs
    out = []
    for ev in evs:
        s = int(ev["start_step"]) - min_s
        e = int(ev["end_step"]) - min_s
        if e <= s:
            continue
        out.append({
            "start_step": s,
            "end_step": e,
            "pitch": int(ev["pitch"]),
            "velocity": int(ev["velocity"]),
        })
    return out


def dedup_events(evs: list) -> list:
    """Remove exact duplicates (same s,e,p,v), preserving first occurrence."""
    seen = set()
    out = []
    for ev in evs:
        k = (int(ev["start_step"]), int(ev["end_step"]), int(ev["pitch"]), int(ev["velocity"]))
        if k in seen:
            continue
        seen.add(k)
        out.append(ev)
    return out


def limit_polyphony_per_step(evs: list, max_poly: int) -> list:
    """Limit number of notes that start at the same step (keep highest-velocity first)."""
    if max_poly is None or max_poly <= 0:
        return evs

    buckets = {}
    for ev in evs:
        s = int(ev["start_step"])
        buckets.setdefault(s, []).append(ev)

    out = []
    for s, group in buckets.items():
        if len(group) <= max_poly:
            out.extend(group)
            continue
        group_sorted = sorted(group, key=lambda x: (-int(x["velocity"]), int(x["pitch"]), int(x["end_step"])))
        out.extend(group_sorted[:max_poly])

    out = sorted(out, key=lambda x: (int(x["start_step"]), int(x["end_step"]), int(x["pitch"]), int(x["velocity"])))
    return out


def is_4_4(midi: miditoolkit.MidiFile) -> bool:
    if hasattr(midi, "time_signature_changes") and midi.time_signature_changes:
        for ts in midi.time_signature_changes:
            if ts.numerator != 4 or ts.denominator != 4:
                return False
    return True


with open(processed_jsonl_path, "a", encoding="utf-8", newline="\n") as f_ok, \
     open(errors_jsonl_path, "a", encoding="utf-8", newline="\n") as f_err, \
     open(discarded_jsonl_path, "a", encoding="utf-8", newline="\n") as f_disc, \
     open(JSON_MANIFEST, "r", encoding="utf-8") as manifest:

    for line_str in manifest:
        line_str = line_str.strip()
        if not line_str:
            continue

        try:
            line = js.loads(line_str)
            summary["total"] += 1
        except Exception:
            summary["errors"] += 1
            bump_reason(summary, "manifest_json_parse_error")
            f_err.write(js.dumps({
                "error": "manifest_json_parse_error",
                "raw_line": line_str[:500]
            }, ensure_ascii=False) + "\n")
            continue

        song_id = line.get("song_id")
        rel_midi_path = line.get("midi_path")

        try:
            if not rel_midi_path:
                raise ValueError("missing midi_path")

            midi_abs = os.path.normpath(os.path.join(manifest_dir, rel_midi_path))
            if not os.path.isfile(midi_abs):
                raise FileNotFoundError(f"midi not found: {midi_abs}")

            midi = miditoolkit.MidiFile(midi_abs)

            old_tpb = getattr(midi, "ticks_per_beat", 0)
            if old_tpb <= 0:
                raise ValueError("invalid ticks_per_beat")

            if not is_4_4(midi):
                summary["discarded"] += 1
                bump_reason(summary, "not_4_4")
                f_disc.write(js.dumps({
                    "song_id": song_id,
                    "midi_path": rel_midi_path,
                    "reason": "not_4_4",
                }, ensure_ascii=False) + "\n")
                continue

            drum_idx = safe_int(line.get("drum_idx"))
            bass_idx = safe_int(line.get("bass_idx"))
            harm_idx = safe_int(line.get("harm_idx"))
            if drum_idx is None or bass_idx is None or harm_idx is None:
                raise ValueError("missing/invalid track index (drum_idx/bass_idx/harm_idx)")

            tracks = midi.instruments
            n_tracks = len(tracks)
            if not (0 <= drum_idx < n_tracks and 0 <= bass_idx < n_tracks and 0 <= harm_idx < n_tracks):
                raise IndexError(f"track index out of range: n_tracks={n_tracks}, drum={drum_idx}, bass={bass_idx}, harm={harm_idx}")

            scale = TARGET_TPB / old_tpb

            role_notes = {
                "drums": tracks[drum_idx].notes,
                "bass": tracks[bass_idx].notes,
                "harmony": tracks[harm_idx].notes,
            }

            loop_steps = line.get("loop_steps")
            loop_steps = safe_int(loop_steps, default=None)
            if loop_steps is None:
                loop_steps = LOOP_STEPS_CFG
            if loop_steps is None:
                loop_steps = int(STEPS_PER_BEAT * 4)

            if loop_steps <= 0:
                raise ValueError(f"invalid loop_steps={loop_steps}")

            drums_out, bass_out, harm_out = [], [], []

            for role_name, notes in role_notes.items():
                for note in notes:
                    s = int(round(note.start * scale))
                    e = int(round(note.end   * scale))
                    if e <= s:
                        e = s + 1

                    s = int(round(s / STEP_TICKS) * STEP_TICKS)
                    e = int(round(e / STEP_TICKS) * STEP_TICKS)
                    if e <= s:
                        e = s + STEP_TICKS

                    start_step = s // STEP_TICKS
                    end_step   = e // STEP_TICKS
                    if end_step <= start_step:
                        end_step = start_step + 1

                    ev = {
                        "start_step": int(start_step),
                        "end_step": int(end_step),
                        "pitch": int(note.pitch),
                        "velocity": int(note.velocity),
                    }

                    if role_name == "drums":
                        drums_out.append(ev)
                    elif role_name == "bass":
                        bass_out.append(ev)
                    else:
                        harm_out.append(ev)

            drums_out = clip_events_to_loop(drums_out, loop_steps)
            bass_out  = clip_events_to_loop(bass_out,  loop_steps)
            harm_out  = clip_events_to_loop(harm_out,  loop_steps)

            drums_out = shift_events_to_zero(drums_out)
            bass_out  = shift_events_to_zero(bass_out)
            harm_out  = shift_events_to_zero(harm_out)

            drums_out = dedup_events(drums_out)
            bass_out  = dedup_events(bass_out)
            harm_out  = dedup_events(harm_out)

            harm_out = limit_polyphony_per_step(harm_out, MAX_HARMONY_POLY)

            if (len(drums_out) + len(bass_out) + len(harm_out)) == 0:
                summary["discarded"] += 1
                bump_reason(summary, "empty_after_clipping")
                f_disc.write(js.dumps({
                    "song_id": song_id,
                    "midi_path": rel_midi_path,
                    "reason": "empty_after_clipping",
                    "loop_steps": loop_steps
                }, ensure_ascii=False) + "\n")
                continue

            genre     = line.get("genre", "unknown")
            bpm_bucket = line.get("bpm_bucket")
            if bpm_bucket is None:
                raw_bpm = float(line.get("bpm", 120))
                if raw_bpm > 200:
                    raw_bpm /= 2.0
                raw_bpm = max(60.0, min(200.0, raw_bpm))
                bpm_bucket = int(round(raw_bpm / 10.0) * 10)

            detected_key = detect_key_from_events(bass_out, harm_out)

            user_prompt = (
                f"<GENRE_{genre}> <BPM_{int(bpm_bucket)}> <KEY_{detected_key}> <LEN_{int(loop_steps)}>\n"
                f"{INSTRUCTION}"
            )

            roles_dict = {"drums": drums_out, "bass": bass_out, "harmony": harm_out}
            assistant_tokens = roles_to_tokens(roles_dict)
            assistant_out = "<LOOP>\n" + assistant_tokens + "\n</LOOP>"

            row = {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_out}
                ],
                "meta": {
                    "song_id": song_id,
                    "bpm": line.get("bpm"),
                    "bpm_bucket": bpm_bucket,
                    "genre": genre,
                    "key": detected_key,
                    "tpb": TARGET_TPB,
                    "grid": QUANT_GRID,
                    "step_ticks": STEP_TICKS,
                    "loop_steps": loop_steps,
                    "max_harmony_polyphony": MAX_HARMONY_POLY,
                    "source_midi_path": rel_midi_path,
                }
            }

            f_ok.write(js.dumps(row, ensure_ascii=False) + "\n")
            summary["ok"] += 1

        except Exception as e:
            summary["errors"] += 1
            bump_reason(summary, "exception")
            f_err.write(js.dumps({
                "song_id": song_id,
                "midi_path": rel_midi_path,
                "error": str(e),
            }, ensure_ascii=False) + "\n")
            continue

with open(summary_json_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(js.dumps(summary, ensure_ascii=False, indent=2) + "\n")