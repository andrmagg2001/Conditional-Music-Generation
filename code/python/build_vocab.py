import json as js


def build_vocab(
    train_path: str = "data/json/processed/train.jsonl",
    vocab_path: str = "data/json/vocab.json",
) -> None:
    """Build a deterministic vocabulary from a JSONL training split."""
    special_tokens: list[str] = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    token_set: set[str] = set()
    total_rows: int = 0
    malformed_rows: int = 0

    with open(train_path, "r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            total_rows += 1
            try:
                obj = js.loads(line)
                prompt = obj["messages"][0]["content"]
                answer = obj["messages"][1]["content"]
                text = f"<BOS> {prompt}\n{answer} <EOS>"
                token_set.update(text.split())
            except (js.JSONDecodeError, KeyError, IndexError, TypeError):
                malformed_rows += 1
                print(f"[WARN] Skipping malformed row at index {row_idx}")

    token_set -= set(special_tokens)
    ordered_tokens: list[str] = sorted(token_set)

    token_to_id: dict[str, int] = {tok: idx for idx, tok in enumerate(special_tokens)}
    next_id: int = len(token_to_id)
    for tok in ordered_tokens:
        token_to_id[tok] = next_id
        next_id += 1

    id_to_token: dict[str, str] = {str(v): k for k, v in token_to_id.items()}

    payload: dict[str, object] = {
        "vocab_version": 1,
        "special_tokens": special_tokens,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "stats": {
            "rows_seen": total_rows,
            "rows_malformed": malformed_rows,
            "vocab_size": len(token_to_id),
        },
    }

    with open(vocab_path, "w", encoding="utf-8") as f:
        js.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved vocab to: {vocab_path}")
    print(f"[OK] Vocab size: {len(token_to_id)}")
    print(f"[OK] Rows seen: {total_rows} | Malformed rows: {malformed_rows}")


build_vocab()