import os
import json as js

os.makedirs("data/json/reports", exist_ok=True)

with open("data/json/vocab.json", "r") as f:
    vocab_data = js.load(f)

token_to_id = vocab_data["token_to_id"]
id_to_token = vocab_data["id_to_token"]
unk_id      = token_to_id["<UNK>"]
pad_id      = token_to_id["<PAD>"]

print(f"Vocab size: {len(token_to_id)}")
print(f"<UNK> ID: {unk_id}, <PAD> ID: {pad_id}")

def encode(text: str) -> list[int]:
    """Tokenize text in token IDs (like in the dataset_loader)"""
    tokens = text.split()
    ids = [token_to_id.get(tok, unk_id) for tok in tokens]
    return ids

def decode(ids: list[int]) -> str:
    """convert id in text"""
    tokens = [id_to_token.get(str(id_val), "<UNK>") for id_val in ids]
    return " ".join(tokens)

def normalize(text: str) -> str:
    """Normalize the text for the comparison"""
    return " ".join(text.split())

def classify_issue(original_text: str, roundtrip_text: str, ids: list[int]) -> dict:
    """
    Classify the edge case type.
    """
    orig_norm = normalize(original_text)
    rt_norm   = normalize(roundtrip_text)

    if orig_norm == rt_norm:
        return {"category": "PASS", "reason": None}
    
    unk_count = rt_norm.count("<UNK>")
    if unk_count > 0:
        return {
            "category": "UNKNOWN_TOKENS",
            "reason": f"{unk_count} token sconosciuti nel risultato",
            "unk_count": unk_count
        }
    
    orig_has_newline = "\n" in original_text
    if orig_has_newline and "\n" not in roundtrip_text:
        return {
            "category": "NEWLINE_COLLAPSE",
            "reason": "Newline è diventato spazio (split/join behavior)"
        }
    
    return {
        "category": "OTHER_MISMATCH",
        "reason": f"Text differs: '{orig_norm[:50]}' vs '{rt_norm[:50]}'"
    }



subset_path = "data/json/processed/train_roundtrip_subset_500.jsonl"
with open(subset_path, "r") as f:
    subset_lines = [line.strip() for line in f if line.strip()]

print(f"\nProcessing {len(subset_lines)} samples...\n")

results = {
    "total": len(subset_lines),
    "pass": 0,
    "fail": 0,
    "by_category": {}
}

edge_cases_log = []

for idx, line in enumerate(subset_lines):
    try:
        obj = js.loads(line)
        prompt = obj["messages"][0]["content"]
        answer = obj["messages"][1]["content"]
        song_id = obj["meta"]["song_id"]
        
        text = f"<BOS> {prompt}\n{answer} <EOS>"
        
        ids = encode(text)
        roundtrip_text = decode(ids)
        
        issue = classify_issue(text, roundtrip_text, ids)
        
        if issue["category"] == "PASS":
            results["pass"] += 1
        else:
            results["fail"] += 1
            cat = issue["category"]
            results["by_category"][cat] = results["by_category"].get(cat, 0) + 1
            
            edge_cases_log.append({
                "sample_idx": idx,
                "song_id": song_id,
                "category": cat,
                "reason": issue["reason"],
                "details": issue.get("details", {}),
                "original_len": len(ids),
                "unk_count": issue.get("unk_count", 0)
            })
    
    except Exception as e:
        results["fail"] += 1
        results["by_category"]["MALFORMED"] = results["by_category"].get("MALFORMED", 0) + 1
        edge_cases_log.append({
            "sample_idx": idx,
            "song_id": "unknown",
            "category": "MALFORMED",
            "reason": str(e)
        })

print(f"Results: {results['pass']} PASS, {results['fail']} FAIL")
print(f"Breakdown: {results['by_category']}")

summary_path = "data/json/reports/roundtrip_summary_500.json"
with open(summary_path, "w") as f:
    js.dump(results, f, indent=2)
print(f"\nSummary saved: {summary_path}")

edge_cases_path = "data/json/reports/roundtrip_edge_cases_500.jsonl"
with open(edge_cases_path, "w") as f:
    for item in edge_cases_log:
        f.write(js.dumps(item) + "\n")
print(f"Edge cases saved: {edge_cases_path}")