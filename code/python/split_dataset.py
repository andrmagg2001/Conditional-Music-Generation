import json as js
import random
import yaml

YAML_PATH = "code/python/preprocess.yaml"
PROCESSED_DIR = ""
SEED = 0
SPLIT_RATIO = {}
BY_SONG = {}

with open(YAML_PATH, "r", encoding="utf-8") as stream:
    temp = yaml.safe_load(stream)
    PROCESSED_DIR = temp["paths"]["output_json_dir"]
    SEED = int(temp["dataset"]["seed"])
    SPLIT_RATIO = temp["dataset"]["split"]

rng = random.Random(SEED)
processed_path = f"{PROCESSED_DIR}/processed.jsonl"

row_total = 0
rows_invalid = 0

with open(processed_path, "r", encoding="utf-8") as j_pro:
    for line in j_pro:
        line = line.strip()
        if not line:
            continue

        row_total += 1

        try:
            row = js.loads(line)
        except Exception:
            rows_invalid += 1
            continue

        song_id = row.get("meta", {}).get("song_id")
        if not song_id:
            rows_invalid += 1
            continue

        BY_SONG.setdefault(song_id, []).append(row)

print(f"Total rows: {row_total}")
print(f"Invalid rows: {rows_invalid}")
print(f"Unique song_ids: {len(BY_SONG)}")

song_ids = list(BY_SONG.keys())
rng.shuffle(song_ids)

n = len(song_ids)
n_train = int(n * SPLIT_RATIO["train"])
n_val = int(n * SPLIT_RATIO["val"])

train_ids = set(song_ids[:n_train])
val_ids = set(song_ids[n_train:n_train + n_val])
test_ids = set(song_ids[n_train + n_val:])

train_rows = [row for sid in train_ids for row in BY_SONG[sid]]
val_rows = [row for sid in val_ids for row in BY_SONG[sid]]
test_rows = [row for sid in test_ids for row in BY_SONG[sid]]

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(js.dumps(row, ensure_ascii=False) + "\n")

train_path = f"{PROCESSED_DIR}/train.jsonl"
val_path = f"{PROCESSED_DIR}/val.jsonl"
test_path = f"{PROCESSED_DIR}/test.jsonl"

write_jsonl(train_path, train_rows)
write_jsonl(val_path, val_rows)
write_jsonl(test_path, test_rows)

print(f"Train songs: {len(train_ids)} | rows: {len(train_rows)}")
print(f"Val songs:   {len(val_ids)} | rows: {len(val_rows)}")
print(f"Test songs:  {len(test_ids)} | rows: {len(test_rows)}")

print(f"Leak train∩val:  {len(train_ids & val_ids)}")
print(f"Leak train∩test: {len(train_ids & test_ids)}")
print(f"Leak val∩test:   {len(val_ids & test_ids)}")

print(f"Rows check: {len(train_rows) + len(val_rows) + len(test_rows) + rows_invalid} == {row_total}")