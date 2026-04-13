import random
import json as js

random.seed(42)

def extract_subset(path : str = "data/json/processed/train.jsonl", subset_num : int = 500):
    
    with open(path, "r") as f:
        all_rows = [line.strip() for line in f if line.strip()]
    print(f"Total Rows: {len(all_rows)}")

    if len(all_rows) < subset_num:
        print("Subset is major of len of all rows")
        exit(1)
        
    subset = random.sample(all_rows, subset_num)

    with open("data/json/processed/train_roundtrip_subset_500.jsonl", "w") as f:
        for line in subset:
            f.write(f"{line}\n")

    metadata = {
        "source": "data/json/processed/train.jsonl",
        "subset_size": subset_num,
        "seed": 42,
        "total_rows_source": len(all_rows)
    }

    with open("data/json/processed/train_roundtrip_subset_500.metadata.json", "w") as f:
        js.dump(metadata, f, indent=2)

    print(f"Extracted {subset_num} rows → train_roundtrip_subset_500_seed42.jsonl")




extract_subset()