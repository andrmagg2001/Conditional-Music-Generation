import torch
import json as js
import os
from typing import Any
from torch.utils.data import DataLoader


class Dataset:
    def __init__(
        self,
        path_js: str,
        max_len: int | None = None,
        cache_path: str | None = None,
        vocab_path: str = "data/json/vocab.json"
    ):
        self.path_js = path_js
        self.max_len = max_len
        self.cache_path = cache_path
        self.rows: list[str] = []
        self.samples: list[dict[str, Any]] | None = None

        with open(vocab_path, "r", encoding="utf-8") as vocab_file:
            vocab_payload = js.load(vocab_file)

        self.token_to_id: dict[str, int] = vocab_payload.get("token_to_id", {})
        if not isinstance(self.token_to_id, dict) or not self.token_to_id:
            raise ValueError(f"Invalid or empty token_to_id in vocab file: {vocab_path}")

        if "<UNK>" not in self.token_to_id:
            raise ValueError(f"Missing required token '<UNK>' in vocab file: {vocab_path}")
        if "<PAD>" not in self.token_to_id:
            raise ValueError(f"Missing required token '<PAD>' in vocab file: {vocab_path}")

        self.unk_id: int = int(self.token_to_id["<UNK>"])
        self.pad_id: int = int(self.token_to_id["<PAD>"])
        

        with open(self.path_js, "r", encoding="utf-8") as js_file:
            self.rows = [line.strip() for line in js_file if line.strip()]

        if not self.rows:
            raise ValueError(f"No valid row in: {self.path_js}")

        if self.cache_path is not None:
            self.samples = self._load_or_build_cache(self.cache_path)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self.samples is not None:
            return len(self.samples)
        return len(self.rows)

    def _build_sample_from_row(self, line: str, idx: int) -> dict[str, Any]:
        """Parse and tokenize one JSONL row into a training sample."""
        obj = js.loads(line)
        prompt = obj["messages"][0]["content"]
        answer = obj["messages"][1]["content"]
        song_id = obj["meta"]["song_id"]

        if not prompt or not answer or not song_id:
            raise ValueError("Empty required field")

        text = f"<BOS> {prompt}\n{answer} <EOS>"
        token_ids = self._encode_text(text)
        if len(token_ids) < 2:
            raise ValueError("Tokenized sample is too short (<2 tokens)")

        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "song_id": song_id,
        }

    def _load_or_build_cache(self, cache_path: str) -> list[dict[str, Any]]:
        """
        Cache v1 behavior:
        - If cache file exists, load it (HIT).
        - Otherwise build samples and save them (MISS).
        """
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        if os.path.isfile(cache_path):
            print(f"[CACHE HIT] {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as cache_file:
                payload = js.load(cache_file)
            samples = payload.get("samples", [])
            if not isinstance(samples, list):
                raise ValueError(f"Invalid cache format in: {cache_path}")
            return samples

        print(f"[CACHE MISS] {cache_path}")
        samples: list[dict[str, Any]] = []
        for idx, line in enumerate(self.rows):
            try:
                sample = self._build_sample_from_row(line, idx)
                samples.append(sample)
            except (js.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
                raise ValueError(f"Malformed sample at index {idx}: {exc}") from exc

        payload = {
            "cache_version": 1,
            "source_path": self.path_js,
            "max_len": self.max_len,
            "num_samples": len(samples),
            "samples": samples,
        }
        with open(cache_path, "w", encoding="utf-8") as cache_file:
            js.dump(payload, cache_file, ensure_ascii=False)

        return samples

    def _encode_text(self, text: str) -> list[int]:
        """Encode a text string into token ids using the loaded vocabulary."""

        tokens = text.split()
        token_ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokens]

        if self.max_len is not None and self.max_len > 0:
            token_ids = token_ids[: self.max_len]

        token_ids = [int(x) for x in token_ids]
        
        return token_ids

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Return one training sample from a single JSONL row.

        Output fields:
        - input_ids: token ids shifted right (all except last)
        - target_ids: token ids shifted left (all except first)
        - song_id: sample identifier from metadata
        """
        if self.samples is not None:
            return self.samples[idx]

        line = self.rows[idx]
        try:
            return self._build_sample_from_row(line, idx)

        except (js.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed sample at index {idx}: {exc}") from exc


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function to pad input_ids and target_ids in a batch.

    Pads sequences to the maximum length in the batch using 0 as the padding token.
    """
    pad_id: int = 0

    padded_inputs: list[list[int]] = []
    padded_targets: list[list[int]] = []
    attention_masks: list[list[int]] = []

    input_ids_list: list[list[int]] = [s["input_ids"] for s in batch]
    target_ids_list: list[list[int]] = [s["target_ids"] for s in batch]
    song_ids_list: list[str] = [s["song_id"] for s in batch]

    max_len: int = max(len(ids) for ids in input_ids_list)

    for inp, tgt in zip(input_ids_list, target_ids_list):
        cur_len: int = len(inp)
        pad_len: int = max_len - cur_len

        padded_inp: list[int] = inp + [pad_id] * pad_len
        padded_tgt: list[int] = tgt + [pad_id] * pad_len
        mask: list[int] = [1] * cur_len + [0] * pad_len

        padded_inputs.append(padded_inp)
        padded_targets.append(padded_tgt)
        attention_masks.append(mask)

    return {
        "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
        "target_ids": torch.tensor(padded_targets, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "song_ids": song_ids_list,
    }


def make_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int,
    max_len: int | None,
    seed: int,
    cache_dir: str | None = None,
    vocab_path: str = "data/json/vocab.json",
) -> dict[str, DataLoader]:
    """
    Build train/validation/test DataLoaders with deterministic train shuffling.

    Args:
        train_path: Path to train JSONL file.
        val_path: Path to validation JSONL file.
        test_path: Path to test JSONL file.
        batch_size: Batch size for all splits.
        max_len: Optional max token length per sample.
        seed: Random seed for reproducible train shuffling.
        cache_dir: Optional directory where split caches are stored.
        vocab_path: Path to vocabulary json file.

    Returns:
        dict[str, DataLoader]: Mapping with keys "train", "val", "test".
    """
    cache_root: str = cache_dir or os.path.join(os.path.dirname(train_path), "cache_tokenized")
    os.makedirs(cache_root, exist_ok=True)

    def split_cache_path(split_path: str) -> str:
        split_name: str = os.path.splitext(os.path.basename(split_path))[0]
        len_tag: str = "none" if max_len is None else str(max_len)
        return os.path.join(cache_root, f"{split_name}.maxlen-{len_tag}.cache.json")

    train_ds: Dataset = Dataset(
        train_path,
        max_len,
        cache_path=split_cache_path(train_path),
        vocab_path=vocab_path,
    )
    val_ds: Dataset = Dataset(
        val_path,
        max_len,
        cache_path=split_cache_path(val_path),
        vocab_path=vocab_path,
    )
    test_ds: Dataset = Dataset(
        test_path,
        max_len,
        cache_path=split_cache_path(test_path),
        vocab_path=vocab_path,
    )

    train_gen: torch.Generator = torch.Generator()
    train_gen.manual_seed(seed)

    train_loader: DataLoader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=train_gen,
        collate_fn=collate_fn,
    )
    val_loader: DataLoader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader: DataLoader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    