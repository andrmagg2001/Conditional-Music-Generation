import argparse
from pathlib import Path
import torch

from train_baseline import BaselineTransformerLM, load_vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="data/checkpoints/conditioned_transformer/best.pth")
    parser.add_argument("--out", type=str, default="data/checkpoints/conditioned_transformer/best.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    state = ckpt["model"]

    vocab_size, pad_id = load_vocab(cfg["vocab_path"])

    model = BaselineTransformerLM(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_len=cfg["max_len"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    bsz = 1
    seq_len = min(args.seq_len, cfg["max_len"])
    input_ids = torch.randint(low=0, high=vocab_size, size=(bsz, seq_len), dtype=torch.long)
    attention_mask = torch.ones((bsz, seq_len), dtype=torch.long)

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        args.out,
        export_params=True,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=args.opset,
    )

    print("Saved ONNX:", args.out)


if __name__ == "__main__":
    main()