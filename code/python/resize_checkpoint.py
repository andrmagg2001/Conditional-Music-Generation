import argparse
import json
import pathlib
import torch


def main():
    parser = argparse.ArgumentParser(description="Resize checkpoint for new vocab")
    parser.add_argument("--old-ckpt", type=str, required=True)
    parser.add_argument("--new-vocab", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.old_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["model"]

    with open(args.new_vocab, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    new_vocab_size = vocab["stats"]["vocab_size"]

    old_embed = sd["token_embed.weight"]
    old_vocab_size, d_model = old_embed.shape

    print(f"Old vocab: {old_vocab_size}, New vocab: {new_vocab_size}, d_model: {d_model}")

    if new_vocab_size == old_vocab_size:
        print("Vocab size unchanged, no resize needed.")
        return

    if new_vocab_size < old_vocab_size:
        print("WARNING: new vocab is smaller, truncating embeddings.")
        sd["token_embed.weight"] = old_embed[:new_vocab_size]
        sd["lm_head.weight"] = sd["lm_head.weight"][:new_vocab_size]
    else:
        n_new = new_vocab_size - old_vocab_size
        print(f"Adding {n_new} new token embeddings (randomly initialized)")

        new_rows = torch.randn(n_new, d_model) * 0.02
        sd["token_embed.weight"] = torch.cat([old_embed, new_rows], dim=0)

        old_lm = sd["lm_head.weight"]
        new_lm_rows = torch.randn(n_new, d_model) * 0.02
        sd["lm_head.weight"] = torch.cat([old_lm, new_lm_rows], dim=0)

    ckpt["model"] = sd

    if "optimizer" in ckpt:
        del ckpt["optimizer"]
        print("Removed optimizer state (will be re-initialized)")
    if "scheduler" in ckpt:
        del ckpt["scheduler"]
        print("Removed scheduler state (will be re-initialized)")
    if "scaler" in ckpt:
        del ckpt["scaler"]

    ckpt["global_step"] = 0
    ckpt["epoch"] = 0
    ckpt["best_val_loss"] = float("inf")
    print("Reset training counters (step=0, epoch=0)")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(out_path))
    print(f"Saved resized checkpoint to: {out_path}")
    print(f"  token_embed: {sd['token_embed.weight'].shape}")
    print(f"  lm_head:     {sd['lm_head.weight'].shape}")


if __name__ == "__main__":
    main()
