import argparse
from contextlib import nullcontext
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from dataset_loader import make_dataloaders


@dataclass
class TrainConfig:
    train_path: str = "data/json/processed/train.jsonl"
    val_path: str = "data/json/processed/val.jsonl"
    test_path: str = "data/json/processed/test.jsonl"
    vocab_path: str = "data/json/vocab.json"
    cache_dir: str = "data/json/processed/cache_tokenized"
    output_dir: str = "data/checkpoints/baseline_transformer"
    seed: int = 42
    batch_size: int = 8
    max_len: int = 512
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    grad_accum_steps: int = 4
    warmup_ratio: float = 0.05
    epochs: int = 2
    max_steps: int = 300
    eval_every: int = 50
    save_every: int = 100
    eval_batches: int = 40
    num_workers: int = 0
    pin_memory: bool = False
    use_amp: bool = True
    resume_path: str | None = None
    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-4
    oom_skip_batch: bool = True
    max_consecutive_oom_skips: int = 200
    log_every: int = 1
    eval_tqdm: bool = False


class BaselineTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds model max_len {self.max_len}")

        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embed(input_ids) + self.pos_embed(pos_ids)
        x = self.drop(x)

        causal_mask = self._causal_mask(seq_len, input_ids.device)
        key_padding_mask = attention_mask == 0
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        x = self.ln_f(x)
        return self.lm_head(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_vocab(vocab_path: str) -> tuple[int, int]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    token_to_id = payload.get("token_to_id", {})
    if "<PAD>" not in token_to_id:
        raise ValueError("<PAD> token not found in vocabulary")
    return len(token_to_id), int(token_to_id["<PAD>"])


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def apply_device_profile(cfg: TrainConfig, args: argparse.Namespace, device: torch.device) -> None:
    """Apply an aggressive configuration profile for Apple MPS with large VRAM."""
    if device.type != "mps":
        return

    if args.batch_size is None:
        cfg.batch_size = 6
    if args.max_len is None:
        cfg.max_len = 256
    if args.d_model is None:
        cfg.d_model = 256
    if args.layers is None:
        cfg.n_layers = 6
    if args.heads is None:
        cfg.n_heads = 8
    if args.grad_accum is None:
        cfg.grad_accum_steps = 12
    if args.max_steps is None:
        cfg.max_steps = 1500
    if args.epochs is None:
        cfg.epochs = 3
    if args.eval_every is None:
        cfg.eval_every = 120
    if args.save_every is None:
        cfg.save_every = 240

    cfg.d_ff = cfg.d_model * 4
    cfg.lr = 3e-4
    cfg.min_lr = 3e-5
    cfg.eval_batches = 48


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith("bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


def is_oom_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "mps backend out of memory" in message


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def restore_torch_rng_state(rng_state: Any) -> None:
    """Restore torch RNG state from multiple historical serialization formats."""
    if rng_state is None:
        return

    try:
        if isinstance(rng_state, torch.Tensor):
            torch.set_rng_state(rng_state.detach().cpu().contiguous().to(dtype=torch.uint8).clone())
            return

        if isinstance(rng_state, (list, tuple)):
            torch.set_rng_state(torch.ByteTensor(list(rng_state)))
            return

        if isinstance(rng_state, (bytes, bytearray)):
            torch.set_rng_state(torch.ByteTensor(list(rng_state)))
            return

        torch.set_rng_state(rng_state)
    except Exception as exc:
        print(f"[WARN] Could not restore torch RNG state from checkpoint: {exc}")


def autocast_context(device: torch.device, enabled: bool):
    """Use AMP only where the runtime supports it (CUDA in this environment)."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader,
    device: torch.device,
    pad_id: int,
    max_batches: int,
    amp_enabled: bool,
    show_progress: bool,
    oom_skip_batch: bool,
) -> float:
    model.eval()
    losses = []
    skipped_oom_batches = 0

    eval_iter = data_loader
    if show_progress:
        eval_iter = tqdm(data_loader, total=min(len(data_loader), max_batches), desc="Validation", leave=False)

    for batch_idx, batch in enumerate(eval_iter):
        if batch_idx >= max_batches:
            break

        try:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with autocast_context(device, amp_enabled):
                logits = model(input_ids, attention_mask)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                    ignore_index=pad_id,
                )
        except RuntimeError as exc:
            if oom_skip_batch and is_oom_error(exc):
                clear_device_cache(device)
                skipped_oom_batches += 1
                if skipped_oom_batches <= 3 or skipped_oom_batches % 10 == 0:
                    print(f"[WARN] OOM batch skipped during evaluation (count={skipped_oom_batches}).")
                continue
            raise
        losses.append(float(loss.item()))

    model.train()
    if not losses:
        print("[WARN] No valid evaluation batches processed.")
        return float("nan")
    return float(sum(losses) / len(losses))


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    global_step: int,
    epoch: int,
    best_val_loss: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": asdict(config),
        "global_step": global_step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[int, int, float]:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])

    scaler_state = payload.get("scaler")
    if amp_enabled and scaler_state:
        scaler.load_state_dict(scaler_state)

    if "python_rng_state" in payload:
        random.setstate(payload["python_rng_state"])
    if "numpy_rng_state" in payload:
        np.random.set_state(payload["numpy_rng_state"])
    if "torch_rng_state" in payload:
        restore_torch_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

    global_step = int(payload.get("global_step", 0))
    epoch = int(payload.get("epoch", 0))
    best_val_loss = float(payload.get("best_val_loss", float("inf")))
    return global_step, epoch, best_val_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline causal Transformer LM")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=None)
    parser.add_argument("--oom-skip-batch", action="store_true")
    parser.add_argument("--max-consecutive-oom-skips", type=int, default=None)
    parser.add_argument("--eval-tqdm", action="store_true")
    parser.add_argument("--log-every", type=int, default=None)
    return parser.parse_args()


def save_run_artifacts(
    output_dir: str,
    train_steps: list[int],
    train_losses: list[float],
    val_steps: list[int],
    val_losses: list[float],
    best_val_loss: float,
    test_loss: float,
    final_step: int,
    final_epoch: int,
) -> None:
    summary = {
        "final_step": final_step,
        "final_epoch": final_epoch,
        "num_train_points": len(train_steps),
        "num_val_points": len(val_steps),
        "last_train_loss": train_losses[-1] if train_losses else None,
        "last_val_loss": val_losses[-1] if val_losses else None,
        "best_val_loss": None if math.isinf(best_val_loss) else best_val_loss,
        "test_loss": test_loss,
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Could not generate plots (matplotlib unavailable): {exc}")
        return

    if train_steps and train_losses:
        plt.figure(figsize=(8, 4.5))
        plt.plot(train_steps, train_losses, label="train_loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Train Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "train_loss.png"), dpi=150)
        plt.close()

    if val_steps and val_losses:
        plt.figure(figsize=(8, 4.5))
        plt.plot(val_steps, val_losses, label="val_loss", color="tab:orange")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "val_loss.png"), dpi=150)
        plt.close()

    if train_steps and train_losses and val_steps and val_losses:
        plt.figure(figsize=(8, 4.5))
        plt.plot(train_steps, train_losses, label="train_loss")
        plt.plot(val_steps, val_losses, label="val_loss", color="tab:orange")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Train vs Val Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
        plt.close()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig()
    device = pick_device()
    apply_device_profile(cfg, args, device)

    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.eval_every is not None:
        cfg.eval_every = args.eval_every
    if args.eval_batches is not None:
        cfg.eval_batches = args.eval_batches
    if args.save_every is not None:
        cfg.save_every = args.save_every
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.max_len is not None:
        cfg.max_len = args.max_len
    if args.d_model is not None:
        cfg.d_model = args.d_model
    if args.layers is not None:
        cfg.n_layers = args.layers
    if args.heads is not None:
        cfg.n_heads = args.heads
    if args.grad_accum is not None:
        cfg.grad_accum_steps = args.grad_accum
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.resume is not None:
        cfg.resume_path = args.resume
    if args.early_stop_patience is not None:
        cfg.early_stop_patience = args.early_stop_patience
    if args.early_stop_min_delta is not None:
        cfg.early_stop_min_delta = args.early_stop_min_delta
    if args.oom_skip_batch:
        cfg.oom_skip_batch = True
    if args.max_consecutive_oom_skips is not None:
        cfg.max_consecutive_oom_skips = max(1, args.max_consecutive_oom_skips)
    if args.eval_tqdm:
        cfg.eval_tqdm = True
    if args.log_every is not None:
        cfg.log_every = max(1, args.log_every)

    cfg.d_ff = cfg.d_model * 4
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    vocab_size, pad_id = load_vocab(cfg.vocab_path)
    print(f"[INFO] device={device} | vocab_size={vocab_size} | pad_id={pad_id}")
    print(
        "[INFO] config="
        f"batch={cfg.batch_size}, max_len={cfg.max_len}, d_model={cfg.d_model}, "
        f"layers={cfg.n_layers}, heads={cfg.n_heads}, d_ff={cfg.d_ff}, "
        f"grad_accum={cfg.grad_accum_steps}, lr={cfg.lr}"
    )

    loaders = make_dataloaders(
        train_path=cfg.train_path,
        val_path=cfg.val_path,
        test_path=cfg.test_path,
        batch_size=cfg.batch_size,
        max_len=cfg.max_len,
        seed=cfg.seed,
        cache_dir=cfg.cache_dir,
        vocab_path=cfg.vocab_path,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    model = BaselineTransformerLM(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_len=cfg.max_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = build_optimizer(model, cfg)

    updates_per_epoch = max(1, len(train_loader) // max(1, cfg.grad_accum_steps))
    total_updates = min(cfg.max_steps, cfg.epochs * updates_per_epoch)
    warmup_steps = max(1, int(cfg.warmup_ratio * total_updates))
    min_lr_ratio = cfg.min_lr / cfg.lr
    scheduler = make_scheduler(optimizer, warmup_steps, total_updates, min_lr_ratio)

    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    log_path = os.path.join(cfg.output_dir, "metrics.jsonl")
    latest_ckpt = os.path.join(cfg.output_dir, "latest.pth")
    best_ckpt = os.path.join(cfg.output_dir, "best.pth")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    best_val_loss = float("inf")
    start_epoch = 1
    no_improve_evals = 0
    stop_training = False
    running_loss = 0.0
    running_tokens = 0
    window_start = time.time()
    train_steps: list[int] = []
    train_losses: list[float] = []
    val_steps: list[int] = []
    val_losses: list[float] = []
    consecutive_oom_skips = 0

    if cfg.resume_path is not None:
        resume_path = cfg.resume_path
        if resume_path == "latest":
            resume_path = latest_ckpt
        if not os.path.isfile(resume_path):
            if cfg.resume_path == "latest":
                print(f"[WARN] Resume requested but checkpoint missing: {resume_path}. Starting fresh run.")
            else:
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        else:
            global_step, resumed_epoch, best_val_loss = load_checkpoint(
                resume_path,
                model,
                optimizer,
                scheduler,
                scaler,
                device,
                amp_enabled,
            )
            start_epoch = max(1, resumed_epoch)
            print(f"[INFO] resumed from {resume_path} at step={global_step}, epoch={resumed_epoch}")

    log_mode = "a" if (cfg.resume_path is not None and os.path.isfile(log_path)) else "w"
    with open(log_path, log_mode, encoding="utf-8") as log_file:
        for epoch in range(start_epoch, cfg.epochs + 1):
            epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
            for batch_idx, batch in enumerate(epoch_bar, start=1):
                try:
                    input_ids = batch["input_ids"].to(device)
                    target_ids = batch["target_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)

                    with autocast_context(device, amp_enabled):
                        logits = model(input_ids, attention_mask)
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            target_ids.view(-1),
                            ignore_index=pad_id,
                        )
                        scaled_loss = loss / cfg.grad_accum_steps

                    if torch.isnan(loss) or torch.isinf(loss):
                        raise RuntimeError("Loss became NaN or Inf, stopping training")

                    scaler.scale(scaled_loss).backward()
                except RuntimeError as exc:
                    if cfg.oom_skip_batch and is_oom_error(exc):
                        optimizer.zero_grad(set_to_none=True)
                        clear_device_cache(device)
                        consecutive_oom_skips += 1
                        if consecutive_oom_skips <= 3 or consecutive_oom_skips % 10 == 0:
                            print(
                                "[WARN] OOM batch skipped "
                                f"(consecutive={consecutive_oom_skips}). "
                                "Consider lowering --batch-size or --max-len."
                            )
                        if consecutive_oom_skips >= cfg.max_consecutive_oom_skips:
                            print(
                                "[EARLY STOP] Too many consecutive OOM skips "
                                f"({consecutive_oom_skips}), stopping training."
                            )
                            stop_training = True
                            break
                        continue
                    raise

                consecutive_oom_skips = 0
                running_loss += float(loss.item())
                running_tokens += int(attention_mask.sum().item())

                if batch_idx % cfg.grad_accum_steps != 0:
                    continue

                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item())
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                global_step += 1
                elapsed = max(1e-8, time.time() - window_start)
                train_loss = running_loss / cfg.grad_accum_steps
                toks_per_sec = running_tokens / elapsed
                current_lr = float(optimizer.param_groups[0]["lr"])

                event = {
                    "step": global_step,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "lr": current_lr,
                    "grad_norm": grad_norm,
                    "tokens_per_sec": toks_per_sec,
                }
                train_steps.append(global_step)
                train_losses.append(train_loss)
                log_file.write(json.dumps(event) + "\n")
                log_file.flush()

                if global_step % cfg.log_every == 0:
                    print(
                        f"[TRAIN] step={global_step:04d} epoch={epoch} "
                        f"loss={train_loss:.4f} lr={current_lr:.6f} "
                        f"grad_norm={grad_norm:.3f} tok/s={toks_per_sec:.0f}"
                    )
                epoch_bar.set_postfix(
                    step=global_step,
                    loss=f"{train_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                )

                running_loss = 0.0
                running_tokens = 0
                window_start = time.time()

                if global_step % cfg.eval_every == 0:
                    val_loss = evaluate(
                        model=model,
                        data_loader=val_loader,
                        device=device,
                        pad_id=pad_id,
                        max_batches=cfg.eval_batches,
                        amp_enabled=amp_enabled,
                        show_progress=cfg.eval_tqdm,
                        oom_skip_batch=cfg.oom_skip_batch,
                    )
                    print(f"[VAL] step={global_step:04d} val_loss={val_loss:.4f}")
                    eval_event = {
                        "step": global_step,
                        "epoch": epoch,
                        "val_loss": val_loss,
                    }
                    val_steps.append(global_step)
                    val_losses.append(val_loss)
                    log_file.write(json.dumps(eval_event) + "\n")
                    log_file.flush()

                    improvement = best_val_loss - val_loss
                    if improvement > cfg.early_stop_min_delta:
                        best_val_loss = val_loss
                        no_improve_evals = 0
                        save_checkpoint(
                            best_ckpt,
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            cfg,
                            global_step,
                            epoch,
                            best_val_loss,
                        )
                        print(f"[CKPT] best updated at step={global_step:04d}")
                    else:
                        no_improve_evals += 1
                        if no_improve_evals >= cfg.early_stop_patience:
                            print("[EARLY STOP] validation did not improve, stopping training.")
                            stop_training = True

                if global_step % cfg.save_every == 0:
                    save_checkpoint(
                        latest_ckpt,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        cfg,
                        global_step,
                        epoch,
                        best_val_loss,
                    )
                    print(f"[CKPT] latest saved at step={global_step:04d}")

                if global_step >= cfg.max_steps:
                    stop_training = True
                    break

            if stop_training:
                break

    clear_device_cache(device)
    test_loss = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        pad_id=pad_id,
        max_batches=cfg.eval_batches,
        amp_enabled=amp_enabled,
        show_progress=cfg.eval_tqdm,
        oom_skip_batch=cfg.oom_skip_batch,
    )
    print(f"[TEST] loss={test_loss:.4f}")

    save_checkpoint(
        latest_ckpt,
        model,
        optimizer,
        scheduler,
        scaler,
        cfg,
        global_step,
        epoch,
        best_val_loss,
    )
    save_run_artifacts(
        output_dir=cfg.output_dir,
        train_steps=train_steps,
        train_losses=train_losses,
        val_steps=val_steps,
        val_losses=val_losses,
        best_val_loss=best_val_loss,
        test_loss=test_loss,
        final_step=global_step,
        final_epoch=epoch,
    )
    print(f"[DONE] training completed at step={global_step}")
    print(f"[DONE] latest checkpoint: {latest_ckpt}")
    print(f"[DONE] run artifacts in: {cfg.output_dir}")
    if os.path.isfile(best_ckpt):
        print(f"[DONE] best checkpoint:   {best_ckpt}")


if __name__ == "__main__":
    main()

