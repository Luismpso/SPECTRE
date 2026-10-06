"""Train the phase-1 closed-set speaker classifier.

    python -m spectre.train --config configs/baseline.yaml
    python -m spectre.train --manifest data/manifests/dev-clean.csv --epochs 5
    python -m spectre.train --resume runs/<run>/last.pt          # continue an interrupted run

A full training state (weights, optimiser, scheduler, RNG, history) is saved to
runs/<run>/last.pt after every epoch, so Ctrl+C at any point loses at most the
epoch in progress.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SpeakerDataset
from .evaluate import evaluate_utterances
from .model import SpectreNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id: int) -> None:
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


@torch.no_grad()
def eval_crops(model, dl, device, amp) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss(reduction="sum")
    for wav, y in dl:
        wav, y = wav.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(wav)
        loss_sum += crit(logits.float(), y).item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return loss_sum / total, correct / total


def load_cfg(a: argparse.Namespace) -> dict:
    cfg = yaml.safe_load(a.config.read_text())
    if a.manifest:
        cfg["data"]["manifest"] = a.manifest
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
    if a.backbone:
        cfg["model"]["backbone"] = a.backbone
    if a.batch_size:
        cfg["train"]["batch_size"] = a.batch_size
    if a.num_workers is not None:
        cfg["train"]["num_workers"] = a.num_workers
    if a.no_pretrained:
        cfg["model"]["pretrained"] = False
    cfg["train"]["max_steps"] = a.max_steps
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    p.add_argument("--resume", type=Path, help="path to runs/<run>/last.pt to continue training")
    p.add_argument("--manifest", type=str, help="override data.manifest")
    p.add_argument("--epochs", type=int, help="override train.epochs")
    p.add_argument("--backbone", type=str, help="override model.backbone")
    p.add_argument("--batch-size", type=int, help="override train.batch_size")
    p.add_argument("--num-workers", type=int, help="override train.num_workers")
    p.add_argument("--max-steps", type=int, default=None, help="stop each epoch early (smoke tests)")
    p.add_argument("--no-pretrained", action="store_true")
    a = p.parse_args()

    ckpt = None
    if a.resume:
        ckpt = torch.load(a.resume, map_location="cpu", weights_only=False)
        cfg, run_dir = ckpt["cfg"], a.resume.parent
        if a.num_workers is not None:  # safe to change on resume; anything affecting the schedule is not
            cfg["train"]["num_workers"] = a.num_workers
        cfg["model"]["pretrained"] = False  # weights come from the checkpoint
    else:
        cfg = load_cfg(a)

    tc = cfg["train"]
    seed_everything(tc["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(tc.get("amp", True)) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    df = pd.read_csv(cfg["data"]["manifest"], dtype={"speaker": str, "chapter": str})
    label_map = ckpt["label_map"] if ckpt else {s: i for i, s in enumerate(sorted(df["speaker"].unique()))}
    splits = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    print(f"Device: {device} · speakers: {len(label_map)} · "
          + " · ".join(f"{k}: {len(v)}" for k, v in splits.items()))

    nw = tc["num_workers"]
    loader_kw = dict(num_workers=nw, pin_memory=device.type == "cuda", worker_init_fn=worker_init,
                     persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None)
    train_dl = DataLoader(SpeakerDataset(splits["train"], label_map, cfg, "train"),
                          batch_size=tc["batch_size"], shuffle=True, drop_last=True, **loader_kw)
    val_dl = DataLoader(SpeakerDataset(splits["val"], label_map, cfg, "val"),
                        batch_size=tc["batch_size"], shuffle=False, **loader_kw)

    model = SpectreNet(cfg, len(label_map)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    max_steps = tc.get("max_steps")
    steps_per_epoch = min(len(train_dl), max_steps or len(train_dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], pct_start=0.1,
                                                total_steps=tc["epochs"] * steps_per_epoch)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    if ckpt:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        set_rng_state(ckpt["rng"])
        start_epoch, history, best_acc = ckpt["epoch"] + 1, ckpt["history"], ckpt["best_acc"]
        print(f"↻ Resumed {run_dir} from epoch {ckpt['epoch']} (best val acc {best_acc:.3f})")
    else:
        run_dir = Path("runs") / f"{cfg['run_name']}_{time.strftime('%Y%m%d-%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        start_epoch, history, best_acc = 1, [], -1.0

    try:
        for epoch in range(start_epoch, tc["epochs"] + 1):
            model.train()
            t0, run_loss, run_correct, seen = time.time(), 0.0, 0, 0
            bar = tqdm(train_dl, total=steps_per_epoch, desc=f"epoch {epoch}/{tc['epochs']}", leave=False)
            for step, (wav, y) in enumerate(bar):
                if step >= steps_per_epoch:
                    break
                wav, y = wav.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    logits = model(wav)
                    loss = crit(logits.float(), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
                run_loss += loss.item() * y.numel()
                run_correct += (logits.argmax(1) == y).sum().item()
                seen += y.numel()
                bar.set_postfix(loss=f"{run_loss / seen:.3f}", acc=f"{run_correct / seen:.3f}")

            val_loss, val_acc = eval_crops(model, val_dl, device, amp)
            rec = {"epoch": epoch, "train_loss": run_loss / seen, "train_acc": run_correct / seen,
                   "val_loss": val_loss, "val_acc": val_acc, "secs": round(time.time() - t0, 1)}
            history.append(rec)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            flag = ""
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({"model": model.state_dict(), "cfg": cfg, "label_map": label_map, "epoch": epoch},
                           run_dir / "best.pt")
                flag = " ★"
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(), "rng": rng_state(), "epoch": epoch,
                        "history": history, "best_acc": best_acc, "cfg": cfg, "label_map": label_map},
                       run_dir / "last.pt")
            print(f"epoch {epoch:3d} · train loss {rec['train_loss']:.3f} acc {rec['train_acc']:.3f} · "
                  f"val loss {val_loss:.3f} acc {val_acc:.3f} · {rec['secs']}s{flag}")
    except KeyboardInterrupt:
        done = history[-1]["epoch"] if history else 0
        print(f"\n⏸ Interrupted. Last completed epoch: {done}. Continue with:\n"
              f"   python -m spectre.train --resume {run_dir / 'last.pt'}")
        return

    # final test on full utterances with the best checkpoint (new recording sessions)
    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test = evaluate_utterances(model, splits["test"], label_map, cfg, device)
    test["best_val_acc"], test["best_epoch"] = best_acc, best["epoch"]
    (run_dir / "results.json").write_text(json.dumps(test, indent=2))
    print(f"\nTEST (full utterances, unseen sessions): top-1 {test['top1']:.3f} · top-5 {test['top5']:.3f}")
    print(f"Artefacts in {run_dir}")


if __name__ == "__main__":
    main()
