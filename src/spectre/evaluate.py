"""Full-utterance evaluation: slide a crop-length window over each utterance,
average the softmax probabilities and take the arg-max speaker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SpeakerDataset
from .model import SpectreNet


def sliding_windows(wav: torch.Tensor, win: int, hop: int) -> torch.Tensor:
    if wav.numel() <= win:
        reps = -(-win // max(wav.numel(), 1))
        return wav.repeat(reps)[:win].unsqueeze(0)
    return wav.unfold(0, win, hop)  # [n_windows, win]


@torch.no_grad()
def evaluate_utterances(model, df, label_map, cfg, device, desc="test") -> dict:
    model.eval()
    ds = SpeakerDataset(df, label_map, cfg, mode="test")
    dl = DataLoader(ds, batch_size=1, num_workers=cfg["train"].get("num_workers", 0))
    sr = cfg["data"]["sample_rate"]
    win = int(cfg["data"]["crop_seconds"] * sr)
    hop = int(cfg["data"]["eval_hop_seconds"] * sr)

    top1 = top5 = 0
    for wav, y in tqdm(dl, desc=desc, leave=False):
        windows = sliding_windows(wav[0], win, hop).to(device)
        probs = F.softmax(model(windows).float(), dim=-1).mean(0)
        top = probs.topk(min(5, probs.numel())).indices.cpu()
        top1 += int(top[0] == y[0])
        top5 += int((top == y[0]).any())
    n = len(ds)
    return {"top1": top1 / n, "top5": top5 / n, "n_utterances": n}


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a trained SPECTRE checkpoint on the test split")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--split", default="test")
    a = p.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg, label_map = ck["cfg"], ck["label_map"]
    cfg["model"]["pretrained"] = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectreNet(cfg, len(label_map)).to(device)
    model.load_state_dict(ck["model"])

    df = pd.read_csv(cfg["data"]["manifest"], dtype={"speaker": str, "chapter": str})
    res = evaluate_utterances(model, df[df["split"] == a.split], label_map, cfg, device, a.split)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
