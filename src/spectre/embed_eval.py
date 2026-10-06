"""Evaluate a speaker-embedding checkpoint the way speaker-recognition papers do.

    python -m spectre.embed_eval --ckpt runs/<run>/best.pt

Three numbers come out:

1. **Verification EER / minDCF on unseen speakers** — every pair of utterances from
   speakers never seen in training (LibriSpeech dev-clean + test-clean) is scored by cosine
   similarity; same-speaker pairs are only taken across *different* chapters (sessions).
2. **Enroll-and-identify on unseen speakers** — each new speaker is enrolled with ~10 s
   of speech from one chapter, then their utterances from other chapters are identified
   among all enrolled speakers. This is the "add a new person without retraining" scenario.
3. **Closed-set test on known speakers** — the 251 training speakers are enrolled from their
   training utterances and the test split is identified by nearest embedding, which is
   directly comparable with the phase-1 classifier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import load_audio
from .model import build_model


# --------------------------------------------------------------------------- embeddings
@torch.no_grad()
def extract(model, paths: list[str], cfg: dict, device, desc: str = "embedding") -> torch.Tensor:
    model.eval()
    sr = cfg["data"]["sample_rate"]
    max_len = int(cfg.get("eval", {}).get("max_utt_seconds", 20) * sr)
    out = []
    for p in tqdm(paths, desc=desc, leave=False):
        wav = torch.from_numpy(load_audio(p, sr))[:max_len].to(device)
        out.append(F.normalize(model.embed(wav.unsqueeze(0)).float(), dim=-1).cpu())
    return torch.cat(out)


# --------------------------------------------------------------------------- metrics
def eer_and_mindcf(target: np.ndarray, nontarget: np.ndarray, p_target: float = 0.01) -> dict:
    scores = np.concatenate([target, nontarget])
    labels = np.concatenate([np.ones_like(target), np.zeros_like(nontarget)])
    order = np.argsort(scores)[::-1]
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    frr = 1 - tp / tp[-1]          # miss rate when accepting the top-k scores
    far = fp / fp[-1]              # false-alarm rate
    i = np.nanargmin(np.abs(frr - far))
    dcf = p_target * frr + (1 - p_target) * far
    min_dcf = dcf.min() / min(p_target, 1 - p_target)
    return {"eer": float((frr[i] + far[i]) / 2), "min_dcf": float(min_dcf),
            "eer_threshold": float(scores[order][i]),
            "n_target": int(len(target)), "n_nontarget": int(len(nontarget))}


def verification(emb: torch.Tensor, df: pd.DataFrame) -> dict:
    sim = (emb @ emb.T).numpy()
    spk = df["speaker"].to_numpy()
    chap = df["chapter"].to_numpy()
    iu = np.triu_indices(len(df), k=1)
    same_spk = spk[iu[0]] == spk[iu[1]]
    same_chap = chap[iu[0]] == chap[iu[1]]
    s = sim[iu]
    return eer_and_mindcf(s[same_spk & ~same_chap], s[~same_spk])


def enroll_and_identify(emb: torch.Tensor, df: pd.DataFrame, enroll_seconds: float, seed: int = 0) -> dict:
    """Enroll each speaker from one chapter (~enroll_seconds of speech); test on their other chapters."""
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    centroids, names, test_idx = [], [], []
    for spk, g in df.groupby("speaker"):
        chapters = sorted(g["chapter"].unique())
        if len(chapters) < 2:
            continue
        enroll_chap = chapters[rng.integers(len(chapters))]
        e = g[g["chapter"] == enroll_chap].sample(frac=1, random_state=seed)
        take = e.index[np.cumsum(e["duration"].to_numpy()) - e["duration"].to_numpy() < enroll_seconds]
        centroids.append(F.normalize(emb[take].mean(0), dim=-1))
        names.append(spk)
        test_idx.extend(g.index[g["chapter"] != enroll_chap])
    C = torch.stack(centroids)
    scores = emb[test_idx] @ C.T
    pred = scores.argmax(1).numpy()
    truth = np.array([names.index(s) for s in df.loc[test_idx, "speaker"]])
    top5 = (scores.topk(min(5, len(names)), dim=1).indices.numpy() == truth[:, None]).any(1)
    return {"top1": float((pred == truth).mean()), "top5": float(top5.mean()),
            "n_speakers": len(names), "n_test_utterances": len(test_idx)}


def closed_set(model, df: pd.DataFrame, cfg: dict, device, per_speaker: int, seed: int = 0) -> dict:
    tr = df[df["split"] == "train"].sample(frac=1, random_state=seed).groupby("speaker").head(per_speaker)
    te = df[df["split"] == "test"]
    e_tr = extract(model, tr["path"].tolist(), cfg, device, "enroll known")
    e_te = extract(model, te["path"].tolist(), cfg, device, "test known")
    names = sorted(tr["speaker"].unique())
    C = F.normalize(torch.stack([e_tr[(tr["speaker"] == s).to_numpy()].mean(0) for s in names]), dim=-1)
    scores = e_te @ C.T
    truth = torch.tensor([names.index(s) for s in te["speaker"]])
    return {"top1": float((scores.argmax(1) == truth).float().mean()),
            "top5": float((scores.topk(5, dim=1).indices == truth[:, None]).any(1).float().mean()),
            "n_speakers": len(names), "n_test_utterances": len(te)}


# --------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description="Speaker-embedding evaluation (EER, enroll-and-identify)")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--skip-closed-set", action="store_true", help="skip the (slower) known-speaker test")
    a = p.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg, label_map = ck["cfg"], ck["label_map"]
    ev = cfg.get("eval", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, len(label_map)).to(device)
    model.load_state_dict(ck["model"])

    results: dict = {"checkpoint": str(a.ckpt), "epoch": ck.get("epoch")}

    unseen = pd.concat([pd.read_csv(m, dtype={"speaker": str, "chapter": str})
                        for m in ev.get("unseen_manifests", []) if Path(m).exists()], ignore_index=True)
    overlap = set(unseen["speaker"]) & set(label_map) if len(unseen) else set()
    if overlap:
        raise ValueError(f"{len(overlap)} 'unseen' speakers were in training — check eval.unseen_manifests")
    if len(unseen):
        emb = extract(model, unseen["path"].tolist(), cfg, device, "unseen speakers")
        results["unseen_verification"] = verification(emb, unseen)
        results["unseen_enroll_identify"] = enroll_and_identify(emb, unseen, ev.get("enroll_seconds", 10))
    else:
        print("⚠ No unseen-speaker manifests found — run: python -m spectre.data --subset dev-clean (and test-clean)")

    if not a.skip_closed_set:
        known = pd.read_csv(cfg["data"]["manifest"], dtype={"speaker": str, "chapter": str})
        results["known_closed_set"] = closed_set(model, known, cfg, device, ev.get("enroll_per_speaker", 20))

    out = a.ckpt.parent / "embed_results.json"
    out.write_text(json.dumps(results, indent=2))

    print()
    if "unseen_verification" in results:
        v, e = results["unseen_verification"], results["unseen_enroll_identify"]
        print(f"UNSEEN speakers ({e['n_speakers']}) · verification EER {100 * v['eer']:.2f} % · "
              f"minDCF(0.01) {v['min_dcf']:.3f} · threshold {v['eer_threshold']:.3f}")
        print(f"UNSEEN enroll {ev.get('enroll_seconds', 10)} s → identify · top-1 {e['top1']:.3f} · top-5 {e['top5']:.3f}")
    if "known_closed_set" in results:
        k = results["known_closed_set"]
        print(f"KNOWN speakers ({k['n_speakers']}) by embedding · top-1 {k['top1']:.3f} · top-5 {k['top5']:.3f}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
