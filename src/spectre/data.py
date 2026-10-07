"""Manifest building, session-aware splitting and the waveform dataset.

Split policy (avoids the classic leakage of putting clips from the same
recording in both train and test, which makes the model learn the
microphone/room instead of the voice):

  * speaker with >= 3 chapters -> one chapter for test, one for val, rest for train
  * speaker with 2 chapters    -> one chapter for test; val = 10% of utterances of the other
  * speaker with 1 chapter     -> utterance-level 70/15/15 split (flagged as same_session)

In LibriSpeech a "chapter" is a separate recording session, so the test
numbers reflect generalisation to a new session of a known speaker.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# --------------------------------------------------------------------------- manifest
def scan_librispeech(subset_dir: Path) -> pd.DataFrame:
    rows = []
    for path in tqdm(sorted(subset_dir.glob("*/*/*.flac")), desc="Scanning"):
        info = sf.info(path)
        rows.append(
            {
                "path": str(path),
                "speaker": path.parts[-3],
                "chapter": path.parts[-2],
                "duration": info.frames / info.samplerate,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No .flac files under {subset_dir}")
    return pd.DataFrame(rows)


def assign_splits(df: pd.DataFrame, seed: int = 42, val_frac: float = 0.10) -> pd.DataFrame:
    rng = random.Random(seed)
    df = df.copy()
    df["split"] = "train"
    df["same_session_split"] = False

    for spk, g in df.groupby("speaker"):
        chapters = sorted(g["chapter"].unique())
        rng.shuffle(chapters)
        if len(chapters) >= 3:
            df.loc[g.index[g["chapter"] == chapters[0]], "split"] = "test"
            df.loc[g.index[g["chapter"] == chapters[1]], "split"] = "val"
        elif len(chapters) == 2:
            df.loc[g.index[g["chapter"] == chapters[0]], "split"] = "test"
            rest = list(g.index[g["chapter"] == chapters[1]])
            rng.shuffle(rest)
            df.loc[rest[: max(1, int(len(rest) * val_frac))], "split"] = "val"
        else:
            idx = list(g.index)
            rng.shuffle(idx)
            n_test = max(1, int(len(idx) * 0.15))
            n_val = max(1, int(len(idx) * 0.15))
            df.loc[idx[:n_test], "split"] = "test"
            df.loc[idx[n_test : n_test + n_val], "split"] = "val"
            df.loc[g.index, "same_session_split"] = True
    return df


def build_manifest(subset_dirs: Path | list[Path], out_csv: Path, seed: int = 42) -> pd.DataFrame:
    """Scan one or more LibriSpeech subsets (their speakers are disjoint) into one split manifest."""
    dirs = [subset_dirs] if isinstance(subset_dirs, Path) else list(subset_dirs)
    df = assign_splits(pd.concat([scan_librispeech(d) for d in dirs], ignore_index=True), seed=seed)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    hours = df.groupby("split")["duration"].sum() / 3600
    print(f"\n✓ Manifest written to {out_csv}")
    print(f"  speakers: {df['speaker'].nunique()} · utterances: {len(df)}")
    for split in ("train", "val", "test"):
        print(f"  {split:5s}: {(df['split'] == split).sum():6d} utts · {hours.get(split, 0):6.2f} h")
    n_same = df.loc[df["same_session_split"], "speaker"].nunique()
    if n_same:
        print(f"  ⚠ {n_same} speaker(s) have a single chapter → same-session split for them")
    return df


# --------------------------------------------------------------------------- dataset
def _to_mono_resampled(wav: np.ndarray, sr: int, sample_rate: int) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != sample_rate:
        import torchaudio.functional as AF

        wav = AF.resample(torch.from_numpy(np.ascontiguousarray(wav)), sr, sample_rate).numpy()
    return wav.astype(np.float32, copy=False)


def load_audio(path: str, sample_rate: int) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    return _to_mono_resampled(wav, sr, sample_rate)


def load_segment(path: str, sample_rate: int, n: int, where: str = "random") -> np.ndarray:
    """Read only an n-sample segment from disk (random or centred) instead of the whole file.
    Falls back to a full read when the file needs resampling or is shorter than n."""
    with sf.SoundFile(path) as f:
        total, sr = f.frames, f.samplerate
        if sr != sample_rate or total <= n:
            wav = f.read(dtype="float32", always_2d=False)
            return fix_length(_to_mono_resampled(wav, sr, sample_rate), n,
                              start=None if where == "random" else max(0, (len(wav) - n) // 2))
        start = np.random.randint(0, total - n + 1) if where == "random" else (total - n) // 2
        f.seek(start)
        wav = f.read(n, dtype="float32", always_2d=False)
    return _to_mono_resampled(wav, sr, sample_rate)


def fix_length(wav: np.ndarray, n: int, start: int | None = None) -> np.ndarray:
    """Crop to n samples (random start if start is None) or repeat-pad if shorter."""
    if len(wav) < n:
        reps = int(np.ceil(n / max(len(wav), 1)))
        return np.tile(wav, reps)[:n]
    if start is None:
        start = np.random.randint(0, len(wav) - n + 1)
    return wav[start : start + n]


class SpeakerDataset(Dataset):
    """Returns (waveform[T], label). Train: random crop + waveform augmentation.
    Val: deterministic centre crop. Test: full utterance (see evaluate.py)."""

    def __init__(self, df: pd.DataFrame, label_map: dict[str, int], cfg: dict, mode: str):
        self.df = df.reset_index(drop=True)
        self.label_map = label_map
        self.mode = mode
        self.sr = cfg["data"]["sample_rate"]
        self.n = int(cfg["data"]["crop_seconds"] * self.sr)
        self.aug = cfg.get("augment", {})

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, wav: np.ndarray) -> np.ndarray:
        lo, hi = self.aug.get("gain_db", [0, 0])
        wav = wav * 10 ** (np.random.uniform(lo, hi) / 20)
        if np.random.rand() < self.aug.get("noise_prob", 0.0):
            snr = np.random.uniform(*self.aug.get("snr_db", [10, 30]))
            p_sig = np.mean(wav**2) + 1e-10
            wav = wav + np.random.randn(len(wav)).astype(np.float32) * np.sqrt(p_sig / 10 ** (snr / 10))
        return wav.astype(np.float32)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        label = self.label_map[row["speaker"]]
        if self.mode == "train":
            wav = self._augment(load_segment(row["path"], self.sr, self.n, "random"))
        elif self.mode == "val":
            wav = load_segment(row["path"], self.sr, self.n, "centre")
        else:  # test: full utterance, windowed in evaluate.py
            wav = load_audio(row["path"], self.sr)
        return torch.from_numpy(np.ascontiguousarray(wav)), label


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build a split manifest for one or more LibriSpeech subsets")
    p.add_argument("--subset", nargs="+", default=["dev-clean"],
                   help="e.g. --subset train-clean-100 train-clean-360 (merged into one manifest)")
    p.add_argument("--root", type=Path, default=Path("data/raw/LibriSpeech"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    name = "+".join(a.subset)
    build_manifest([a.root / s for s in a.subset], a.out or Path(f"data/manifests/{name}.csv"), a.seed)
