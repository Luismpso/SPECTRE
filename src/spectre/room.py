"""Simulated rooms and one-microphone conversations, to measure and calibrate how well voices are told apart.

Every LibriSpeech speaker reads in their own room with their own microphone, so a model trained on it learns
that the channel is part of the voice. In a real conversation everyone shares one room and one microphone, and
different people suddenly look alike. These simulations reproduce that: real voices the model never heard,
passed together through one synthetic room (reverberation, microphone colour, background noise).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, fftconvolve, lfilter

SR = 16000


def make_room(rng: np.random.Generator) -> dict:
    """A random room and microphone: reverberation time 0.25-0.7 s, direct-to-reverberant ratio 0-10 dB,
    a band-pass microphone and pink-ish background noise 12-25 dB below the speech."""
    rt60 = rng.uniform(0.25, 0.7)
    t = np.arange(int(rt60 * SR)) / SR
    tail = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)        # -60 dB after rt60
    tail[: int(0.003 * SR)] = 0
    tail *= 10 ** (-rng.uniform(0, 10) / 20) / np.sqrt(np.sum(tail ** 2))
    tail[0] = 1.0                                                          # the direct sound
    b, a = butter(2, [rng.uniform(80, 200), rng.uniform(5000, 7500)], btype="band", fs=SR)
    return {"rir": tail, "ba": (b, a), "snr": rng.uniform(12, 25), "pink": rng.uniform(0.0, 0.98)}


def apply_room(x: np.ndarray, room: dict, rng: np.random.Generator, speech: np.ndarray | None = None) -> np.ndarray:
    """Play a signal in the room. `speech` marks the samples used to set the noise level (default: all)."""
    y = lfilter(*room["ba"], fftconvolve(x, room["rir"])[: len(x)])
    noise = lfilter([1.0], [1.0, -room["pink"]], rng.standard_normal(len(y)))
    level = np.sqrt(np.mean(y[speech] ** 2)) if speech is not None and speech.any() else np.sqrt(np.mean(y ** 2))
    noise *= level / (np.sqrt(np.mean(noise ** 2)) + 1e-12) * 10 ** (-room["snr"] / 20)
    y = (y + noise).astype(np.float32)
    return y / max(1.0, float(np.abs(y).max()) / 0.95)


def sessions(manifest: str, min_utts: int = 12) -> dict:
    """speaker → list of recording sessions (chapters), each a list of utterance paths of at least 2 s."""
    df = pd.read_csv(manifest)
    df = df[df.duration >= 2.0]
    out: dict = {}
    for (spk, _), g in df.groupby(["speaker", "chapter"]):
        if len(g) >= min_utts:
            out.setdefault(spk, []).append(list(g.path))
    return out


def conversation(rng: np.random.Generator, pool: dict, n_speakers: int, n_turns: int, room: bool = True):
    """A conversation between `n_speakers` people (one session each, at different distances from the microphone):
    turns of 0.8-6 s, the same person goes on after a pause 20 % of the time, answers come 0.15-1 s after the
    other person stops. Returns (16 kHz audio, [(start, end, speaker)])."""
    spks = rng.choice(sorted(pool), n_speakers, replace=False)
    utts = {s: list(rng.permutation(pool[s][rng.integers(len(pool[s]))])) for s in spks}
    gain = {s: 10 ** (rng.uniform(-6, 3) / 20) for s in spks}
    parts, truth, t = [np.zeros(int(0.8 * SR), np.float32)], [], 0.8
    cur = rng.choice(spks)
    for k in range(n_turns):
        if k:
            nxt = cur if rng.random() < 0.2 else rng.choice([s for s in spks if s != cur])
            pause = rng.uniform(0.6, 1.5) if nxt == cur else rng.uniform(0.15, 1.0)
            parts.append(np.zeros(int(pause * SR), np.float32))
            t, cur = t + pause, nxt
        wav, _ = sf.read(utts[cur].pop(), dtype="float32")
        n = min(len(wav), int(np.exp(rng.uniform(np.log(0.8), np.log(6.0))) * SR))
        s0 = rng.integers(0, len(wav) - n + 1)
        parts.append(wav[s0: s0 + n] * gain[cur])
        truth.append((t, t + n / SR, cur))
        t += n / SR
    parts.append(np.zeros(SR, np.float32))
    clean = np.concatenate(parts)
    the_room = make_room(rng)                       # drawn even without the room, so both versions match
    if not room:
        return clean / max(1.0, float(np.abs(clean).max()) / 0.95), truth
    return apply_room(clean, the_room, rng, speech=np.abs(clean) > 1e-3), truth


def overlap_by_speaker(truth, a: float, b: float) -> dict:
    out: dict = {}
    for s0, s1, spk in truth:
        o = min(b, s1) - max(a, s0)
        if o > 0:
            out[spk] = out.get(spk, 0.0) + o
    return out


def score(lines, truth) -> dict:
    """lines: [(start, end, label)]. The share of speech time given to the right person (after the best one-to-one
    match of labels to people), and whether any label holds ≥ 20 % of the speech of two people (a merge)."""
    spks = sorted({s for *_, s in truth}, key=str)
    labels = sorted({c for *_, c in lines}, key=str)
    M = np.zeros((len(labels), len(spks)))
    for a, b, c in lines:
        for spk, o in overlap_by_speaker(truth, a, b).items():
            M[labels.index(c), spks.index(spk)] += o
    r, k = linear_sum_assignment(-M)
    merged = sum(1 for row in M if row.sum() > 0 and (row >= 0.2 * row.sum()).sum() >= 2)
    return {"accuracy": M[r, k].sum() / max(M.sum(), 1e-9), "speakers": len(spks), "labels": len(labels),
            "merged": merged}
