"""Calibrate and measure how well the conversation captions tell voices apart, for one SPECTRE model.

    python -m spectre.conv_eval                         # the latest phase-2 model (a few minutes on a GPU)
    python -m spectre.conv_eval --ckpt runs/<run>/best.pt

LibriSpeech speakers each read with their own microphone, so a model trained on them partly learns the channel
as if it were the voice. Around one microphone everyone shares the channel and different people look alike.
This command measures that, and fixes it as far as it can without retraining:

1. Room directions — dev-clean clips are played in simulated rooms; the directions in which a room moves the
   model's embeddings are learned (nuisance attribute projection) and removed from every embedding afterwards.
2. Conversations — one-microphone conversations between 2-4 test-clean speakers (never seen in training, and
   other people than in step 1) go through exactly the live pipeline: turns, voice-change cuts, voice rules.
3. The settings that give the most speech to the right person, without inventing extra speakers, are saved to
   runs/<run>/conversation.json, which `python -m spectre.conversation` loads automatically.

Needs the dev-clean and test-clean manifests (the phase-2 evaluation data).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from . import live, room
from .conversation import Line, TurnSegmenter, VoiceRules, Voices, relabel, split_by_voice, CALIBRATION

MANIFESTS = {"dev": Path("data/manifests/dev-clean.csv"), "test": Path("data/manifests/test-clean.csv")}


class CachedEmbedder:
    """Raw embeddings of audio slices, computed once (slices of the same turn are recognised by their memory)."""

    def __init__(self, embedder):
        self.embedder, self.cache = embedder, {}

    @staticmethod
    def _key(w: np.ndarray) -> tuple:
        return w.__array_interface__["data"][0], len(w)

    def __call__(self, wavs) -> np.ndarray:
        wavs = [wavs] if isinstance(wavs, np.ndarray) else list(wavs)
        todo = [w for w in wavs if self._key(w) not in self.cache]
        for i in range(0, len(todo), 64):
            for w, e in zip(todo[i: i + 64], self.embedder(todo[i: i + 64])):
                self.cache[self._key(w)] = e
        return np.stack([self.cache[self._key(w)] for w in wavs])


def room_directions(embed, rng: np.random.Generator, per_speaker: int = 15, n_rooms: int = 40, passes: int = 3,
                    keep: int = 10) -> np.ndarray:
    """The directions in which simulated rooms move the embeddings of dev-clean clips (strongest first)."""
    df = pd.read_csv(MANIFESTS["dev"])
    df = df[df.duration >= 3].groupby("speaker").head(per_speaker)
    clips = []
    for path in df.path:
        w, _ = sf.read(path, dtype="float32")
        n = min(len(w), int(np.exp(rng.uniform(np.log(1.0), np.log(5.0))) * room.SR))
        s0 = rng.integers(0, len(w) - n + 1)
        clips.append(w[s0: s0 + n])
    rooms = [room.make_room(rng) for _ in range(n_rooms)]
    clean = embed(clips)
    shifts = [embed([room.apply_room(c, rooms[rng.integers(n_rooms)], rng) for c in clips]) - clean
              for _ in range(passes)]
    return np.linalg.svd(np.concatenate(shifts), full_matrices=False)[2][:keep]


def conversations(n: int, rng: np.random.Generator) -> list[dict]:
    """Simulated one-microphone conversations between test-clean speakers, already cut into turns."""
    pool = room.sessions(str(MANIFESTS["test"]))
    out = []
    for i in range(n):
        audio, truth = room.conversation(rng, pool, 2 + i % 3, int(rng.integers(10, 19)))
        seg = TurnSegmenter(room.SR)
        turns = [t for s in range(0, len(audio), 1600) for t in seg.push(audio[s: s + 1600])] + seg.flush()
        out.append({"truth": truth, "turns": turns})
    return out


def run_rules(conv: dict, embed: CachedEmbedder, rules: VoiceRules) -> list[tuple[float, float, int]]:
    """Exactly the live pipeline (without words): voice-change cuts, then voices turn by turn."""
    project = lambda w: rules.project(embed(w))
    parts = [p for t in conv["turns"] for p in split_by_voice(t, project, rules.cut, rules.min_cut_side_s)]
    if not parts:
        return []
    E = project([p.audio for p in parts])
    voices, lines = Voices(rules), []
    for p, e in zip(parts, E):
        spk, follows = voices.assign(e, p.speech_s)
        lines.append(Line(p.start, p.end, spk, p.speech_s, emb=e, follows=follows))
        relabel(lines, voices)
    return [(ln.start, ln.end, ln.speaker) for ln in lines]


def evaluate(convs: list[dict], embed: CachedEmbedder, rules: VoiceRules) -> dict:
    scores = [room.score(run_rules(c, embed, rules), c["truth"]) for c in convs]
    return {"accuracy": float(np.mean([s["accuracy"] for s in scores])),
            "merged": float(np.mean([s["merged"] > 0 for s in scores])),
            "extra": float(np.mean([s["labels"] - s["speakers"] for s in scores]))}


def calibrate(convs: list[dict], embed: CachedEmbedder, directions: np.ndarray, max_extra: float = 0.5,
              log=print) -> tuple[VoiceRules, dict, dict]:
    """Try the candidate settings; the best is the one giving the most speech to the right person among those that
    add at most `max_extra` spurious speakers per conversation. Returns (rules, its result, best without rooms)."""
    grids = {0: dict(T=(0.55, 0.6, 0.65, 0.7), S=(0.45, 0.5, 0.55)),
             3: dict(T=(0.3, 0.35, 0.4, 0.45, 0.5), S=(0.2, 0.25, 0.3, 0.35)),
             5: dict(T=(0.3, 0.35, 0.4, 0.45, 0.5), S=(0.2, 0.25, 0.3, 0.35)),
             8: dict(T=(0.3, 0.35, 0.4, 0.45, 0.5), S=(0.2, 0.25, 0.3, 0.35))}
    results = []
    for k, g in grids.items():
        for T, S, nm in itertools.product(g["T"], g["S"], (0.3, 0.45)):
            rules = VoiceRules(threshold=T, cut=S, new_margin=nm, room=directions[:k] if k else None)
            r = evaluate(convs, embed, rules)
            results.append((r, k, rules))
        best_k = max((x for x in results if x[1] == k), key=lambda x: x[0]["accuracy"])
        log(f"  {'no room compensation' if k == 0 else f'{k} room directions removed':<28} best: "
            f"{100 * best_k[0]['accuracy']:.1f} % of the speech to the right person "
            f"(same voice ≥ {best_k[2].threshold:.2f}, cut < {best_k[2].cut:.2f})")
    ok = [x for x in results if x[0]["extra"] <= max_extra] or results
    best = max(ok, key=lambda x: x[0]["accuracy"])
    plain = [x for x in results if x[1] == 0]
    base = max([x for x in plain if x[0]["extra"] <= max_extra] or plain, key=lambda x: x[0]["accuracy"])
    return best[2], best[0], {**base[0], "threshold": base[2].threshold, "cut": base[2].cut}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m spectre.conv_eval", description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", type=Path, help="SPECTRE ECAPA checkpoint (default: the latest phase-2 run)")
    p.add_argument("--conversations", type=int, default=150, help="simulated conversations (default: %(default)s)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--cpu", action="store_true", help="run the model on the CPU")
    a = p.parse_args(argv)
    missing = [str(m) for m in MANIFESTS.values() if not m.exists()]
    if missing:
        sys.exit(f"Missing {', '.join(missing)}. Download and index them first:\n"
                 "  python scripts/download_librispeech.py --subset dev-clean\n"
                 "  python scripts/download_librispeech.py --subset test-clean\n"
                 "  python -m spectre.data --subset dev-clean\n  python -m spectre.data --subset test-clean")
    ckpt = a.ckpt or live.find_checkpoint()
    model = live.Embedder(ckpt, "cpu" if a.cpu else None)
    embed = CachedEmbedder(model)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    print(f"Model {model.model_id} on {model.device.type}")
    print("1/3  Learning the room directions (dev-clean speakers in simulated rooms)…", flush=True)
    directions = room_directions(model, rng)
    print(f"2/3  Simulating {a.conversations} one-microphone conversations (test-clean speakers)…", flush=True)
    convs = conversations(a.conversations, rng)
    print("3/3  Trying the voice settings…", flush=True)
    rules, result, base = calibrate(convs, embed, directions)
    k = 0 if rules.room is None else len(rules.room)
    out = {"model": model.model_id, "threshold": rules.threshold, "cut": rules.cut,
           "merge_margin": rules.merge_margin, "follow_below_s": rules.follow_below_s, "new_margin": rules.new_margin,
           "min_cut_side_s": rules.min_cut_side_s, "room_directions": [] if k == 0 else rules.room.tolist(),
           **result, "without_room_compensation": base, "conversations": a.conversations, "seed": a.seed,
           "created": datetime.now().isoformat(timespec="seconds")}
    path = Path(ckpt).parent / CALIBRATION
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nOne-microphone conversations, 2-4 people never heard in training ({a.conversations}, "
          f"{time.time() - t0:.0f} s):")
    print(f"  without room compensation : {100 * base['accuracy']:.1f} % of the speech to the right person · "
          f"two people merged in {100 * base['merged']:.0f} % of conversations")
    print(f"  with room compensation    : {100 * result['accuracy']:.1f} % · merged in {100 * result['merged']:.0f} % · "
          f"{result['extra']:+.2f} speakers per conversation ({k} room directions removed, same voice ≥ "
          f"{rules.threshold:.2f}, cut < {rules.cut:.2f})")
    print(f"✓ Saved to {path}; spectre.conversation uses it from now on.")


if __name__ == "__main__":
    main()
