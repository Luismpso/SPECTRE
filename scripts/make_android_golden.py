"""Golden fixtures for the Kotlin port (android/core): the same inputs through the Python implementation.

    python scripts/make_android_golden.py      # → android/core/src/test/resources/golden.json

Signals and vectors come from a small linear congruential generator that the Kotlin tests implement too, so the
fixtures stay text-only. Regenerate them whenever the Python conversation logic changes.
"""
from __future__ import annotations

import json
import math
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from spectre import conversation as cv

OUT = Path("android/core/src/test/resources/golden.json")
SR = 16000


class Lcg:
    """x ← 1664525·x + 1013904223 (mod 2³²); next() is uniform in [0, 1)."""

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    def next(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state / 2 ** 32


def synth(spec: list[dict], seed: int = 1) -> np.ndarray:
    """tone: amp·sin(2π·f·n/sr) (n restarts per segment); quiet: amp·(2u − 1); talk: a tone with 60 ms gaps."""
    rng, parts = Lcg(seed), []
    for seg in spec:
        n = int(seg["s"] * SR)
        if seg["kind"] == "quiet":
            parts.append(np.array([seg["amp"] * (2 * rng.next() - 1) for _ in range(n)], np.float32))
        else:
            t = np.arange(n) / SR
            x = seg["amp"] * np.sin(2 * math.pi * seg["f"] * t)
            if seg["kind"] == "talk":
                x = x * (t % 0.26 < 0.2)
            parts.append(x.astype(np.float32))
    return np.concatenate(parts)


def unit_vectors(seed: int, n: int, dim: int = 192) -> np.ndarray:
    rng = Lcg(seed)
    v = np.array([[2 * rng.next() - 1 for _ in range(dim)] for _ in range(n)])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def fixtures() -> dict:
    g: dict = {}
    words = ["João", "Joana", "Ana-Rita", "Inês", "çãÉ…ß", "  Olá!  ", "Ó João", "D'Artagnan", "Zé"]
    g["fold"] = [[w, cv.fold(w)] for w in words]
    pairs = [("joao", "joana"), ("tedo", "pedro"), ("katarina", "catarina"), ("pedor", "pedro"), ("mariana", "marianna"),
             ("abcde", "edcba"), ("rafael", "raphael"), ("juana", "joana"), ("a", "b"), ("aaaa", "aaab"), ("", "x")]
    g["ratio"] = [[a, b, SequenceMatcher(None, a, b).ratio()] for a, b in pairs]

    lines = ["Ó João, há quanto tempo não te via?", "Olá, Joana. Tudo bem contigo? Que bom ver te.",
             "Olá Joana, tudo bem contigo? Que bom ver te!", "Tudo ótimo. Olha, apresente que o Pedro trabalha comigo.",
             "Apresento-te a Inês e o Rui.", "Muito prazer, João. Eu sou o Pedro.", "Muito prazer, João.",
             "O prazer é meu, Tedo. A Rita também vem juntar conosco?", "A Rita Chega mais tarde ficou presa no trabalho.",
             "Então, vamos andando, que estou cheio de fome.", "Eu sou do Porto.", "Obrigado, Senhor.",
             "A camisola rosa é tua.", "Fui a Lisboa ontem.", "Chamo-me Ana.", "O meu nome é Rui.", "Aqui é a Joana!",
             "Pedro, anda cá.", "Anda cá, Pedro!", "Obrigado, Maria!", "De nada, Pedro.", "Esta é a Ana, a minha irmã.",
             "Já conheces o Tiago?", "A avó João foi.", "Bom dia, Catarina. Este é o Duarte.", "Sou a Beatriz e tu?",
             "Oh João, há quanto tempo não te via?", "Ó, Maria, anda cá.", "Oh, a Rita também vem?", "Eu sou Pedro.",
             "Sou Benfica desde pequeno.", "Eu sou Maria, muito prazer."]
    g["lines"] = []
    for line in lines:
        rn = cv.rule_names(line)
        g["lines"].append({"line": line, "rule_names": [list(x) for x in rn],
                           "checked": [list(x) for x in cv.parse_names(cv.RuleReader().ask([], line), line)],
                           "vocative": {n: cv.vocative(n, line) for n, _ in rn},
                           "own": {n: cv.says_own_name(n, line) for n, _ in rn}})
    g["written_in"] = [[n, l, cv.written_in(n, l)] for n, l in [("João", "ola joao!"), ("Inês", "a Ines chegou"),
                       ("Catarina", "Olá Katarina"), ("Joana", "Olá João"), ("Pedro", "O prazer é meu, Tedo.")]]
    g["check_role"] = [[n, r, l, cv.check_role(n, r, l)] for n, r, l in [
        ("Joana", "self", "Olá Joana, tudo bem contigo?"), ("Tedo", "self", "O prazer é meu, Tedo."),
        ("Ana", "self", "A Ana chegou."), ("Pedro", "mentioned", "Anda cá, Pedro!"), ("Pedro", "self", "Eu sou o Pedro.")]]

    conversations = [
        {"lines": [[0, [["João", "addressed"]]], [1, [["Joana", "addressed"]]]], "fixed": {}},
        {"lines": [[0, [["João", "addressed"]]], [1, None]], "fixed": {}},
        {"lines": [[0, [["Rita", "mentioned"]]], [1, []]], "fixed": {}},
        {"lines": [[0, [["Pedro", "self"]]], [1, [["Pedro", "addressed"]]], [0, [["Pedro", "mentioned"]]]], "fixed": {}},
        {"lines": [[0, [["João", "addressed"]]], [1, []]], "fixed": {"7": "João"}},
        {"lines": [[0, [["Joao", "addressed"]]], [1, [["Joana", "addressed"]]], [0, [["João", "addressed"]]]], "fixed": {}},
        {"lines": [[0, []], [1, [["Ana", "addressed"]]], [2, []]], "fixed": {}},
        {"lines": [[0, [["Pedro", "introduced"]]], [1, [["Pedro", "self"]]], [2, [["Pedro", "replied"]]], [0, []]], "fixed": {}},
        {"lines": [[0, [["João", "addressed"]]], [1, [["Joana", "addressed"]]], [0, [["Pedro", "introduced"]]],
                   [2, [["João", "addressed"], ["Pedro", "self"]]], [1, [["Tedo", "replied"], ["Rita", "mentioned"]]],
                   [0, [["Rita", "mentioned"]]], [2, []]], "fixed": {}},
    ]
    g["resolve"] = []
    for c in conversations:
        lines_ = [cv.Line(0.0, 1.0, spk, 1.0, "…", None if names is None else [tuple(x) for x in names])
                  for spk, names in c["lines"]]
        res = cv.resolve_names(lines_, {int(k): v for k, v in c["fixed"].items()})
        c["result"] = {str(spk): {"name": n.name, "score": None if math.isinf(n.score) else n.score,
                                  "evidence": n.evidence} for spk, n in res.items()}
        g["resolve"].append(c)

    spec = [{"kind": "quiet", "s": 1, "amp": 1e-4}, {"kind": "tone", "f": 200, "s": 1.5, "amp": 0.1},
            {"kind": "quiet", "s": 1, "amp": 1e-4}, {"kind": "tone", "f": 300, "s": 1, "amp": 0.1},
            {"kind": "quiet", "s": 0.3, "amp": 1e-4}, {"kind": "tone", "f": 300, "s": 1, "amp": 0.1},
            {"kind": "quiet", "s": 1, "amp": 1e-4}, {"kind": "tone", "f": 500, "s": 0.1, "amp": 0.1},
            {"kind": "quiet", "s": 1, "amp": 1e-4}, {"kind": "talk", "f": 250, "s": 18, "amp": 0.1},
            {"kind": "quiet", "s": 1, "amp": 1e-4}, {"kind": "tone", "f": 220, "s": 1.2, "amp": 0.1}]
    wav = synth(spec)
    seg = cv.TurnSegmenter(SR)
    turns = [t for s in range(0, len(wav), 1600) for t in seg.push(wav[s: s + 1600])] + seg.flush()
    g["segmenter"] = {"spec": spec, "seed": 1, "samples": len(wav),
                      "turns": [{"start": t.start, "end": t.end, "speech_s": t.speech_s, "cuts": t.cuts,
                                 "samples": len(t.audio)} for t in turns]}

    base = unit_vectors(11, 3)
    noise = unit_vectors(12, 30)
    room = unit_vectors(13, 2)
    room[1] -= (room[1] @ room[0]) * room[0]
    room[1] /= np.linalg.norm(room[1])
    who = [0, 1, 0, 2, 1, 1, 0, 2, 2, 0, 1, 0, 2, 0, 1, 2, 0, 1, 2, 0]
    secs = [2.0, 1.5, 0.6, 2.5, 0.4, 3.0, 1.2, 0.8, 2.2, 0.5, 1.9, 2.8, 0.3, 1.1, 2.4, 1.6, 0.7, 2.0, 1.3, 0.9]
    raw = [base[w] * 1.0 + noise[k] * 0.9 for k, w in enumerate(who)]
    g["voices"] = []
    for rules in (cv.VoiceRules(threshold=0.4, cut=0.3, merge_margin=0.2, new_margin=0.3),
                  cv.VoiceRules(threshold=0.5, cut=0.3, merge_margin=0.15, new_margin=0.35, room=room)):
        voices, lines_, steps = cv.Voices(rules), [], []
        for k, (r, s) in enumerate(zip(raw, secs)):
            e = rules.project(r)[0]
            spk, follows = voices.assign(e, s)
            lines_.append(cv.Line(k, k + 1, spk, s, emb=e, follows=follows))
            cv.relabel(lines_, voices)
            steps.append({"assigned": spk, "follows": follows, "labels": [ln.speaker for ln in lines_]})
        g["voices"].append({"rules": {"threshold": rules.threshold, "cut": rules.cut, "merge_margin": rules.merge_margin,
                                      "follow_below_s": rules.follow_below_s, "new_margin": rules.new_margin,
                                      "min_cut_side_s": rules.min_cut_side_s,
                                      "room_directions": [] if rules.room is None else rules.room.tolist()},
                            "raw": [r.tolist() for r in raw], "seconds": secs, "steps": steps})
    return g


def onnx_fixture(g: dict) -> None:
    """PyTorch embeddings of LCG clips, for the Kotlin ONNX check (only with a calibrated model exported here)."""
    from spectre import live
    try:
        ckpt = live.find_checkpoint()
    except SystemExit:
        return
    emb = live.Embedder(ckpt, "cpu")
    specs = [[{"kind": "tone", "f": 180, "s": 1.0, "amp": 0.1}, {"kind": "quiet", "s": 0.5, "amp": 0.01}],
             [{"kind": "talk", "f": 310, "s": 3.0, "amp": 0.2}]]
    g["onnx"] = {"model": emb.model_id, "clips": specs,
                 "embeddings": [emb(synth(s, seed=5))[0].tolist() for s in specs]}


if __name__ == "__main__":
    g = fixtures()
    onnx_fixture(g)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(g, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"✓ {OUT} ({OUT.stat().st_size / 1024:.0f} KB; ONNX embeddings for {g.get('onnx', {}).get('model', 'no model')})")
