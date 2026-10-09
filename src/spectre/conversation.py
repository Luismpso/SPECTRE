"""SPECTRE phase 4 — live conversation captions that work out who is who from what is said.

    python -m spectre.conversation                                  # live, from the microphone
    python -m spectre.conversation --file conversa.wav              # the same pipeline over a recording
    python -m spectre.conversation --file conversa.wav --realtime   # ...at real speed, as if it were live
    python -m spectre.conversation --llm none                       # captions and voices only, no names

Everything runs locally:

  microphone ─► turns (cut at pauses, and where the voice changes) ─► SPECTRE voice embedding
             ─► online clustering of the voices ─► Whisper transcription
             ─► a local LLM reads each new line and says how names appear in it
                (said as their own name / addressed / introduced / only mentioned)
             ─► turn-taking rules turn those cues into evidence for each voice ─► names, also on earlier lines

"Olá João" — the next *other* voice is probably João, and the speaker is not. "Olá Joana, tudo bem?" — the
previous other voice is probably Joana. "Eu sou o Pedro" — the speaker is Pedro. "A Rita chega mais tarde" —
nobody gets the name Rita. The LLM only *reads* (understanding a sentence is what language models are good at);
the code *decides* with simple, testable rules, so a small local model is enough, each LLM call is short (one
line) and every name comes with its evidence.

Audio is never written to disk.
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import warnings
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf

from . import live

ROLES = ("self", "addressed", "introduced", "mentioned")
STEP = 0.01                       # voice activity detection: 25 ms frames every 10 ms


# =========================================================================== turns
@dataclass
class Turn:
    start: float                                    # seconds since the start
    end: float
    audio: np.ndarray                               # 16 kHz mono
    speech_s: float                                 # seconds of detected speech
    cuts: list[int] = field(default_factory=list)   # short pauses inside the turn (sample index at 16 kHz)


class TurnSegmenter:
    """Streaming energy-based voice activity detection that cuts the audio into turns at pauses.

    A turn ends after `gap_s` of silence. A long turn ends at the first short pause after `soft_max_s` (so
    captions keep coming during a monologue) and never lasts more than `max_turn_s`. Shorter pauses inside a
    turn are remembered: if the voice changes there, the turn is cut (see split_by_voice). Works at the input
    sample rate and resamples each finished turn once, so there are no block-boundary artefacts."""

    def __init__(self, sr: int, gap_s: float = 0.6, soft_max_s: float = 6.0, soft_gap_s: float = 0.25,
                 max_turn_s: float = 15.0, cut_gap_s: float = 0.2, min_speech_s: float = 0.3,
                 margin_db: float = 10.0, min_db: float = -55.0, preroll_s: float = 0.25):
        self.sr, self.frame, self.hop = sr, int(0.025 * sr), int(STEP * sr)
        self.gap, self.soft_gap, self.cut_gap = (round(s / STEP) for s in (gap_s, soft_gap_s, cut_gap_s))
        self.soft_max, self.max_len = round(soft_max_s / STEP), round(max_turn_s / STEP)
        self.min_speech_s, self.margin_db, self.min_db = min_speech_s, margin_db, min_db
        self.pending = np.zeros(0, np.float32)
        self.levels: deque = deque(maxlen=1000)               # last 10 s of frame levels → noise floor
        self.floor = -120.0
        self.n = 0                                            # frames seen so far
        self.pre: deque = deque(maxlen=max(1, round(preroll_s / STEP)))
        self.hops: list[np.ndarray] | None = None             # the turn in progress, 10 ms at a time
        self.cuts: list[int] = []
        self.start_n = self.last_active = self.active = 0

    @property
    def speaking(self) -> float:
        """Seconds of speech in the turn in progress (0 when nobody is speaking)."""
        return self.active * STEP if self.hops is not None else 0.0

    def recent_audio(self, seconds: float = 3.0) -> np.ndarray:
        """The end of the turn in progress, at 16 kHz — to show who is speaking before the turn ends."""
        if not self.hops:
            return np.zeros(0, np.float32)
        return live.resample(np.concatenate(self.hops[-round(seconds / STEP):]), self.sr)

    def push(self, block: np.ndarray) -> list[Turn]:
        self.pending = np.concatenate([self.pending, np.asarray(block, np.float32).reshape(-1)])
        out = []
        while len(self.pending) >= self.frame:
            frame, hop = self.pending[: self.frame], self.pending[: self.hop]
            self.pending = self.pending[self.hop:]
            db = 10 * np.log10(float(np.mean(frame.astype(np.float64) ** 2)) + 1e-12)
            self.levels.append(db)
            if self.n % 50 == 0 and len(self.levels) >= 20:
                p10 = float(np.percentile(self.levels, 10))       # speech always has pauses: the quietest 10 %
                self.floor = p10 if len(self.levels) >= 300 else min(p10, -50.0)   # is noise (unless it starts mid-speech)
            active = db > max(self.floor + self.margin_db, self.min_db)
            if self.hops is None:
                self.pre.append(hop)
                if active:
                    self.hops, self.start_n = list(self.pre), self.n - len(self.pre) + 1
                    self.last_active, self.active = self.n, 1
            else:
                self.hops.append(hop)
                if active:
                    pause = self.n - self.last_active - 1
                    if pause >= self.cut_gap:                 # a short pause inside the turn
                        self.cuts.append(len(self.hops) - 1 - (pause + 1) // 2)
                    self.last_active, self.active = self.n, self.active + 1
                silent = self.n - self.last_active
                if silent >= self.gap or (len(self.hops) >= self.soft_max and silent >= self.soft_gap):
                    out += self._close(silent)
                elif len(self.hops) >= self.max_len:
                    out += self._close(0)
            self.n += 1
        return out

    def flush(self) -> list[Turn]:
        """End of the stream: close the turn in progress."""
        if self.hops is None:
            return []
        self.hops.append(self.pending)
        self.pending = np.zeros(0, np.float32)
        return self._close(self.n - 1 - self.last_active)

    def _close(self, trailing: int) -> list[Turn]:
        hops = self.hops[: len(self.hops) - max(0, trailing - 10)]            # keep 100 ms after the speech
        cuts = [round(c * self.hop * live.SR / self.sr) for c in self.cuts if 0 < c < len(hops)]
        speech_s, start = self.active * STEP, self.start_n * self.hop / self.sr
        self.hops, self.cuts, self.active = None, [], 0
        self.pre.clear()
        if speech_s < self.min_speech_s or not hops:
            return []
        raw = np.concatenate(hops)
        return [Turn(start, start + len(raw) / self.sr, live.resample(raw, self.sr), speech_s, cuts)]


def split_by_voice(turn: Turn, embedder, threshold: float, min_side_s: float = 0.6, window_s: float = 2.0) -> list[Turn]:
    """Someone answering quickly (a pause shorter than the turn gap) leaves several voices in one turn. At each
    short pause, compare the speech just before it with the speech just after it (up to the neighbouring pauses,
    at most `window_s`, at least `min_side_s`) and cut where the voices differ."""
    sr, n = live.SR, len(turn.audio)
    lo, win = int(min_side_s * sr), int(window_s * sr)
    cuts = [c for c in turn.cuts if lo <= c <= n - lo]
    if not cuts:
        return [turn]
    bounds, windows = [0, *cuts, n], []
    for k, c in enumerate(cuts, start=1):
        a = min(c - lo, max(bounds[k - 1], c - win))
        b = max(c + lo, min(bounds[k + 1], c + win))
        windows += [turn.audio[a:c], turn.audio[c:b]]
    E = embedder(windows)
    keep = [c for c, s in zip(cuts, (E[0::2] * E[1::2]).sum(axis=1)) if s < threshold]
    if not keep:
        return [turn]
    edges, per_sample = [0, *keep, n], turn.speech_s / max(n, 1)
    return [Turn(turn.start + a / sr, turn.start + b / sr, turn.audio[a:b], (b - a) * per_sample,
                 [x - a for x in turn.cuts if a < x < b]) for a, b in zip(edges[:-1], edges[1:])]


# =========================================================================== voices
CALIBRATION = "conversation.json"         # written next to the model by `python -m spectre.conv_eval`


@dataclass
class VoiceRules:
    """How turns are given to voices. The defaults work without calibration; `python -m spectre.conv_eval`
    measures the best values for a model on simulated one-microphone conversations and saves them next to it."""
    threshold: float = 0.65           # a turn joins a voice when it is at least this similar to it
    cut: float = 0.5                  # a short pause inside a turn is cut when the voices around it are below this
    merge_margin: float = 0.2         # two voices are merged when their similarity reaches threshold + this
    follow_below_s: float = 1.0       # turns with less speech follow the closest voice…
    new_margin: float = 0.45          # …unless they are below threshold − this: clearly someone new
    min_cut_side_s: float = 0.8       # speech needed on each side of a pause to judge a change of voice
    room: np.ndarray | None = None    # directions removed from every embedding (what one room adds to all voices)
    summary: str = ""                 # what the calibration measured

    @property
    def merge(self) -> float:
        return self.threshold + self.merge_margin

    @property
    def new_voice(self) -> float:
        return self.threshold - self.new_margin

    def project(self, E: np.ndarray) -> np.ndarray:
        """Room compensation: remove the room directions from the embeddings and renormalise them."""
        E = np.atleast_2d(np.asarray(E, float))
        if self.room is not None and len(self.room):
            E = E - (E @ self.room.T) @ self.room
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    @classmethod
    def for_model(cls, ckpt: Path, model_id: str) -> "VoiceRules":
        """The calibration saved next to the model, or the defaults when there is none (or it is for another model)."""
        f = Path(ckpt).parent / CALIBRATION
        d = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        if d.get("model") != model_id:
            return cls()
        return cls(threshold=d["threshold"], cut=d["cut"], merge_margin=d["merge_margin"],
                   follow_below_s=d["follow_below_s"], new_margin=d["new_margin"], min_cut_side_s=d["min_cut_side_s"],
                   room=np.asarray(d["room_directions"], float) if d.get("room_directions") else None,
                   summary=f"{100 * d['accuracy']:.0f} % of the speech to the right person in simulated rooms")


class Voices:
    """Online speaker clustering. Turns with enough speech lead: they shape the voices. Short turns, whose
    embeddings are unreliable, follow the closest voice and move if a closer one appears later. Enrolled people
    (phase 3) are fixed voices."""

    def __init__(self, rules: VoiceRules, enrolled: dict[str, np.ndarray] | None = None):
        self.rules = rules
        self.clusters: dict[int, dict] = {}
        self.next_label = 1
        for name, c in (enrolled or {}).items():
            self.clusters[len(self.clusters)] = {"sum": rules.project(c)[0], "embs": [], "seconds": 0.0,
                                                 "fixed": name, "label": name}

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-12)

    def closest(self, emb: np.ndarray) -> tuple[int | None, float]:
        best, best_sim = None, -1.0
        for cid, c in self.clusters.items():
            sim = float(self._unit(c["sum"]) @ emb)
            if sim > best_sim:
                best, best_sim = cid, sim
        return best, best_sim

    def assign(self, emb: np.ndarray, seconds: float) -> tuple[int, bool]:
        """(voice id, True if the turn only follows that voice). A new voice when nobody is close enough."""
        best, sim = self.closest(emb)
        if best is not None and seconds < self.rules.follow_below_s and sim >= self.rules.new_voice:
            return best, True
        if best is None or sim < self.rules.threshold:
            best = max(self.clusters, default=-1) + 1
            self.clusters[best] = {"sum": np.zeros_like(emb, dtype=float), "embs": [], "seconds": 0.0,
                                   "fixed": None, "label": f"Speaker {self.next_label}"}
            self.next_label += 1
        c = self.clusters[best]
        if c["fixed"] is None:                                 # enrolled voices keep their reference
            c["sum"] = c["sum"] + emb * max(seconds, 0.5)
        c["embs"].append(emb)
        c["seconds"] += seconds
        return best, False

    def unassign(self, cid: int, emb: np.ndarray, seconds: float) -> None:
        """Undo an assignment (the turn was only noise); a speaker left without turns disappears."""
        c = self.clusters.get(cid)
        k = next((k for k, e in enumerate(c["embs"]) if e is emb), None) if c else None
        if k is None:
            return
        del c["embs"][k]
        c["seconds"] -= seconds
        if c["fixed"] is None:
            c["sum"] = c["sum"] - emb * max(seconds, 0.5)
            if not c["embs"]:
                del self.clusters[cid]
                if c["label"] == f"Speaker {self.next_label - 1}":
                    self.next_label -= 1

    def merge(self) -> dict[int, int]:
        """Merge clusters that turned out to be the same voice; returns {removed id: kept id}."""
        moved: dict[int, int] = {}
        changed = True
        while changed:
            changed = False
            ids = sorted(self.clusters)
            for a in ids:
                for b in ids:
                    if a >= b or a not in self.clusters or b not in self.clusters:
                        continue
                    A, B = self.clusters[a], self.clusters[b]
                    if A["fixed"] and B["fixed"]:
                        continue
                    if float(self._unit(A["sum"]) @ self._unit(B["sum"])) >= self.rules.merge:
                        keep, drop = (a, b) if (A["fixed"] or A["seconds"] >= B["seconds"]) and not B["fixed"] else (b, a)
                        K, D = self.clusters[keep], self.clusters.pop(drop)
                        if K["fixed"] is None:
                            K["sum"] = K["sum"] + D["sum"]
                        K["embs"] += D["embs"]
                        K["seconds"] += D["seconds"]
                        moved = {k: (keep if v == drop else v) for k, v in moved.items()}
                        moved[drop] = keep
                        changed = True
        return moved

    def fixed_names(self) -> dict[int, str]:
        return {cid: c["fixed"] for cid, c in self.clusters.items() if c["fixed"]}


# =========================================================================== reading names with an LLM
SYSTEM_PROMPT = """You read one new line of a transcribed conversation (the previous lines are only context) and
list the people's names written in that NEW line, saying how each one appears:
- "self": the speaker says it is their own name ("Eu sou o Pedro", "Chamo-me Ana", "O meu nome é Rui", "Aqui é a Joana").
- "addressed": the speaker talks TO that person, calling them by name ("Olá João", "Obrigado, Maria", "Pedro, anda cá",
  "O prazer é meu, Pedro"). A name set off by a comma at the start or end of a sentence is someone being addressed.
- "introduced": the speaker presents that person to someone ("apresento-te o Pedro", "esta é a Ana").
- "mentioned": the person is only talked about ("A Rita chega mais tarde").
Only names written in the NEW line count, spelled as they are written there. Titles or family words on their own
("professor", "mãe") are not names. If the new line has no names, the list is empty."""

SCHEMA = {"type": "object", "required": ["names"], "properties": {"names": {"type": "array", "items": {
    "type": "object", "required": ["name", "role"],
    "properties": {"name": {"type": "string"}, "role": {"type": "string", "enum": list(ROLES)}}}}}}


def fold(text: str) -> str:
    """Lower-case, accent-free form used to compare names ('João' == 'joao')."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower().strip()


def written_in(name: str, line: str) -> bool:
    """Whether a name really appears in the line (small LLMs copy names from the context)."""
    words = re.findall(r"[^\W\d_]+", fold(line))
    parts = [p for p in re.findall(r"[^\W\d_]+", fold(name)) if len(p) > 1]
    return bool(parts) and any(w == p or (len(p) >= 4 and SequenceMatcher(None, w, p).ratio() >= 0.85)
                               for w in words for p in parts)


SELF_CUES = {"sou", "chamo", "chamam", "nome", "aqui", "fala", "am", "name"}


def says_own_name(name: str, line: str) -> bool:
    """A self-introduction word shortly before the name ("Eu sou o Pedro", "Chamo-me Ana", "O meu nome é Rui",
    "Aqui é a Joana"). Without one, "self" is a misreading — "O prazer é meu, Pedro" calls Pedro."""
    words = re.findall(r"[^\W\d_]+", fold(line))
    first = next((p for p in re.findall(r"[^\W\d_]+", fold(name)) if len(p) > 1), "")
    return any((w == first or (len(first) >= 4 and SequenceMatcher(None, w, first).ratio() >= 0.85))
               and SELF_CUES & set(words[max(0, k - 3): k]) for k, w in enumerate(words)) if first else False


CALLING = r"(?:ola|oi|bom dia|boa tarde|boa noite|adeus|tchau|ate logo|ate amanha|muito prazer|prazer|bem-vind[oa]|hello|hi)"
ANSWERING = r"(?:muito obrigad[oa]|obrigad[oa]|o prazer e meu|de nada|igualmente|desculp[ae]|thanks|thank you)"


def vocative(name: str, line: str) -> str | None:
    """How a name calls someone, from the words around it: "answer" right after thanks or a reply ("Obrigado,
    Maria", "O prazer é meu, Pedro") — that person spoke before; "call" after a greeting, after the particle "ó"
    (Whisper also writes "oh"), or set off by a comma ("Olá, João", "Ó João", "Oh João", "Pedro, anda cá", "Anda cá,
    Pedro!"); None otherwise."""
    first = next((p for p in re.findall(r"[^\W\d_]+", fold(name)) if len(p) > 1), "")
    if not first:
        return None
    t, n = fold(line), re.escape(first)
    marked = fold(re.sub(r"(?<![^\W\d_])(?:[óÓôÔ]|[Oo]h)(?=[\s,])", "\x01", line))   # the particle, before folding
    if re.search(rf"\b{ANSWERING}\s*[,!]?\s*{n}\b", t):
        return "answer"
    if (re.search(rf"\b{CALLING}\s*[,!]?\s*{n}\b", t) or re.search(rf"\x01[\s,]+{n}\b", marked)
            or re.search(rf",\s*{n}\b\s*(?:[.!?…]|$)", t) or re.search(rf"(?:^|[.!?]\s+){n}\s*,", t)):
        return "call"
    return None


def check_role(name: str, role: str, line: str) -> str | None:
    """Correct the usual misreadings with the words themselves: "self" needs a self-introduction ("Olá Joana" calls
    Joana); a vocative is someone addressed; one that answers ("O prazer é meu, Pedro") points back to that person."""
    v = vocative(name, line)
    if role == "self" and not says_own_name(name, line):
        role = "addressed" if v else None
    elif role == "mentioned" and v:
        role = "addressed"
    return "replied" if role == "addressed" and v == "answer" else role


PRIORITY = ("self", "addressed", "replied", "introduced", "mentioned")


def parse_names(answer: str, line: str) -> list[tuple[str, str]]:
    """LLM answer (JSON) → [(name, role)] keeping only names written in the line, one role per name."""
    m = re.search(r"\{.*\}", answer or "", re.S)
    try:
        items = json.loads(m.group(0)).get("names", []) if m else []
    except (json.JSONDecodeError, AttributeError):
        return []
    best: dict[str, tuple[str, str]] = {}
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        name, role = str(it.get("name", "")).strip(), str(it.get("role", "")).strip().lower()
        if role not in ROLES or not name or not written_in(name, line):
            continue
        role = check_role(name, role, line)
        if role is None:                                                # a misreading: not evidence of anything
            continue
        k = fold(name)
        if k not in best or PRIORITY.index(role) < PRIORITY.index(best[k][1]):   # one role per name
            best[k] = (name, role)
    return list(best.values())


def _user_prompt(context: list[str], line: str) -> str:
    ctx = "\n".join(context[-4:]) or "(start of the conversation)"
    return f"Previous lines:\n{ctx}\n\nNew line:\n{line}"


class OllamaReader:
    """A model served by a local Ollama (https://ollama.com), with JSON-schema structured output."""
    NUM_CTX = 2048          # prompts are short (< 600 tokens): a small context leaves GPU memory for Whisper

    def __init__(self, model: str, host: str = "http://127.0.0.1:11434", timeout: float = 120):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # local: never through a proxy

    def check(self) -> str | None:
        """None if ready, otherwise what to do."""
        try:
            with self.http.open(self.host + "/api/tags", timeout=3) as r:
                models = [m["name"] for m in json.load(r).get("models", [])]
        except (urllib.error.URLError, OSError, ValueError):
            return (f"Ollama is not running at {self.host}. Install it from https://ollama.com and run "
                    f"'ollama pull {self.model}'.")
        if (self.model if ":" in self.model else self.model + ":latest") not in models:
            return f"The model '{self.model}' is not in Ollama yet: run 'ollama pull {self.model}'."
        return None

    def warm(self) -> None:
        """Load the model now (an empty chat loads it), so the first line is read without waiting."""
        self._chat({"model": self.model, "messages": [], "stream": False, "keep_alive": "30m",
                    "options": {"num_ctx": self.NUM_CTX}})

    def ask(self, context: list[str], line: str) -> str:
        return self._chat({"model": self.model, "stream": False, "format": SCHEMA,
                           "options": {"temperature": 0, "num_ctx": self.NUM_CTX},
                           "keep_alive": "30m",
                           "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                        {"role": "user", "content": _user_prompt(context, line)}]})["message"]["content"]

    def _chat(self, body: dict) -> dict:
        req = urllib.request.Request(self.host + "/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with self.http.open(req, timeout=self.timeout) as r:
            return json.load(r)


class HFReader:
    """Any Hugging Face chat model through transformers (no Ollama needed)."""

    def __init__(self, model: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to(self.device).eval()

    def check(self) -> str | None:
        return None

    def warm(self) -> None:
        pass

    def ask(self, context: list[str], line: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT + "\nReply with JSON only, for example "
                 '{"names": [{"name": "João", "role": "addressed"}]} or {"names": []}.'},
                {"role": "user", "content": _user_prompt(context, line)}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=120, do_sample=False)
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


# =========================================================================== reading names without an LLM
FIRST_NAMES = frozenset(fold(n) for n in """
Abel Abílio Adão Adelino Adriano Afonso Agostinho Albano Alberto Albino Alexandre Alfredo Álvaro Américo Amílcar André
Ângelo Aníbal António Antônio Armando Arnaldo Artur Arthur Augusto Aurélio Baltasar Benjamim Bento Bernardo Bruno
Caetano Caio Camilo Carlos Cauã Celso César Cristiano Cristóvão Custódio Daniel Dário David Davi Diego Diogo Dinis
Domingos Duarte Edgar Edson Eduardo Elias Emanuel Emílio Enzo Ernesto Estêvão Eugénio Fabiano Fábio Fausto Felipe
Félix Fernando Filipe Flávio Francisco Frederico Gabriel Gaspar Gil Gilberto Gonçalo Guilherme Gustavo Heitor Hélder
Hélio Henrique Horácio Hugo Humberto Igor Inácio Isaac Ivo Jaime João Joaquim Jonas Jorge José Josué Juliano Júlio
Kevin Leandro Leonardo Leonel Lorenzo Lourenço Lucas Luciano Lúcio Luís Luiz Manuel Marcelo Marcos Marco Mário Martim
Mateus Matheus Matias Maurício Miguel Moisés Murilo Natan Nélson Nicolau Noah Norberto Nuno Octávio Otávio Orlando
Óscar Osvaldo Patrício Paulo Pedro Pietro Rafael Raimundo Ramiro Raul Reinaldo Renan Renato Ricardo Roberto Rodrigo
Rogério Romeu Ronaldo Rúben Rui Salvador Samuel Sandro Santiago Sebastião Sérgio Silvano Silvestre Simão Tadeu Telmo
Teodoro Thiago Tiago Tomás Valentim Valter Vasco Vicente Victor Vítor Vinícius Wagner Wesley William Xavier Yuri Zé
Adelaide Adriana Alexandra Alice Amanda Amélia Ana Andreia Ângela Anabela Antónia Aurora Bárbara Beatriz Benedita
Bianca Bruna Camila Carla Carlota Carmen Carolina Catarina Cecília Célia Clara Cláudia Constança Cristiana Cristina
Daniela Débora Diana Elisa Elisabete Ema Emília Eva Fabiana Fátima Fernanda Filipa Flávia Francisca Gabriela Giovanna
Glória Graça Helena Heloísa Inês Irene Isabel Isabela Isadora Ivone Jéssica Joana Joaquina Júlia Juliana Laís Lara
Larissa Laura Leonor Letícia Lídia Lívia Lorena Lúcia Luana Luísa Luiza Madalena Mafalda Manuela Mara Márcia Margarida
Maria Mariana Marina Marisa Marta Matilde Melissa Mónica Natália Natacha Nádia Olívia Paula Patrícia Pietra Priscila
Raquel Rebeca Regina Renata Rita Rosa Rosana Rosário Sabrina Salomé Sandra Sara Sílvia Simone Sofia Sónia Sophia
Susana Tânia Tatiana Teresa Thaís Valentina Valéria Vanessa Vera Verónica Vitória Viviane Yara Yasmin Zélia
""".split())
NOT_NAMES = frozenset(fold(w) for w in """senhor senhora sr sra dona dom doutor doutora dr dra professor professora
engenheiro engenheira deus mãe pai mano mana filho filha amigo amiga querido querida pessoal malta gente chefe menino
menina rapaz rapariga tio tia avó avô primo prima""".split())
SELF_INTRO = (r"\b(?:eu\s+)?sou\s+(?:o|a)\s+{n}\b", r"\bchamo-me\s+{n}\b", r"\bme\s+chamo\s+{n}\b",
              r"\bmeu\s+nome\s+e\s+{n}\b", r"\baqui\s+(?:e|fala)\s+(?:o|a)\s+{n}\b")
SELF_INTRO_BARE = r"\b(?:eu\s+)?sou\s+{n}\b"      # "Eu sou Pedro": without the article, only for common first names
INTRODUCING = (r"\bapresent[\w-]*\s+(?:[\w,]+\s+){{0,4}}?(?:o|a)\s+{n}\b", r"\b(?:este|esta)\s+e\s+(?:o|a)\s+{n}\b",
               r"\bconhec[\w-]*\s+(?:o|a)\s+{n}\b")


def name_role(name: str, line: str) -> str:
    """How a name is used, from the words around it: its own name, introduced, called (a vocative) or mentioned."""
    t, n = fold(line), re.escape(fold(name))
    if any(re.search(p.format(n=n), t) for p in SELF_INTRO) or (
            fold(name) in FIRST_NAMES and re.search(SELF_INTRO_BARE.format(n=n), t)):
        return "self"
    if any(re.search(p.format(n=n), t) for p in INTRODUCING):
        return "introduced"
    return "addressed" if vocative(name, line) else "mentioned"


def rule_names(line: str) -> list[tuple[str, str]]:
    """Names in a line without a language model: capitalised common first names, and any other capitalised word
    used as a name (called, introduced, said as one's own); a sentence never starts with an unknown name."""
    out, start = [], True
    for m in re.finditer(r"[^\W\d_]+|[.!?…]", line):
        word = m.group(0)
        if word in ".!?…":
            start = True
            continue
        first_word, start = start, False
        key = fold(word)
        if not word[0].isupper() or len(key) < 2 or key in NOT_NAMES:
            continue
        known = key in FIRST_NAMES
        if first_word and not known:
            continue
        role = name_role(word, line)
        if known or role != "mentioned":
            out.append((word, role))
    return out


class RuleReader:
    """Reads names with rules instead of a language model (instant, nothing to install; used on the phone)."""

    def check(self) -> str | None:
        return None

    def warm(self) -> None:
        pass

    def ask(self, context: list[str], line: str) -> str:
        return json.dumps({"names": [{"name": n, "role": r} for n, r in rule_names(line)]}, ensure_ascii=False)


def make_reader(spec: str):
    """'ollama:gemma3:12b' · 'hf:Qwen/Qwen2.5-3B-Instruct' · 'rules' · 'none'."""
    if spec in ("none", ""):
        return None
    if spec == "rules":
        return RuleReader()
    kind, _, model = spec.partition(":")
    if kind == "ollama" and model:
        return OllamaReader(model)
    if kind == "hf" and model:
        return HFReader(model)
    sys.exit(f"Unknown --llm '{spec}'. Use ollama:<model>, hf:<model>, rules or none.")


# =========================================================================== deciding who is who
@dataclass(eq=False)                              # lines are compared by identity
class Line:
    start: float
    end: float
    speaker: int
    seconds: float
    text: str | None = None                       # None while being transcribed
    names: list[tuple[str, str]] | None = None    # None until the LLM has read it
    emb: np.ndarray | None = field(default=None, repr=False)   # room-compensated voice embedding
    raw: np.ndarray | None = field(default=None, repr=False)   # as the model gives it (what --remember saves)
    follows: bool = False                         # a short turn: it follows the closest voice


@dataclass
class Naming:
    name: str
    score: float
    evidence: list[str]


WEIGHTS = {"self": 3.0, "spoke_after": 2.0, "answered_with": 2.0, "said_it": -3.0}


def resolve_names(lines: list[Line], fixed: dict[int, str], min_score: float = 2.0) -> dict[int, Naming]:
    """Turn the per-line cues into at most one name per voice, with turn-taking rules:
    - "Eu sou o Pedro"                    → the speaker is Pedro                                    (+3)
    - "Olá João" / "apresento-te o João"  → the next *other* voice is João                          (+2)
    - "Olá Joana, tudo bem?"              → the previous *other* voice is Joana (answering by name) (+2)
      (a name in an answer — "Obrigado, Maria", "O prazer é meu, Pedro" — only points back)
    - saying a name to or about someone   → the speaker is not that person                         (−3)
      ("Olá João", "A Rita chega mais tarde": nobody gains the name Rita)
    The evidence adds up, so one misheard or misread line can be outweighed. A voice needs at least
    `min_score`; each name goes to one voice, strongest evidence first; enrolled people keep their names."""
    score: dict[tuple[int, str], float] = defaultdict(float)
    evidence: dict[tuple[int, str], list[str]] = defaultdict(list)
    first: dict[tuple[int, str], int] = {}
    spelling: dict[str, Counter] = defaultdict(Counter)
    sp = [ln.speaker for ln in lines]

    def other(i: int, step: int) -> int | None:
        j = i + step
        while 0 <= j < len(lines):
            if sp[j] != sp[i]:
                return j
            j += step
        return None

    def add(spk: int, key: str, w: float, why: str, i: int) -> None:
        score[(spk, key)] += w
        if w > 0:                                               # the reasons shown are the positive ones
            evidence[(spk, key)].append(why)
            first.setdefault((spk, key), i)

    for i, ln in enumerate(lines):
        for name, role in ln.names or []:
            key = fold(name)
            spelling[key][name] += 1
            if role == "self":
                add(sp[i], key, WEIGHTS["self"], f'line {i + 1}: said their name ("{name}")', i)
                continue
            add(sp[i], key, WEIGHTS["said_it"], "", i)
            if role in ("addressed", "introduced"):                     # not "replied"
                j = other(i, +1)
                if j is not None:
                    verb = "called" if role == "addressed" else "introduced"
                    add(sp[j], key, WEIGHTS["spoke_after"], f'line {j + 1}: spoke right after "{name}" was {verb} '
                        f'(line {i + 1})', j)
            if role in ("addressed", "replied"):
                j = other(i, -1)
                if j is not None:
                    add(sp[j], key, WEIGHTS["answered_with"], f'line {i + 1}: called "{name}" in the reply to '
                        f'line {j + 1}', i)

    taken = {fold(n) for n in fixed.values()}
    result = {spk: Naming(n, float("inf"), ["recognised by voice (enrolled)"]) for spk, n in fixed.items()}
    candidates = sorted(((s, -first[k], k) for k, s in score.items() if s >= min_score), reverse=True)
    for s, _, (spk, key) in candidates:
        if spk in result or key in taken:
            continue
        name = max(spelling[key].items(), key=lambda kv: (kv[1], not kv[0].isascii(), kv[0][:1].isupper()))[0]
        # ↑ the spelling used most often; on a tie the one with accents ("João" over "Joao"), then capitalised
        result[spk] = Naming(name, s, evidence[(spk, key)])
        taken.add(key)
    return result


# =========================================================================== transcription
HALLUCINATIONS = ("amara.org", "obrigado por assistir", "inscreva-se", "subscreva o canal", "legendas pela comunidade")


class Transcriber:
    """Whisper through transformers (it uses the PyTorch already installed for SPECTRE)."""

    def __init__(self, model: str = "openai/whisper-large-v3-turbo", language: str = "pt", device: str | None = None):
        import torch
        import transformers
        from transformers import pipeline
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()             # (download progress is still shown)
        warnings.filterwarnings("ignore", module="transformers")
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.pipe = pipeline("automatic-speech-recognition", model=model, device=dev,
                             dtype=torch.float16 if dev == "cuda" else torch.float32)
        if dev == "cuda":
            self(np.zeros(live.SR, np.float32))               # the first call on a GPU is slow: get it done now

    def __call__(self, audio: np.ndarray) -> str:
        kw = {"task": "transcribe"} | ({"language": self.language} if self.language != "auto" else {})
        text = self.pipe({"raw": audio, "sampling_rate": live.SR}, generate_kwargs=kw)["text"].strip()
        return "" if any(h in text.lower() for h in HALLUCINATIONS) else text


# =========================================================================== the conversation
def relabel(lines: list[Line], voices: Voices) -> None:
    """After a change: merge voices that turned out to be one, and let short turns follow the closest voice."""
    moved = voices.merge()
    for ln in lines:
        ln.speaker = moved.get(ln.speaker, ln.speaker)
        if ln.follows:
            cid, _ = voices.closest(ln.emb)
            ln.speaker = ln.speaker if cid is None else cid


class Conversation:
    """Holds the lines, the voices and the current names. A turn is processed in two stages: hear (voice,
    then transcription — the caption appears) and read (the LLM — names may appear, also on earlier lines)."""

    def __init__(self, embedder, transcriber, reader, voices: Voices, on_change=lambda: None):
        self.embedder, self.transcriber, self.reader, self.voices = embedder, transcriber, reader, voices
        self.lines: list[Line] = []
        self.names: dict[int, Naming] = resolve_names([], voices.fixed_names())
        self.lock = threading.RLock()
        self.on_change = on_change
        self.speaking: tuple[int, str] | None = None     # (voice, label) of the turn in progress
        self.reader_error: str | None = None
        self.problem: str | None = None

    def embed(self, wavs) -> np.ndarray:
        """Voice embeddings with the room compensation of the rules."""
        return self.voices.rules.project(self.embedder(wavs))

    def split(self, turn: Turn) -> list[Turn]:
        rules = self.voices.rules
        return split_by_voice(turn, self.embed, rules.cut, rules.min_cut_side_s)

    def hear(self, turn: Turn) -> Line | None:
        """Voice and transcription of a finished turn; returns its line (None if it was only noise)."""
        raw = self.embedder(turn.audio)[0]
        emb = self.voices.rules.project(raw)[0]
        with self.lock:
            spk, follows = self.voices.assign(emb, turn.speech_s)
            line = Line(turn.start, turn.end, spk, turn.speech_s, emb=emb, raw=raw, follows=follows)
            self.lines.append(line)
            self._relabel()
            self._update_names()                                # the voice may already have a name
        self.on_change()
        try:
            text = self.transcriber(turn.audio)
        except Exception as err:                                # keep going; the error is shown
            self.problem, text = f"transcription failed: {type(err).__name__}: {err}", ""
        with self.lock:
            if text:
                line.text = text
            else:                                               # a cough, a door...: forget it
                self.lines.remove(line)
                if not line.follows:
                    self.voices.unassign(line.speaker, emb, turn.speech_s)
                self._relabel()
            self._update_names()
        self.on_change()
        return line if text else None

    def read(self, line: Line) -> None:
        """Let the LLM read a line (with the previous lines as context) and update the names."""
        with self.lock:
            if line not in self.lines:
                return
            text = line.text or ""
            context = [ln.text for ln in self.lines[: self.lines.index(line)] if ln.text]
        names: list[tuple[str, str]] = []
        if self.reader is not None and text:
            try:
                names = parse_names(self.reader.ask(context, text), text)
                self.reader_error = None
            except Exception as err:                            # keep captioning even if the LLM fails
                self.reader_error = f"{type(err).__name__}: {err}"
        with self.lock:
            line.names = names
            self._update_names()
        self.on_change()

    def peek(self, audio: np.ndarray) -> None:
        """Who is speaking in the turn in progress, before it ends — with the same rules, as if the turn were
        already a line: right after "Olá João", a voice never heard before is shown as João at once."""
        emb = self.embed(audio)[0]
        with self.lock:
            cid, sim = self.voices.closest(emb)
            if cid is None or sim < self.voices.rules.threshold:
                cid = max(self.voices.clusters, default=-1) + 1     # the id this new voice will get
            names = resolve_names(self.lines + [Line(0.0, 0.0, cid, 0.0)], self.voices.fixed_names())
            who = names[cid].name if cid in names else self.voices.clusters.get(cid, {}).get("label", "a new voice")
            self.speaking = (cid, who)
        self.on_change()

    def _relabel(self) -> None:
        relabel(self.lines, self.voices)

    def _update_names(self) -> None:
        self.names = resolve_names(self.lines, self.voices.fixed_names())

    def label(self, spk: int) -> str:
        n = self.names.get(spk)
        return n.name if n else self.voices.clusters.get(spk, {}).get("label", f"Speaker {spk}")

    def voices_heard(self) -> list[int]:
        return list(dict.fromkeys(ln.speaker for ln in self.lines))


# =========================================================================== output
COLORS = ["cyan", "magenta", "green", "yellow", "bright_blue", "red", "bright_magenta", "bright_green"]


def _clock(t: float, tenths: bool = False) -> str:
    t = round(t, 1) if tenths else int(t)
    return f"{int(t // 60):02d}:{t % 60:04.1f}" if tenths else f"{t // 60:02d}:{t % 60:02d}"


def render(conv: Conversation, status: str, width: int = 100, height: int = 30):
    """The live panel: as many of the latest lines as fit, who is speaking now, and the voices so far."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text
    inner = max(20, width - 4)
    rows_of = lambda t: 1 + len(t.plain) // inner                # rough height of a wrapped line
    with conv.lock:
        voices = conv.voices_heard()
        footer = Text("voices: " + ", ".join(conv.label(s) for s in voices) if voices else "", style="dim")
        now = None
        if conv.speaking is not None:
            cid, who = conv.speaking
            now = Text(f"▶ {who} is speaking…", style=f"bold {COLORS[cid % len(COLORS)]}")
        budget = max(3, height - 5 - rows_of(footer) - (rows_of(now) if now else 0))
        rows, used = [], 0
        for ln in reversed(conv.lines):
            t = Text(f"{_clock(ln.start)}  ", style="dim")
            t.append(f"{conv.label(ln.speaker)}: ", style=f"bold {COLORS[ln.speaker % len(COLORS)]}")
            t.append("…" if ln.text is None else ln.text)
            if rows and used + rows_of(t) > budget:
                break
            rows.append(t)
            used += rows_of(t)
    body = rows[::-1] or [Text("Waiting for someone to speak…", style="dim")]
    return Panel(Group(*body, *([now] if now else []), Text(""), footer), title="SPECTRE · live captions",
                 subtitle=status, subtitle_align="left", border_style="bright_black")


class PlainPrinter:
    """Output without the live panel: each line once it is transcribed, and a note when a voice gets a name."""

    def __init__(self, conv: Conversation, out=lambda s: print(s, flush=True)):
        self.conv, self.out = conv, out
        self.printed: set[Line] = set()
        self.told: dict[int, str | None] = {}
        self.lock = threading.Lock()

    def __call__(self) -> None:
        conv = self.conv
        with self.lock, conv.lock:
            for ln in conv.lines:
                if ln.text and ln not in self.printed:
                    self.printed.add(ln)
                    self.out(f"[{_clock(ln.start, True)}] {conv.label(ln.speaker)}: {ln.text}")
            for spk in dict.fromkeys(ln.speaker for ln in conv.lines if ln in self.printed):
                c = conv.voices.clusters.get(spk, {})
                n = conv.names.get(spk)
                name = n.name if n else None
                if c.get("fixed") or self.told.get(spk) == name or (name is None and spk not in self.told):
                    continue
                base = c.get("label", f"Speaker {spk}")
                self.out(f"   ↳ {base} is {name} — {n.evidence[0]}" if name else f"   ↳ {base} has no name any more")
                self.told[spk] = name


def transcript(conv: Conversation) -> str:
    with conv.lock:
        out = [f"[{_clock(ln.start, True)}] {conv.label(ln.speaker)}: {ln.text}" for ln in conv.lines if ln.text]
        heard = set(conv.voices_heard())
        found = {s: n for s, n in conv.names.items() if s in heard}
        if found:
            out.append("\nHow the names were found:")
            for _, n in sorted(found.items(), key=lambda kv: kv[1].name):
                out.append(f"  {n.name}: " + "; ".join(n.evidence[:3]))
    return "\n".join(out)


# =========================================================================== running it
class Runner:
    """Two background workers so the audio never waits for the models and captions never wait for the LLM."""

    def __init__(self, conv: Conversation):
        self.conv, self.turns, self.to_read = conv, queue.Queue(), queue.Queue()
        self.skip = self.closed = False
        self.threads = [threading.Thread(target=self._hear_loop, daemon=True),
                        threading.Thread(target=self._read_loop, daemon=True)]
        for t in self.threads:
            t.start()

    def _hear_loop(self) -> None:
        while (turn := self.turns.get()) is not None:
            if self.skip:
                continue
            try:
                for part in self.conv.split(turn):
                    if (line := self.conv.hear(part)) is not None:
                        self.to_read.put(line)
            except Exception as err:                            # never let one bad turn stop the captions
                self.conv.problem = f"{type(err).__name__}: {err}"
        self.to_read.put(None)

    def _read_loop(self) -> None:
        while (line := self.to_read.get()) is not None:
            try:
                self.conv.read(line)
            except Exception as err:
                self.conv.problem = f"{type(err).__name__}: {err}"

    def finish(self, skip_pending: bool = False) -> None:
        """Wait for the turns already queued (or skip them) and stop the workers."""
        self.skip = self.skip or skip_pending
        if not self.closed:
            self.turns.put(None)
            self.closed = True
        for t in self.threads:
            while t.is_alive():
                t.join(0.2)                                     # short waits keep Ctrl+C working on Windows


def run(args) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sd = mic = None
    if not args.file:                                           # no microphone? say so before loading models
        sd = live._sounddevice()
        mic, mic_sr = live.input_device(sd, args.device)

    ckpt = args.ckpt or live.find_checkpoint()
    emb = live.Embedder(ckpt, "cpu" if args.cpu else None)
    bank = live.SpeakerBank(args.bank)
    enrolled: dict[str, np.ndarray] = {}
    if bank.speakers:
        if bank.data["model"] == emb.model_id:
            names, C = bank.centroids()
            enrolled = dict(zip(names, C))
        else:
            print(f"(Not using {args.bank}: it was made with another model.)")
    rules = VoiceRules.for_model(ckpt, emb.model_id)
    if args.cluster_threshold is not None:
        rules.threshold = args.cluster_threshold
    if args.merge_threshold is not None:
        rules.merge_margin = args.merge_threshold - rules.threshold
    voices = Voices(rules, enrolled)

    reader = make_reader(args.llm)
    if reader is not None:
        problem = reader.check()
        if problem is None:
            print(f"Loading the language model ({args.llm})…")
            try:
                reader.warm()
            except Exception as err:
                problem = f"Could not load {args.llm}: {err}"
        if problem:
            print(f"⚠ {problem}\n  Continuing without names (captions and voices only).")
            reader = None
    names_status = ("names: off" if reader is None else "names: rules" if args.llm == "rules"
                    else f"names: {args.llm.split(':', 1)[1]}")
    print(f"Loading Whisper ({args.whisper})…")
    transcriber = Transcriber(args.whisper, args.language, "cpu" if args.cpu else None)
    print(f"Voices: {emb.model_id} on {emb.device.type} · same speaker above {rules.threshold:.2f}"
          + (f" · enrolled: {', '.join(enrolled)}" if enrolled else ""))
    if rules.summary:
        print(f"  calibrated for one-microphone conversations (room compensation): {rules.summary}")
    else:
        print("  not calibrated for conversations yet: run 'python -m spectre.conv_eval' once (a few minutes)\n"
              "  so that voices sharing one room and one microphone are told apart better")

    conv = Conversation(emb, transcriber, reader, voices)
    state = {"phase": "starting"}

    def status() -> str:
        bits = [state["phase"], names_status]
        bits += [f"LLM error: {conv.reader_error}"] if conv.reader_error else []
        bits += [f"error: {conv.problem}"] if conv.problem else []
        return " · ".join(bits)

    view = None
    if sys.stdout.isatty() and not args.plain:
        from rich.live import Live
        view = Live(refresh_per_second=8)

        def show() -> None:
            size = view.console.size
            view.update(render(conv, status(), size.width, size.height))
        conv.on_change = show
    else:
        conv.on_change = PlainPrinter(conv)

    def say(phase: str) -> None:
        state["phase"] = phase
        conv.on_change() if view is not None else print(f"— {phase}", flush=True)

    runner = Runner(conv)
    seg: TurnSegmenter | None = None
    last_peek, peeking = [0.0], view is not None and (not args.file or args.realtime)

    def feed(turns: list[Turn]) -> None:
        for turn in turns:
            runner.turns.put(turn)
        if turns:
            conv.speaking = None
        elif peeking and seg.speaking >= 1.0 and seg.n * STEP - last_peek[0] >= 0.5:
            last_peek[0] = seg.n * STEP                         # live: whose voice is this, before the turn ends
            try:
                conv.peek(seg.recent_audio())
            except Exception as err:
                conv.problem = f"{type(err).__name__}: {err}"

    try:
        if view is not None:
            view.start()
        if args.file:
            raw, sr = sf.read(str(args.file), dtype="float32", always_2d=False)
            raw = raw.mean(axis=1) if raw.ndim > 1 else raw
            seg, block, t0 = TurnSegmenter(sr, gap_s=args.gap), int(0.1 * sr), time.monotonic()
            say(f"{'playing' if args.realtime else 'reading'} {args.file.name}")
            for k, s in enumerate(range(0, len(raw), block)):
                if args.realtime:
                    time.sleep(max(0.0, t0 + k * 0.1 - time.monotonic()))
                feed(seg.push(raw[s: s + block]))
            feed(seg.flush())
            say("finishing")
            runner.finish()
        else:
            seg, blocks = TurnSegmenter(mic_sr, gap_s=args.gap), queue.Queue()
            with sd.InputStream(device=mic, channels=1, samplerate=mic_sr, dtype="float32",
                                blocksize=int(mic_sr * 0.1), callback=lambda indata, *_: blocks.put(indata[:, 0].copy())):
                say("listening (Ctrl+C to stop)")
                while True:
                    try:
                        feed(seg.push(blocks.get(timeout=0.5)))
                    except queue.Empty:
                        pass
    except KeyboardInterrupt:
        state["phase"] = "stopping"
        if seg is not None and not args.file:
            for turn in seg.flush():                            # what was being said when Ctrl+C was pressed
                runner.turns.put(turn)
        runner.finish(skip_pending=bool(args.file))
    finally:
        state["phase"], conv.speaking = "done", None
        if view is not None:
            show()
            view.stop()

    text = transcript(conv)
    print("\n" + (text if text else "Nobody spoke."))
    if args.save:
        args.save.write_text(text + "\n", encoding="utf-8")
        print(f"\n✓ Transcript saved to {args.save}")
    if args.remember:
        remember(conv, bank, emb.model_id)


def remember(conv: Conversation, bank: live.SpeakerBank, model_id: str, min_seconds: float = 8.0) -> None:
    """Add the voices that got a name in this conversation to the phase-3 enrollment bank."""
    if bank.speakers and bank.data["model"] != model_id:
        print(f"Not saving voices: {bank.path} was made with another model.")
        return
    saved, short = [], []
    for spk, n in conv.names.items():
        c = conv.voices.clusters.get(spk)
        lines = [ln for ln in conv.lines if ln.speaker == spk and ln.raw is not None]
        if c is None or c["fixed"] or not lines:
            continue
        seconds = sum(ln.seconds for ln in lines)
        if seconds < min_seconds:
            short.append(n.name)
            continue
        bank.add(n.name, np.stack([ln.raw for ln in lines]), seconds, model_id)   # as phase 3 expects them
        saved.append(n.name)
    if saved:
        bank.save()
        print(f"✓ Remembered {', '.join(saved)} in {bank.path}: next time they are recognised by voice.")
    if short:
        print(f"Not remembered (less than {min_seconds:.0f} s of speech): {', '.join(short)}.")
    if not saved and not short:
        print("No voice to remember: nobody got a name.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m spectre.conversation",
                                description="Live conversation captions with names worked out from what is said")
    p.add_argument("--file", type=Path, help="process a recording instead of the microphone (WAV, FLAC, MP3, OGG)")
    p.add_argument("--realtime", action="store_true", help="with --file: go at real speed, as if it were live")
    p.add_argument("--device", help="microphone: number or part of the name (see 'python -m spectre.live devices')")
    p.add_argument("--llm", default="ollama:gemma3:12b",
                   help="who reads the lines: ollama:<model>, hf:<model>, rules (no LLM) or none (default: %(default)s)")
    p.add_argument("--whisper", default="openai/whisper-large-v3-turbo", help="Whisper model (default: %(default)s)")
    p.add_argument("--language", default="pt", help="spoken language, or 'auto' (default: %(default)s)")
    p.add_argument("--ckpt", type=Path, help="SPECTRE ECAPA checkpoint (default: the latest phase-2 run)")
    p.add_argument("--bank", type=Path, default=live.DEFAULT_BANK, help="enrolled people (default: %(default)s)")
    p.add_argument("--cluster-threshold", type=float,
                   help="voice similarity to count as the same speaker (default: the calibration of spectre.conv_eval, "
                        "or 0.65). Raise it if two people share a label, lower it if one person shows up as two")
    p.add_argument("--merge-threshold", type=float, help="similarity to merge two speakers (default: threshold + 0.2)")
    p.add_argument("--gap", type=float, default=0.6, help="pause that ends a turn, in seconds (default: %(default)s)")
    p.add_argument("--remember", action="store_true", help="save the voices that got a name, to recognise them next time")
    p.add_argument("--save", type=Path, help="write the final transcript to this file")
    p.add_argument("--cpu", action="store_true", help="run SPECTRE and Whisper on the CPU")
    p.add_argument("--plain", action="store_true", help="plain text output instead of the live panel")
    return p


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
