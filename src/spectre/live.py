"""SPECTRE live demo — enroll people with a few seconds of speech, then recognise who is talking.

    python -m spectre.live devices                          # list microphones
    python -m spectre.live enroll "Ana"                     # record 20 s from the microphone
    python -m spectre.live enroll "Rui" --file rui.wav      # ...or enroll from audio files
    python -m spectre.live list
    python -m spectre.live remove "Rui"
    python -m spectre.live calibrate                        # suggest a threshold from the enrolled people
    python -m spectre.live identify                         # live: who is speaking right now?
    python -m spectre.live identify --file conversa.wav     # the same pipeline over a recording

How it works: every 0.5 s an energy-based voice activity detector looks at the last 3 s of audio
and keeps the current turn — the speech after the last pause of at least 0.6 s, where speakers
usually change. With at least 1 s of speech, its 192-d ECAPA embedding is compared (cosine) with
each enrolled person; scores are smoothed within the turn and the best match is shown if it is
above the threshold, otherwise the speaker is reported as unknown.

Only voice embeddings are stored (enrollments/speakers.json), never the audio. They are
biometric data: enrollments/ is git-ignored, keep it private.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml

from .model import build_model

SR = 16000
FRAME, HOP = 400, 160                    # 25 ms frames every 10 ms for the voice activity detector
DEFAULT_BANK = Path("enrollments/speakers.json")


# --------------------------------------------------------------------------- audio helpers
def resample(wav: np.ndarray, sr: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr == SR:
        return wav
    import torchaudio.functional as AF
    return AF.resample(torch.from_numpy(np.ascontiguousarray(wav)), sr, SR).numpy()


def frame_db(wav: np.ndarray) -> np.ndarray:
    """RMS level (dBFS) of 25 ms frames every 10 ms."""
    if len(wav) < FRAME:
        return np.zeros(0)
    frames = np.lib.stride_tricks.sliding_window_view(wav, FRAME)[::HOP]
    return 10 * np.log10((frames.astype(np.float64) ** 2).mean(axis=1) + 1e-12)


def noise_floor(wav: np.ndarray) -> float:
    """Background level: speech always has pauses, so the 10th percentile of frame levels is noise."""
    db = frame_db(wav)
    return float(np.percentile(db, 10)) if db.size else -120.0


def voiced(wav: np.ndarray, floor_db: float, margin_db: float = 10.0, min_db: float = -55.0) -> np.ndarray:
    """Keep only the audio louder than the noise floor + margin (with ~100 ms of context around it)."""
    db = frame_db(wav)
    if db.size == 0:
        return wav[:0]
    active = db > max(floor_db + margin_db, min_db)
    active = np.convolve(active.astype(float), np.ones(21), mode="same") > 0
    keep = np.zeros(len(wav), dtype=bool)
    for i in np.flatnonzero(active):
        keep[i * HOP: i * HOP + FRAME] = True
    return wav[keep]


def current_turn(wav: np.ndarray, floor_db: float, margin_db: float = 10.0, gap_s: float = 0.6,
                 min_db: float = -55.0) -> np.ndarray:
    """Voiced audio after the last pause of at least `gap_s` — the current speaker's turn.
    Empty if the window ends in such a pause (nobody is speaking now)."""
    db = frame_db(wav)
    if db.size == 0:
        return wav[:0]
    active = db > max(floor_db + margin_db, min_db)
    gap, run, start = int(gap_s * SR / HOP), 0, 0
    for i, a in enumerate(active):
        run = 0 if a else run + 1
        if run >= gap:
            start = i + 1
    active[:start] = False
    active = np.convolve(active.astype(float), np.ones(21), mode="same") > 0
    keep = np.zeros(len(wav), dtype=bool)
    for i in np.flatnonzero(active):
        keep[i * HOP: i * HOP + FRAME] = True
    return wav[keep]


def read_audio(paths: list[Path]) -> tuple[np.ndarray, int]:
    """Read one or more audio files (any rate, mono or stereo) and join them at 16 kHz."""
    parts = []
    for p in paths:
        wav, sr = sf.read(str(p), dtype="float32", always_2d=False)
        parts.append(resample(wav, sr))
    return np.concatenate(parts), SR


# --------------------------------------------------------------------------- model
def find_checkpoint() -> Path:
    """Most recent phase-2 (ECAPA) run that has a best.pt."""
    runs = []
    for best in Path("runs").glob("*/best.pt"):
        cfg_file = best.parent / "config.yaml"
        if cfg_file.exists() and yaml.safe_load(cfg_file.read_text()).get("model", {}).get("type") == "ecapa":
            runs.append(best)
    if not runs:
        sys.exit("No trained ECAPA model found in runs/. Train one (configs/ecapa.yaml) or pass --ckpt.")
    return max(runs, key=lambda p: p.parent.name.rsplit("_", 1)[-1])  # run folders end in a timestamp


class Embedder:
    def __init__(self, ckpt: Path, device: str | None = None):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        if cfg["model"].get("type") != "ecapa":
            sys.exit(f"{ckpt} is not a speaker-embedding (ECAPA) checkpoint.")
        cfg["model"]["pretrained"] = False
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(cfg, len(ck["label_map"])).to(self.device).eval()
        self.model.load_state_dict(ck["model"])
        self.model_id = f"{ckpt.parent.name}/{ckpt.name}"
        self.ckpt = ckpt

    @torch.no_grad()
    def __call__(self, wavs: list[np.ndarray] | np.ndarray, batch: int = 32) -> np.ndarray:
        """L2-normalised embeddings, one row per waveform. Waveforms of equal length are batched
        (features are normalised per utterance and BatchNorm is frozen, so results are identical)."""
        if isinstance(wavs, np.ndarray):
            wavs = [wavs]
        out: list[np.ndarray | None] = [None] * len(wavs)
        by_len: dict[int, list[int]] = {}
        for i, w in enumerate(wavs):
            by_len.setdefault(len(w), []).append(i)
        for idx in by_len.values():
            for k in range(0, len(idx), batch):
                part = idx[k: k + batch]
                x = torch.from_numpy(np.stack([np.asarray(wavs[i], dtype=np.float32) for i in part])).to(self.device)
                e = F.normalize(self.model.embed(x).float(), dim=-1).cpu().numpy()
                for j, i in enumerate(part):
                    out[i] = e[j]
        return np.stack(out)

    def chunks(self, wav: np.ndarray, win_s: float = 3.0, hop_s: float = 1.5) -> np.ndarray:
        """Embeddings of overlapping 3 s chunks — the same kind of window used for live decisions."""
        n, h = int(win_s * SR), int(hop_s * SR)
        if len(wav) <= n:
            return self(wav)
        return self([wav[s: s + n] for s in range(0, len(wav) - n + 1, h)])

    def default_threshold(self) -> float | None:
        res = self.ckpt.parent / "embed_results.json"
        if res.exists():
            return json.loads(res.read_text()).get("unseen_verification", {}).get("eer_threshold")
        return None


# --------------------------------------------------------------------------- enrolled people
class SpeakerBank:
    """Enrolled people: per person, the embeddings of their 3 s enrollment chunks."""

    def __init__(self, path: Path = DEFAULT_BANK):
        self.path = path
        self.data = {"model": None, "threshold": None, "speakers": {}}
        if path.exists():
            self.data.update(json.loads(path.read_text(encoding="utf-8")))

    @property
    def speakers(self) -> dict:
        return self.data["speakers"]

    def check_model(self, model_id: str) -> None:
        if self.speakers and self.data["model"] != model_id:
            sys.exit(f"The enrolled people were registered with another model ({self.data['model']}).\n"
                     f"Embeddings from different models are not comparable: delete {self.path} and enroll "
                     f"everyone again, or pass --ckpt runs/{self.data['model']}.")

    def add(self, name: str, chunks: np.ndarray, seconds: float, model_id: str, replace: bool = False) -> None:
        old = self.speakers.get(name)
        keep = [] if (replace or old is None) else old["chunks"]
        self.speakers[name] = {"chunks": keep + np.round(chunks, 6).tolist(),
                               "seconds": round(seconds + (0 if replace or old is None else old["seconds"]), 1),
                               "updated": datetime.now().isoformat(timespec="seconds")}
        self.data["model"] = model_id

    def centroids(self) -> tuple[list[str], np.ndarray]:
        names = sorted(self.speakers)
        C = np.stack([np.asarray(self.speakers[n]["chunks"]).mean(axis=0) for n in names])
        return names, C / np.linalg.norm(C, axis=1, keepdims=True)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- live decisions
@dataclass
class Decision:
    t: float                    # seconds since the start
    name: str | None            # recognised person, None = unknown
    score: float | None         # smoothed cosine score of the best match (None = silence)
    best: str | None            # closest enrolled person, even if below the threshold


class LiveIdentifier:
    """Turns a stream of audio blocks (any sample rate) into a decision every `hop_s` seconds."""

    def __init__(self, embedder: Embedder, names: list[str], centroids: np.ndarray, threshold: float,
                 sr_in: int, win_s: float = 3.0, hop_s: float = 0.5, min_speech_s: float = 1.0,
                 smooth: float = 0.5, vad_margin_db: float = 10.0, gap_s: float = 0.6):
        self.embedder, self.names, self.C, self.threshold = embedder, names, centroids, threshold
        self.sr_in, self.win_s, self.hop_s, self.min_speech_s = sr_in, win_s, hop_s, min_speech_s
        self.smooth, self.vad_margin_db, self.gap_s = smooth, vad_margin_db, gap_s
        self.raw = np.zeros(0, dtype=np.float32)      # last 10 s at the input rate (noise floor + window)
        self.history = int(10 * sr_in)
        self.since_last = 0
        self.t = 0.0
        self.ema: np.ndarray | None = None
        self.talk_time: dict[str, float] = {}

    def push(self, block: np.ndarray) -> list[Decision]:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        self.raw = np.concatenate([self.raw, block])[-self.history:]
        self.since_last += len(block)
        decisions = []
        hop = int(self.hop_s * self.sr_in)
        while self.since_last >= hop:
            self.since_last -= hop
            self.t += self.hop_s
            decisions.append(self._decide())
        return decisions

    def _decide(self) -> Decision:
        audio = resample(self.raw, self.sr_in)
        window = audio[-int(self.win_s * SR):]
        speech = current_turn(window, noise_floor(audio), self.vad_margin_db, self.gap_s)
        if len(speech) < self.min_speech_s * SR:
            self.ema = None                            # a pause: the next voice may be someone else
            return Decision(self.t, None, None, None)
        scores = self.C @ self.embedder(speech)[0]
        self.ema = scores if self.ema is None else (1 - self.smooth) * self.ema + self.smooth * scores
        i = int(np.argmax(self.ema))
        name = self.names[i] if self.ema[i] >= self.threshold else None
        key = name or "(unknown)"
        self.talk_time[key] = self.talk_time.get(key, 0.0) + self.hop_s
        return Decision(self.t, name, float(self.ema[i]), self.names[i])


def bar(score: float, width: int = 20) -> str:
    n = int(round(max(0.0, min(1.0, score)) * width))
    return "█" * n + "░" * (width - n)


def show_live(d: Decision, threshold: float) -> None:
    if d.score is None:
        line = "  …  listening (silence)"
    elif d.name:
        line = f"  ▶  {d.name:<16} {bar(d.score)}  {d.score:5.2f}"
    else:
        line = f"  ?  {'unknown':<16} {bar(d.score)}  {d.score:5.2f}   closest: {d.best} (needs ≥ {threshold:.2f})"
    print("\r" + line.ljust(100), end="", flush=True)


def print_talk_time(talk_time: dict[str, float]) -> None:
    total = sum(talk_time.values())
    if not total:
        print("No speech detected.")
        return
    print("Speaking time:")
    for name, s in sorted(talk_time.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<16} {s:6.1f} s  ({100 * s / total:4.1f} %)")


# --------------------------------------------------------------------------- microphone
def _sounddevice():
    try:
        import sounddevice as sd
    except (ImportError, OSError) as err:
        sys.exit(f"Microphone support needs the 'sounddevice' package ({err}).\n"
                 "Install it with:  pip install sounddevice   (on Linux also: sudo apt install libportaudio2)")
    return sd


def _device(device: str | None):
    return int(device) if device is not None and device.isdigit() else device


NO_MIC = ("No microphone found. Plug one in (a headset or webcam microphone works) and check that apps may use it "
          "(Windows: Settings → Privacy & security → Microphone).")


def microphones(sd) -> list[str]:
    """Input devices as printable lines: index, name and audio API."""
    apis = sd.query_hostapis()
    return [f"  {i:>3}  {d['name']}  ({apis[d['hostapi']]['name']})"
            for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


def input_device(sd, device: str | None) -> tuple[int | str | None, int]:
    """The microphone to use and its native sample rate — or a clear explanation when there is none."""
    dev = _device(device)
    try:
        return dev, int(sd.query_devices(dev, "input")["default_samplerate"])
    except (getattr(sd, "PortAudioError", RuntimeError), ValueError):
        pass
    mics = microphones(sd)
    if not mics:
        sys.exit(NO_MIC)
    problem = f"'{device}' is not an available microphone." if device is not None else "There is no default microphone."
    sys.exit(f"{problem} Choose one with --device NUMBER (or set a default microphone in the system sound "
             f"settings):\n" + "\n".join(mics))


def record(seconds: int, device: str | None) -> tuple[np.ndarray, int]:
    sd = _sounddevice()
    dev, sr = input_device(sd, device)
    print(f"Speak naturally for {seconds} s (reading a text aloud works well). Recording...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32", device=dev)
    for i in range(seconds):
        time.sleep(1)
        print(f"\r  {bar((i + 1) / seconds, 30)}  {i + 1:>2}/{seconds} s", end="", flush=True)
    sd.wait()
    print()
    return audio[:, 0], sr


def listen(identifier: LiveIdentifier, device: int | str | None, show) -> None:
    """Feed the microphone (an already resolved device, see input_device) to the identifier."""
    sd = _sounddevice()
    blocks: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        blocks.put(indata[:, 0].copy())

    with sd.InputStream(device=device, channels=1, samplerate=identifier.sr_in, dtype="float32",
                        blocksize=int(identifier.sr_in * 0.1), callback=callback):
        while True:
            for d in identifier.push(blocks.get()):
                show(d)


# --------------------------------------------------------------------------- commands
def cmd_devices(a) -> None:
    sd = _sounddevice()
    mics = microphones(sd)
    if not mics:
        sys.exit(NO_MIC)
    try:
        default = sd.query_devices(None, "input")["name"]
    except (getattr(sd, "PortAudioError", RuntimeError), ValueError):
        default = None
    print("Microphones:\n" + "\n".join(mics))
    print(f"\nDefault: {default or 'none'} — use --device NUMBER (or part of the name) to choose another.")


def cmd_enroll(a) -> None:
    emb = Embedder(a.ckpt or find_checkpoint(), "cpu" if a.cpu else None)
    bank = SpeakerBank(a.bank)
    bank.check_model(emb.model_id)
    if a.file:
        wav, _ = read_audio(a.file)
    else:
        raw, sr_in = record(a.seconds, a.device)
        wav = resample(raw, sr_in)
    if len(wav) == 0 or float(np.abs(wav).max()) < 1e-6:
        sys.exit("The microphone returned pure silence. Check the input device ('devices', --device) and that "
                 "apps may use the microphone (Windows: Settings → Privacy & security → Microphone).")
    peak_share = float((np.abs(wav) > 0.99).mean())
    speech = voiced(wav, noise_floor(wav), a.vad_margin)
    secs = len(speech) / SR
    level = 10 * np.log10(float((speech.astype(np.float64) ** 2).mean()) + 1e-12) if len(speech) else -120.0
    print(f"Heard {secs:.1f} s of speech (level {level:.0f} dBFS).")
    if peak_share > 0.001:
        print("⚠ The audio is clipping — move a bit away from the microphone or lower its gain.")
    if secs < a.min_speech:
        sys.exit(f"Not enough speech: need at least {a.min_speech:.0f} s. Speak closer to the microphone "
                 f"or for longer (--seconds), then try again.")
    chunks = emb.chunks(speech)
    consistency = float((chunks @ (chunks.mean(0) / np.linalg.norm(chunks.mean(0)))).mean())
    bank.add(a.name, chunks, secs, emb.model_id, replace=a.replace)
    bank.save()
    print(f"✓ Enrolled {a.name}: {len(chunks)} chunks · self-consistency {consistency:.2f} · "
          f"{len(bank.speakers)} people in {bank.path}")


def cmd_list(a) -> None:
    bank = SpeakerBank(a.bank)
    if not bank.speakers:
        print("Nobody enrolled yet. Use: python -m spectre.live enroll <name>")
        return
    print(f"Model: {bank.data['model']}" + (f" · calibrated threshold {bank.data['threshold']:.2f}"
                                             if bank.data.get("threshold") is not None else ""))
    for name, s in sorted(bank.speakers.items()):
        print(f"  {name:<20} {s['seconds']:6.1f} s of speech · {len(s['chunks']):3d} chunks · {s['updated']}")


def cmd_remove(a) -> None:
    bank = SpeakerBank(a.bank)
    if a.name not in bank.speakers:
        sys.exit(f"'{a.name}' is not enrolled. Enrolled: {', '.join(sorted(bank.speakers)) or 'nobody'}")
    del bank.speakers[a.name]
    bank.save()
    print(f"✓ Removed {a.name}.")


def cmd_calibrate(a) -> None:
    """Threshold from how similar *different* enrolled people are to each other.

    Same-person scores from the enrollment recordings are inflated (same session, overlapping
    chunks), so they are only reported. The threshold is the 95th percentile of each chunk's score
    against the closest *other* enrolled person — a voice must be closer to someone than 95 % of
    cross-person matches — and never below the model's own EER threshold."""
    bank = SpeakerBank(a.bank)
    names = sorted(bank.speakers)
    if len(names) < 2:
        sys.exit("Enroll at least two people before calibrating.")
    _, C = bank.centroids()
    same, closest = [], []
    for i, n in enumerate(names):
        X = np.asarray(bank.speakers[n]["chunks"])
        for k in range(len(X)):
            far = [j for j in range(len(X)) if abs(j - k) > 1] or [j for j in range(len(X)) if j != k]
            if far:                                 # leave-one-out, without the overlapping neighbours
                own = X[far].mean(0)
                same.append(float(X[k] @ (own / np.linalg.norm(own))))
        closest.extend((X @ np.delete(C, i, axis=0).T).max(axis=1).tolist())
    same, closest = np.array(same), np.array(closest)
    p95, p10 = float(np.percentile(closest, 95)), float(np.percentile(same, 10)) if len(same) else 1.0
    res = Path("runs") / str(bank.data["model"])
    floor = None
    if (res.parent / "embed_results.json").exists():
        floor = json.loads((res.parent / "embed_results.json").read_text()).get("unseen_verification", {}).get("eer_threshold")
    thr = max(p95, floor or 0.0)
    print(f"Same person (enrollment session) : {same.mean():.2f} ± {same.std():.2f}   (optimistic: same session)")
    print(f"Closest other enrolled person    : {closest.mean():.2f} ± {closest.std():.2f} · 95th percentile {p95:.2f}")
    if floor is not None and floor > p95:
        print(f"The model's own EER threshold ({floor:.2f}) is higher, so it is used instead.")
    if thr >= p10:
        thr = (p95 + p10) / 2
        print("⚠ Some enrolled voices are hard to tell apart; enrolling more speech (or in a quieter place) helps.")
    bank.data["threshold"] = thr
    bank.save()
    print(f"✓ Threshold {thr:.2f} saved; 'identify' will use it. Raise it (--threshold) if strangers get named,\n"
          f"  lower it if enrolled people often show as unknown.")


def cmd_identify(a) -> None:
    emb = Embedder(a.ckpt or find_checkpoint(), "cpu" if a.cpu else None)
    bank = SpeakerBank(a.bank)
    if not bank.speakers:
        sys.exit("Nobody enrolled yet. Use: python -m spectre.live enroll <name>")
    bank.check_model(emb.model_id)
    names, C = bank.centroids()
    threshold = next(t for t in (a.threshold, bank.data.get("threshold"), emb.default_threshold(), 0.3)
                     if t is not None)
    print(f"Model {emb.model_id} on {emb.device.type} · {len(names)} people · threshold {threshold:.2f}")
    if a.file:
        wav, _ = read_audio(a.file)
        ident = LiveIdentifier(emb, names, C, threshold, SR, hop_s=a.hop, smooth=a.smooth,
                               vad_margin_db=a.vad_margin)
        last = "start"
        for s in range(0, len(wav), int(0.1 * SR)):         # feed it like a microphone, in 100 ms blocks
            for d in ident.push(wav[s: s + int(0.1 * SR)]):
                label = "(silence)" if d.score is None else (d.name or f"unknown (closest {d.best})")
                if label != last:
                    score = "" if d.score is None else f"  {d.score:.2f}"
                    print(f"  {int(d.t // 60):02d}:{d.t % 60:04.1f}  {label}{score}")
                    last = label
        print_talk_time(ident.talk_time)
        return
    sd = _sounddevice()
    dev, sr_in = input_device(sd, a.device)
    ident = LiveIdentifier(emb, names, C, threshold, sr_in, hop_s=a.hop, smooth=a.smooth,
                           vad_margin_db=a.vad_margin)
    print("Listening... speak! (Ctrl+C to stop)\n")
    try:
        listen(ident, dev, lambda d: show_live(d, threshold))
    except KeyboardInterrupt:
        print("\n")
        print_talk_time(ident.talk_time)


def _common(parser: argparse.ArgumentParser, suppress: bool) -> argparse.ArgumentParser:
    """Options valid before or after the command ('--device 2 enroll Ana' or 'enroll Ana --device 2')."""
    d = (lambda v: argparse.SUPPRESS) if suppress else (lambda v: v)
    parser.add_argument("--bank", type=Path, default=d(DEFAULT_BANK), help="enrolled people (default: %s)" % DEFAULT_BANK)
    parser.add_argument("--ckpt", type=Path, default=d(None), help="ECAPA checkpoint (default: the latest phase-2 run)")
    parser.add_argument("--device", default=d(None), help="microphone: number or part of the name (see 'devices')")
    parser.add_argument("--cpu", action="store_true", default=d(False), help="run the model on the CPU")
    parser.add_argument("--vad-margin", type=float, default=d(10.0), help="dB above the noise floor that counts as speech")
    return parser


def build_parser() -> argparse.ArgumentParser:
    p = _common(argparse.ArgumentParser(prog="python -m spectre.live",
                                        description="SPECTRE live speaker recognition"), suppress=False)
    sub = p.add_subparsers(dest="cmd", required=True)
    add = lambda name, **kw: _common(sub.add_parser(name, **kw), suppress=True)
    add("devices", help="list microphones").set_defaults(func=cmd_devices)
    e = add("enroll", help="register a person from the microphone or from audio files")
    e.add_argument("name")
    e.add_argument("--file", type=Path, nargs="+", help="audio file(s) instead of the microphone")
    e.add_argument("--seconds", type=int, default=20, help="recording length (default: %(default)s)")
    e.add_argument("--min-speech", type=float, default=8.0, help="minimum seconds of speech (default: %(default)s)")
    e.add_argument("--replace", action="store_true", help="replace this person's previous enrollment")
    e.set_defaults(func=cmd_enroll)
    add("list", help="show enrolled people").set_defaults(func=cmd_list)
    r = add("remove", help="delete a person")
    r.add_argument("name")
    r.set_defaults(func=cmd_remove)
    add("calibrate", help="suggest a threshold from the enrolled people").set_defaults(func=cmd_calibrate)
    i = add("identify", help="recognise who is speaking (live, or over a file)")
    i.add_argument("--file", type=Path, nargs="+", help="audio file(s) instead of the microphone")
    i.add_argument("--threshold", type=float, help="minimum score to name someone (default: calibrated value)")
    i.add_argument("--hop", type=float, default=0.5, help="seconds between decisions (default: %(default)s)")
    i.add_argument("--smooth", type=float, default=0.5, help="weight of the newest score, 0-1 (default: %(default)s)")
    i.set_defaults(func=cmd_identify)
    return p


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
