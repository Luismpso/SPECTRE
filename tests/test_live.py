"""Fast tests for the live demo (no dataset, no GPU, no microphone needed).

    pip install pytest && pytest -q
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from spectre import live
from spectre.model import build_model

SR = live.SR
RNG = np.random.default_rng(0)


def tone(freq: float, seconds: float, sr: int = SR, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def quiet(seconds: float, sr: int = SR) -> np.ndarray:
    return (1e-4 * RNG.standard_normal(int(seconds * sr))).astype(np.float32)


class ToneEmbedder:
    """Fake embedder: 200 Hz → person A, 300 Hz → person B, anything else → nobody."""

    def __call__(self, wav):
        spectrum = np.abs(np.fft.rfft(wav))
        f = np.fft.rfftfreq(len(wav), 1 / SR)[np.argmax(spectrum)]
        e = np.array([abs(f - 200) < 20, abs(f - 300) < 20, abs(f - 200) >= 20 and abs(f - 300) >= 20], float)
        return e[None, :]


def make_identifier(sr_in: int = SR) -> live.LiveIdentifier:
    C = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return live.LiveIdentifier(ToneEmbedder(), ["A", "B"], C, threshold=0.5, sr_in=sr_in)


def run(ident: live.LiveIdentifier, wav: np.ndarray, sr: int) -> list[live.Decision]:
    out = []
    for s in range(0, len(wav), int(0.1 * sr)):
        out += ident.push(wav[s: s + int(0.1 * sr)])
    return out


# --------------------------------------------------------------------------- voice activity
def test_voiced_keeps_only_speech():
    wav = np.concatenate([quiet(2), tone(200, 1.5), quiet(2)])
    speech = live.voiced(wav, live.noise_floor(wav))
    assert 1.5 * SR <= len(speech) <= 1.8 * SR          # the tone plus ~100 ms of context on each side


def test_voiced_ignores_silence_and_muted_input():
    assert len(live.voiced(quiet(3), live.noise_floor(quiet(3)))) == 0
    zeros = np.zeros(3 * SR, np.float32)
    assert len(live.voiced(zeros, live.noise_floor(zeros))) == 0


def test_current_turn_starts_after_the_last_pause():
    a, b = tone(200, 1.5), tone(300, 1.2)
    window = np.concatenate([a, quiet(0.8), b])           # A, a pause, then B speaking now
    turn = live.current_turn(window, live.noise_floor(np.concatenate([quiet(5), window])))
    assert 1.2 * SR <= len(turn) <= 1.45 * SR             # only B's speech (plus a little context)
    assert len(live.current_turn(np.concatenate([a, quiet(0.8)]), -80.0)) == 0   # ends in a pause


def test_resample_48k():
    assert abs(len(live.resample(tone(200, 2, sr=48000), 48000)) - 2 * SR) <= 1


# --------------------------------------------------------------------------- live decisions
def test_identifier_names_unknown_and_silence():
    sr = 48000                                            # a typical microphone rate
    wav = np.concatenate([quiet(3, sr), tone(200, 4, sr), quiet(2, sr), tone(300, 4, sr), tone(500, 4, sr)])
    ds = run(make_identifier(sr), wav, sr)
    at = lambda t: next(d for d in ds if abs(d.t - t) < 1e-6)
    assert at(2.5).score is None                          # silence
    assert at(6.5).name == "A"
    assert at(8.5).score is None                          # pause between speakers
    assert at(11.5).name == "B"                           # the pause reset the smoothing: no carry-over from A
    assert at(16.5).name is None and at(16.5).best is not None   # unknown voice: below the threshold
    assert len(ds) == int(len(wav) / sr / 0.5)            # one decision every 0.5 s


def test_identifier_talk_time():
    ident = make_identifier()
    run(ident, np.concatenate([quiet(3), tone(200, 5), quiet(3), tone(300, 3)]), SR)
    assert ident.talk_time["A"] > ident.talk_time["B"] > 0


# --------------------------------------------------------------------------- enrolled people
def test_bank_roundtrip_and_model_check(tmp_path):
    bank = live.SpeakerBank(tmp_path / "speakers.json")
    bank.add("Luís", np.eye(3)[[0, 0]], 12.0, "run_a/best.pt")
    bank.add("João", np.eye(3)[[1]], 9.0, "run_a/best.pt")
    bank.add("Luís", np.eye(3)[[0]], 6.0, "run_a/best.pt")          # a second session adds chunks
    bank.save()
    again = live.SpeakerBank(tmp_path / "speakers.json")
    assert sorted(again.speakers) == ["João", "Luís"]
    assert len(again.speakers["Luís"]["chunks"]) == 3 and again.speakers["Luís"]["seconds"] == 18.0
    names, C = again.centroids()
    assert names == ["João", "Luís"] and np.allclose(np.linalg.norm(C, axis=1), 1)
    again.check_model("run_a/best.pt")
    with pytest.raises(SystemExit):
        again.check_model("run_b/best.pt")                # embeddings of another model are not comparable


# --------------------------------------------------------------------------- real (tiny) model
@pytest.fixture
def tiny_ckpt(tmp_path):
    cfg = yaml.safe_load(open("configs/ecapa.yaml"))
    cfg["model"].update(channels=64, emb_dim=32)
    torch.manual_seed(0)
    model = build_model(cfg, 4)
    path = tmp_path / "run_20260101-000000" / "best.pt"
    path.parent.mkdir()
    torch.save({"model": model.state_dict(), "cfg": cfg, "label_map": {str(i): i for i in range(4)}}, path)
    return path


def test_embedder_batching_matches_one_by_one(tiny_ckpt):
    emb = live.Embedder(tiny_ckpt, "cpu")
    wavs = [RNG.standard_normal(SR * 3).astype(np.float32) * 0.1 for _ in range(5)]
    batched = emb(wavs)
    single = np.stack([emb(w)[0] for w in wavs])
    assert batched.shape == (5, 32) and np.allclose(batched, single, atol=1e-5)
    assert np.allclose(np.linalg.norm(batched, axis=1), 1, atol=1e-5)
    assert emb.chunks(np.concatenate(wavs)).shape[0] == 9   # 15 s in 3 s chunks every 1.5 s


# --------------------------------------------------------------------------- microphone (simulated)
class FakeSoundDevice:
    """Stands in for the sounddevice package: plays a waveform through the input callback."""

    class PortAudioError(Exception):
        pass

    def __init__(self, wav=None, sr=SR, mics=("USB Microphone",), has_default=True):
        self.wav, self.sr, self.has_default = wav, sr, has_default
        self.devices = [{"name": "Speakers", "max_input_channels": 0, "hostapi": 0}] + \
                       [{"name": m, "max_input_channels": 1, "hostapi": 0} for m in mics]

    def query_hostapis(self):
        return ({"name": "MME"},)

    def query_devices(self, device=None, kind=None):
        if device is None and kind is None:
            return self.devices
        if device is None:
            if not self.has_default:
                raise self.PortAudioError("Error querying device -1")   # what Windows does without a default mic
            return {"name": self.devices[1]["name"], "default_samplerate": float(self.sr)}
        if not (isinstance(device, int) and device < len(self.devices) and self.devices[device]["max_input_channels"]):
            raise ValueError(f"No input device matching {device!r}")
        return {"name": self.devices[device]["name"], "default_samplerate": float(self.sr)}

    def rec(self, frames, samplerate, channels, dtype, device=None):
        return self.wav[:frames, None].copy()

    def wait(self):
        pass

    def InputStream(self, callback, blocksize, **kw):
        fake = self

        class Stream:
            def __enter__(self):
                for s in range(0, len(fake.wav), blocksize):
                    callback(fake.wav[s: s + blocksize, None], blocksize, None, None)
                return self

            def __exit__(self, *exc):
                return False

        return Stream()


def test_listen_and_record_with_fake_microphone(monkeypatch):
    sr = 44100
    wav = np.concatenate([quiet(3, sr), tone(300, 4, sr)])
    fake = FakeSoundDevice(wav, sr)
    monkeypatch.setattr(live, "_sounddevice", lambda: fake)
    monkeypatch.setattr(live.time, "sleep", lambda s: None)

    recorded, rate = live.record(2, None)
    assert rate == sr and len(recorded) == 2 * sr

    ident, seen = make_identifier(sr), []

    def show(d):
        seen.append(d)
        if d.t >= 6.5:
            raise KeyboardInterrupt                       # what Ctrl+C does in the real loop

    with pytest.raises(KeyboardInterrupt):
        live.listen(ident, None, show)
    assert seen[-1].name == "B"


def test_no_default_microphone_explains_how_to_choose(monkeypatch):
    fake = FakeSoundDevice(tone(300, 3), mics=("Headset Microphone", "Webcam"), has_default=False)
    monkeypatch.setattr(live, "_sounddevice", lambda: fake)
    with pytest.raises(SystemExit) as err:
        live.record(2, None)
    msg = str(err.value)
    assert "no default microphone" in msg and "--device" in msg and "Headset Microphone" in msg
    assert live.input_device(fake, "2") == (2, SR)                    # choosing one by number works
    with pytest.raises(SystemExit) as err:
        live.input_device(fake, "0")                                  # speakers are not a microphone
    assert "not an available microphone" in str(err.value)


def test_no_microphone_at_all(monkeypatch, capsys):
    fake = FakeSoundDevice(mics=(), has_default=False)
    monkeypatch.setattr(live, "_sounddevice", lambda: fake)
    with pytest.raises(SystemExit) as err:
        live.record(2, None)
    assert "No microphone found" in str(err.value)
    with pytest.raises(SystemExit):
        live.main(["devices"])


def test_devices_lists_only_microphones(monkeypatch, capsys):
    monkeypatch.setattr(live, "_sounddevice", lambda: FakeSoundDevice(mics=("USB Microphone",)))
    live.main(["devices"])
    out = capsys.readouterr().out
    assert "USB Microphone" in out and "Speakers" not in out and "Default: USB Microphone" in out


def test_options_before_or_after_the_command():
    p = live.build_parser()
    a = p.parse_args(["enroll", "Luís", "--device", "3", "--cpu"])
    assert (a.name, a.device, a.cpu, a.bank) == ("Luís", "3", True, live.DEFAULT_BANK)
    a = p.parse_args(["--device", "Webcam", "identify", "--threshold", "0.3"])
    assert (a.device, a.threshold, a.cpu) == ("Webcam", 0.3, False)
