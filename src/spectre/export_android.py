"""Export a SPECTRE model and its conversation calibration for the Android app.

    python -m spectre.export_android                          # latest model → android/app/src/main/assets/
    python -m spectre.export_android --ckpt runs/<run>/best.pt --out some/folder

Writes two files:
  spectre_ecapa.onnx  16 kHz mono waveform [batch, samples] → unit-length 192-d voice embedding. The log-mel
                      front-end is inside the model, with the STFT written as a fixed convolution, so it only
                      needs plain ONNX operators that every runtime has. The weights are stored as float16 (half
                      the size, so the app can be shared as one small APK); ONNX Runtime turns them back into
                      float32 when it loads the model, and the embeddings stay within 0.001 (--fp32 keeps them).
  voices.json         the voice rules measured by `python -m spectre.conv_eval` (thresholds, room compensation).
Nothing is written unless the ONNX model gives the same embeddings as PyTorch.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import live
from .conversation import CALIBRATION, VoiceRules
from .model import LogMelFrontend

DEFAULT_OUT = Path("android/app/src/main/assets")
MODEL_FILE, RULES_FILE = "spectre_ecapa.onnx", "voices.json"


class ConvLogMel(nn.Module):
    """LogMelFrontend (STFT → power → mel → log → normalisation) with the STFT as two fixed convolutions."""

    def __init__(self, fe: LogMelFrontend):
        super().__init__()
        spec, mel = fe.mel.spectrogram, fe.mel.mel_scale
        if not (spec.center and spec.pad_mode == "reflect" and spec.onesided and spec.power == 2.0 and not spec.normalized):
            raise ValueError("unexpected spectrogram settings")
        n_fft, window = spec.n_fft, spec.window.double()
        if len(window) < n_fft:                                   # torch.stft centres a shorter window
            left = (n_fft - len(window)) // 2
            window = F.pad(window, (left, n_fft - len(window) - left))
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
        n = torch.arange(n_fft, dtype=torch.float64)[None, :]
        angle = 2 * math.pi * k * n / n_fft
        self.register_buffer("cos", (torch.cos(angle) * window).float()[:, None, :])
        self.register_buffer("sin", (-torch.sin(angle) * window).float()[:, None, :])
        self.register_buffer("fb", mel.fb.float())
        self.n_fft, self.hop = n_fft, spec.hop_length

    def forward(self, wav: torch.Tensor) -> torch.Tensor:          # [B, T] -> [B, n_mels, frames]
        x = F.pad(wav.unsqueeze(1), (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        re, im = F.conv1d(x, self.cos, stride=self.hop), F.conv1d(x, self.sin, stride=self.hop)
        mel = torch.matmul((re * re + im * im).transpose(1, 2), self.fb).transpose(1, 2)
        x = torch.log(mel + 1e-6)
        x = x - x.mean(dim=-1, keepdim=True)
        return x / (x.std(dim=(-2, -1), keepdim=True) + 1e-5)


class OnnxEmbedder(nn.Module):
    """Waveform → unit-length embedding, exactly as spectre.live.Embedder computes it."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.frontend, self.encoder = ConvLogMel(model.frontend), model.encoder

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(self.frontend(wav)), dim=-1)


def export_onnx(model: nn.Module, path: Path, opset: int = 17) -> None:
    net = OnnxEmbedder(model).eval()
    kw = dict(input_names=["wav"], output_names=["emb"], opset_version=opset,
              dynamic_axes={"wav": {0: "batch", 1: "samples"}, "emb": {0: "batch"}})
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(net, (torch.randn(1, 48000) * 0.1,), str(path), dynamo=False, **kw)
        except TypeError:                                          # older PyTorch: no "dynamo" argument
            torch.onnx.export(net, (torch.randn(1, 48000) * 0.1,), str(path), **kw)


def weights_to_fp16(path: Path, min_size: int = 1024) -> int:
    """Store the large float32 weights of an ONNX model as float16, each followed by a Cast back to float32 (which
    ONNX Runtime folds away when it loads the model: the arithmetic stays float32). Returns how many."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    m = onnx.load(str(path))
    casts = []
    for init in m.graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        a = numpy_helper.to_array(init)
        if a.size < min_size:                                      # biases, scales and constants stay float32
            continue
        name = init.name
        init.CopyFrom(numpy_helper.from_array(a.astype(np.float16), name + "_fp16"))
        casts.append(helper.make_node("Cast", [name + "_fp16"], [name], to=TensorProto.FLOAT, name=name + "_to_fp32"))
    for c in reversed(casts):
        m.graph.node.insert(0, c)
    onnx.checker.check_model(m)
    onnx.save(m, str(path))
    return len(casts)


def max_difference(model: nn.Module, path: Path, clips: list[np.ndarray]) -> float:
    """Largest difference between the ONNX and PyTorch embeddings over the clips."""
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    with torch.no_grad():
        for clip in clips:
            wav = torch.from_numpy(np.ascontiguousarray(clip, dtype=np.float32))[None]
            ref = F.normalize(model.embed(wav).float(), dim=-1).numpy()
            out = session.run(None, {"wav": wav.numpy()})[0]
            worst = max(worst, float(np.abs(out - ref).max()))
    return worst


def voice_rules(ckpt: Path, model_id: str) -> dict:
    """The calibration of spectre.conv_eval for this model (or the defaults, marked as not calibrated)."""
    f = ckpt.parent / CALIBRATION
    d = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    if d.get("model") == model_id:
        keys = ("threshold", "cut", "merge_margin", "follow_below_s", "new_margin", "min_cut_side_s", "room_directions",
                "accuracy")
        return {"model": model_id, "calibrated": True, **{k: d[k] for k in keys}}
    r = VoiceRules()
    return {"model": model_id, "calibrated": False, "threshold": r.threshold, "cut": r.cut,
            "merge_margin": r.merge_margin, "follow_below_s": r.follow_below_s, "new_margin": r.new_margin,
            "min_cut_side_s": r.min_cut_side_s, "room_directions": [], "accuracy": None}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m spectre.export_android", description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", type=Path, help="SPECTRE ECAPA checkpoint (default: the latest phase-2 run)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder (default: %(default)s)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--fp32", action="store_true", help="keep the weights as float32 (a model twice as big)")
    a = p.parse_args(argv)
    if any(importlib.util.find_spec(m) is None for m in ("onnx", "onnxruntime")):
        sys.exit("The export needs the 'onnx' and 'onnxruntime' packages:  pip install onnx onnxruntime")
    ckpt = a.ckpt or live.find_checkpoint()
    emb = live.Embedder(ckpt, "cpu")
    model = emb.model.eval()
    rng = np.random.default_rng(0)
    clips = [rng.standard_normal(n).astype(np.float32) * 0.05 for n in (4800, 16000, 47999, 96000)]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_model = Path(tmp) / MODEL_FILE
        export_onnx(model, tmp_model, a.opset)
        if not a.fp32:
            weights_to_fp16(tmp_model)
        diff = max_difference(model, tmp_model, clips)
        if diff > 1e-3:
            sys.exit(f"The ONNX model differs from PyTorch (max difference {diff:.2e}); nothing was written.")
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / MODEL_FILE).write_bytes(tmp_model.read_bytes())
    rules = voice_rules(Path(ckpt), emb.model_id)
    (a.out / RULES_FILE).write_text(json.dumps(rules, indent=1), encoding="utf-8")
    size = (a.out / MODEL_FILE).stat().st_size / 2 ** 20
    print(f"✓ {a.out / MODEL_FILE} ({size:.0f} MB, {'float32' if a.fp32 else 'float16'} weights; same embeddings as "
          f"PyTorch: max difference {diff:.1e})")
    print(f"✓ {a.out / RULES_FILE} ({'calibrated' if rules['calibrated'] else 'NOT calibrated: run python -m spectre.conv_eval first'}"
          f"{', same voice ≥ %.2f' % rules['threshold']})")


if __name__ == "__main__":
    main()
