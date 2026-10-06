"""Log-mel front-end + timm CNN backbone.

Features are computed on the GPU inside the model, so the DataLoader only
moves raw waveforms and SpecAugment runs on-device.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torchaudio.transforms as T


class LogMelFrontend(nn.Module):
    def __init__(self, sample_rate: int, feat: dict, aug: dict | None = None):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=feat["n_fft"],
            hop_length=feat["hop_length"],
            n_mels=feat["n_mels"],
            f_min=feat.get("f_min", 0),
            f_max=feat.get("f_max"),
            power=2.0,
        )
        aug = aug or {}
        self.n_masks = aug.get("n_masks", 0)
        self.freq_mask = T.FrequencyMasking(aug.get("freq_mask", 0)) if aug.get("freq_mask") else None
        self.time_mask = T.TimeMasking(aug.get("time_mask", 0)) if aug.get("time_mask") else None

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:  # [B, T] -> [B, 1, n_mels, frames]
        with torch.autocast(device_type=wav.device.type, enabled=False):
            x = torch.log(self.mel(wav.float()) + 1e-6)
        # per-utterance, per-frequency mean normalisation (cepstral-mean-style)
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / (x.std(dim=(-2, -1), keepdim=True) + 1e-5)
        if self.training:
            for _ in range(self.n_masks):
                if self.freq_mask is not None:
                    x = self.freq_mask(x)
                if self.time_mask is not None:
                    x = self.time_mask(x)
        return x.unsqueeze(1)


class SpectreNet(nn.Module):
    def __init__(self, cfg: dict, num_classes: int):
        super().__init__()
        self.frontend = LogMelFrontend(cfg["data"]["sample_rate"], cfg["features"], cfg.get("augment"))
        m = cfg["model"]
        self.backbone = timm.create_model(
            m["backbone"],
            pretrained=m.get("pretrained", True),
            in_chans=1,
            num_classes=num_classes,
            drop_rate=m.get("dropout", 0.0),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.frontend(wav))

    @torch.no_grad()
    def embed(self, wav: torch.Tensor) -> torch.Tensor:
        """Pooled penultimate features — a first, untrained-for-it speaker embedding (phase 2 replaces this)."""
        feats = self.backbone.forward_features(self.frontend(wav))
        return self.backbone.forward_head(feats, pre_logits=True)
