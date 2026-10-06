"""ECAPA-TDNN speaker-embedding network and Additive Angular Margin (AAM) softmax.

References
  * Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention, Propagation and
    Aggregation in TDNN Based Speaker Verification", Interspeech 2020.
  * Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    """Squeeze-and-excitation over channels (1-D)."""

    def __init__(self, channels: int, bottleneck: int = 128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(bottleneck, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class Res2Conv1d(nn.Module):
    """Res2Net-style multi-scale dilated convolution."""

    def __init__(self, channels: int, kernel: int, dilation: int, scale: int = 8):
        super().__init__()
        assert channels % scale == 0
        self.scale, self.width = scale, channels // scale
        pad = dilation * (kernel - 1) // 2
        self.convs = nn.ModuleList(
            nn.Conv1d(self.width, self.width, kernel, dilation=dilation, padding=pad) for _ in range(scale - 1)
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(self.width) for _ in range(scale - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(x, self.width, dim=1)
        out, y = [], None
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            y = chunks[i] if i == 0 else y + chunks[i]
            y = bn(F.relu(conv(y)))
            out.append(y)
        out.append(chunks[-1])
        return torch.cat(out, dim=1)


class SERes2Block(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, scale: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 1), nn.ReLU(inplace=True), nn.BatchNorm1d(channels),
            Res2Conv1d(channels, kernel, dilation, scale),
            nn.Conv1d(channels, channels, 1), nn.ReLU(inplace=True), nn.BatchNorm1d(channels),
            SEModule(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AttentiveStatsPool(nn.Module):
    """Channel- and context-dependent attentive statistics pooling."""

    def __init__(self, channels: int, attention_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels * 3, attention_dim, 1), nn.ReLU(inplace=True), nn.BatchNorm1d(attention_dim),
            nn.Tanh(), nn.Conv1d(attention_dim, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T] -> [B, 2C]
        t = x.shape[-1]
        mean = x.mean(dim=-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False).clamp(min=1e-5))
        context = torch.cat([x, mean.expand(-1, -1, t), std.expand(-1, -1, t)], dim=1)
        w = torch.softmax(self.attention(context), dim=-1)
        mu = (x * w).sum(dim=-1)
        sigma = torch.sqrt((((x**2) * w).sum(dim=-1) - mu**2).clamp(min=1e-5))
        return torch.cat([mu, sigma], dim=1)


class ECAPATDNN(nn.Module):
    """[B, n_mels, T] log-mel -> [B, emb_dim] speaker embedding."""

    def __init__(self, n_mels: int = 80, channels: int = 512, emb_dim: int = 192, scale: int = 8):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(n_mels, channels, 5, padding=2), nn.ReLU(inplace=True),
                                  nn.BatchNorm1d(channels))
        self.layers = nn.ModuleList(SERes2Block(channels, 3, d, scale) for d in (2, 3, 4))
        self.mfa = nn.Sequential(nn.Conv1d(channels * 3, channels * 3, 1), nn.ReLU(inplace=True))
        self.pool = AttentiveStatsPool(channels * 3)
        self.pool_bn = nn.BatchNorm1d(channels * 6)
        self.fc = nn.Linear(channels * 6, emb_dim)
        self.emb_bn = nn.BatchNorm1d(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        x = self.mfa(torch.cat(feats, dim=1))  # multi-layer feature aggregation
        x = self.pool_bn(self.pool(x))
        return self.emb_bn(self.fc(x))


class AAMSoftmax(nn.Module):
    """Additive angular margin head: logits = s·cos(θ + m) for the target class, s·cos(θ) otherwise.
    The margin is set from outside (`.margin = …`) so it can be warmed up during training."""

    def __init__(self, emb_dim: int, num_classes: int, margin: float = 0.2, scale: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, emb_dim))
        nn.init.xavier_normal_(self.weight)
        self.margin, self.scale = margin, scale

    def forward(self, emb: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(emb.float()), F.normalize(self.weight.float())).clamp(-1 + 1e-7, 1 - 1e-7)
        self.last_cosine = cosine.detach()  # margin-free scores, for honest training accuracy
        if labels is None or self.margin == 0:
            return self.scale * cosine
        m = self.margin
        sine = torch.sqrt(1.0 - cosine**2)
        phi = cosine * math.cos(m) - sine * math.sin(m)  # cos(θ + m)
        # keep the target logit monotonic in θ when θ + m > π
        phi = torch.where(cosine > math.cos(math.pi - m), phi, cosine - math.sin(math.pi - m) * m)
        one_hot = F.one_hot(labels, cosine.shape[1]).to(cosine.dtype)
        return self.scale * (one_hot * phi + (1 - one_hot) * cosine)
