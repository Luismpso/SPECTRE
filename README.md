# 🎙️ SPECTRE

**SPE**ctral **C**lassifier for **T**alker **RE**cognition — identifying who is speaking from the frequency spectrum of their voice.

> Status: **Phase 1 — baseline in progress**

## Idea

The approach borrows from the BirdCLEF Kaggle competitions: audio is turned into **log-mel spectrograms** and treated as images by an ImageNet-pretrained CNN. Speakers differ from bird species in one important way, though: the set of people is open. The project therefore evolves in phases:

| Phase | Goal | Method |
|---|---|---|
| **1 · Baseline** ✅ pipeline | Closed-set classification of a fixed set of speakers | Log-mel → EfficientNet (timm) → softmax |
| **2 · Embeddings** | Open-set identification: enroll a new person with a few seconds of speech, no retraining | ECAPA-TDNN / ResNet + AAM-softmax, cosine scoring, EER |
| **3 · Demo** | Live microphone identification with an "unknown speaker" threshold | Enrollment + real-time inference |

## Design choices

- **Session-aware splits.** LibriSpeech chapters are separate recording sessions. Test chapters are never seen during training, so the model has to recognise the *voice*, not the microphone or the room (see `src/spectre/data.py`).
- **No pitch-shift augmentation** — it changes speaker identity. Augmentation is additive noise, random gain and SpecAugment (frequency/time masking).
- **GPU front-end.** Mel features are computed inside the model, so the data loader only moves raw waveforms.
- **Full-utterance test evaluation.** A 3 s window slides over each test utterance and the softmax probabilities are averaged.

## Setup

```bash
conda create -n spectre python=3.11 -y && conda activate spectre
# PyTorch for RTX 50xx (Blackwell) needs CUDA 12.8 builds
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

## Quick start

```bash
# 1. Data — LibriSpeech from OpenSLR
python scripts/download_librispeech.py --subset dev-clean          # 40 speakers, ~340 MB (smoke test)
python scripts/download_librispeech.py --subset train-clean-100    # 251 speakers, ~6.3 GB (baseline)

# 2. Manifest with session-aware train/val/test split
python -m spectre.data --subset train-clean-100

# 3. Train (evaluates the best checkpoint on the test split at the end)
python -m spectre.train --config configs/baseline.yaml

# Evaluate any checkpoint again
python -m spectre.evaluate --ckpt runs/<run>/best.pt --split test
```

Each run writes `config.yaml`, `history.csv`, `best.pt` and `results.json` to `runs/<run_name>_<timestamp>/`.

## Structure

```
configs/baseline.yaml          hyper-parameters
scripts/download_librispeech.py
src/spectre/
  data.py       manifest, session-aware split, waveform dataset + augmentation
  model.py      log-mel front-end (+SpecAugment) and timm backbone
  train.py      training loop (AdamW, OneCycle, bf16 AMP, label smoothing)
  evaluate.py   sliding-window full-utterance evaluation (top-1 / top-5)
```

## Results

| Data | Model | Speakers | Test top-1 | Test top-5 |
|---|---|---|---|---|
| train-clean-100 | EfficientNet-B0 | 251 | — | — |

## License

MIT
