# 🎙️ SPECTRE

**SPE**ctral **C**lassifier for **T**alker **RE**cognition — identifying who is speaking from the frequency spectrum of their voice.

> Status: **Phase 2 — speaker embeddings (ECAPA-TDNN + AAM-softmax)**

## Idea

The approach borrows from the BirdCLEF Kaggle competitions: audio is turned into **log-mel spectrograms** and treated as images by an ImageNet-pretrained CNN. Speakers differ from bird species in one important way, though: the set of people is open. The project therefore evolves in phases:

| Phase | Goal | Method |
|---|---|---|
| **1 · Baseline** ✅ | Closed-set classification of a fixed set of speakers | Log-mel → EfficientNet (timm) → softmax |
| **2 · Embeddings** 🔧 | Open-set identification: enroll a new person with a few seconds of speech, no retraining | ECAPA-TDNN + AAM-softmax, cosine scoring, EER |
| **3 · Demo** | Live microphone identification with an "unknown speaker" threshold | Enrollment + real-time inference |

## Design choices

- **Session-aware splits.** LibriSpeech chapters are separate recording sessions. Test chapters are never seen during training, so the model has to recognise the *voice*, not the microphone or the room (see `src/spectre/data.py`).
- **No pitch-shift augmentation** — it changes speaker identity. Augmentation is additive noise, random gain and SpecAugment (frequency/time masking).
- **Fast data loading.** Only the 3 s training/validation crop is read from disk (≈15× faster than decoding the whole FLAC), and mel features are computed on the GPU inside the model.
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

# 3a. Phase 1 — closed-set CNN classifier
python -m spectre.train --config configs/baseline.yaml

# 3b. Phase 2 — ECAPA-TDNN speaker embeddings
python scripts/download_librispeech.py --subset dev-clean           # unseen speakers for evaluation
python scripts/download_librispeech.py --subset test-clean
python -m spectre.data --subset dev-clean && python -m spectre.data --subset test-clean
python -m spectre.train --config configs/ecapa.yaml
python -m spectre.embed_eval --ckpt runs/<run>/best.pt            # EER + enroll-and-identify

# Interrupted (Ctrl+C, crash, laptop asleep)? Continue from the last completed epoch
python -m spectre.train --resume runs/<run>/last.pt

# Evaluate any checkpoint again
python -m spectre.evaluate --ckpt runs/<run>/best.pt --split test
```

Each run writes `config.yaml`, `history.csv`, `best.pt` (best validation weights), `last.pt` (full training state for resuming) and `results.json` to `runs/<run_name>_<timestamp>/`.

## Phase 2 — how the embeddings are evaluated

The classifier head is thrown away after training: speakers are compared by the **cosine similarity of 192-d embeddings**. `spectre.embed_eval` reports three things:

- **Verification EER / minDCF** on 80 speakers that were **never seen in training** (LibriSpeech dev-clean + test-clean). Every same-speaker pair from *different* chapters is scored against every different-speaker pair.
- **Enroll-and-identify on unseen speakers**: each new person is enrolled with ~10 s of speech from one session and identified from their other sessions — the "add someone without retraining" scenario.
- **Closed-set identification** of the 251 training speakers by nearest enrolled embedding, directly comparable with phase 1.

The AAM margin is warmed up over the first epochs (0.04 → 0.2) so training does not collapse early.

## Structure

```
configs/baseline.yaml          phase 1 hyper-parameters
configs/ecapa.yaml             phase 2 hyper-parameters
scripts/download_librispeech.py
src/spectre/
  data.py       manifest, session-aware split, waveform dataset + augmentation
  model.py      log-mel front-end (+SpecAugment), timm CNN (phase 1), embedding net (phase 2)
  ecapa.py      ECAPA-TDNN encoder and AAM-softmax head
  train.py      training loop (AdamW, OneCycle, bf16 AMP, resumable)
  evaluate.py   sliding-window full-utterance classification (top-1 / top-5)
  embed_eval.py EER / minDCF, enroll-and-identify, closed-set by embedding
```

## Results

Phase 1 — closed-set classification (251 known speakers, test = unseen recording sessions):

| Data | Model | Test top-1 | Test top-5 |
|---|---|---|---|
| train-clean-100 | EfficientNet-B0 (ImageNet init) | 84.3 % | 91.1 % |

Phase 2 — speaker embeddings:

| Data | Model | Unseen EER ↓ | Unseen enroll 10 s → top-1 | Known top-1 (by embedding) |
|---|---|---|---|---|
| train-clean-100 | ECAPA-TDNN (C=512) | — | — | — |

## License

MIT
