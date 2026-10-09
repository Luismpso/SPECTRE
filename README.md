# 🎙️ SPECTRE

**SPE**ctral **C**lassifier for **T**alker **RE**cognition — identifying who is speaking from the frequency spectrum of their voice.

> Status: **Phase 4 — conversation captions with names** · phases 1–3 complete

## Idea

The approach borrows from the BirdCLEF Kaggle competitions: audio is turned into **log-mel spectrograms** and treated as images by an ImageNet-pretrained CNN. Speakers differ from bird species in one important way, though: the set of people is open. The project therefore evolves in phases:

| Phase | Goal | Method |
|---|---|---|
| **1 · Baseline** ✅ | Closed-set classification of a fixed set of speakers | Log-mel → EfficientNet (timm) → softmax |
| **2 · Embeddings** ✅ | Open-set identification: enroll a new person with a few seconds of speech, no retraining | ECAPA-TDNN + AAM-softmax, cosine scoring, EER |
| **3 · Demo** ✅ | Live microphone identification with an "unknown speaker" threshold | Enrollment + real-time inference |
| **4 · Conversation** ✅ | Live captions that work out who is who from what is said ("Olá João" → the next voice is João) | Voice clustering + Whisper + local LLM + turn-taking rules |

## Design choices

- **Session-aware splits.** LibriSpeech chapters are separate recording sessions. Test chapters are never seen during training, so the model has to recognise the *voice*, not the microphone or the room (see `src/spectre/data.py`).
- **No pitch-shift augmentation** — it changes speaker identity. Augmentation is additive noise, random gain and SpecAugment (frequency/time masking).
- **Fast data loading.** Only the training/validation crop (2–3 s) is read from disk (≈15× faster than decoding the whole FLAC), and mel features are computed on the GPU inside the model.
- **Full-utterance test evaluation.** A crop-length window (3 s for the CNN, 2 s for ECAPA) slides over each test utterance and the softmax probabilities are averaged.

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

# 3c. Phase 2b — same model, 1,172 training speakers (~30 GB of audio; desktop GPU recommended)
python scripts/download_librispeech.py --subset train-clean-360
python -m spectre.data --subset train-clean-100 train-clean-360    # merged manifest
python -m spectre.train --config configs/ecapa_460.yaml
python -m spectre.embed_eval --ckpt runs/<run>/best.pt

# Interrupted (Ctrl+C, crash, laptop asleep)? Continue from the last completed epoch
python -m spectre.train --resume runs/<run>/last.pt

# Evaluate any checkpoint again
python -m spectre.evaluate --ckpt runs/<run>/best.pt --split test
```

Each run writes `config.yaml`, `history.csv`, `best.pt` (best validation weights), `last.pt` (full training state for resuming) and `results.json` to `runs/<run_name>_<timestamp>/`.

## Phase 2 — how the embeddings are evaluated

The classifier head is thrown away after training: speakers are compared by the **cosine similarity of 192-d embeddings**. `spectre.embed_eval` reports three things:

- **Verification EER / minDCF** on 80 speakers that were **never seen in training** (LibriSpeech dev-clean + test-clean). Every same-speaker pair from *different* chapters is scored against every different-speaker pair.
- **Enroll-and-identify on unseen speakers** (the 60 of them with ≥ 2 sessions): each new person is enrolled with ~10 s of speech from one session and identified from their other sessions — the "add someone without retraining" scenario.
- **Closed-set identification** of the training speakers by nearest enrolled embedding, directly comparable with the classifier head.

The AAM margin is warmed up over the first epochs (0.04 → 0.2) so training does not collapse early.

## Phase 3 — live demo

```bash
pip install sounddevice                      # microphone access (Linux: sudo apt install libportaudio2)
python -m spectre.live enroll "Ana"          # speak for 20 s
python -m spectre.live enroll "Rui"
python -m spectre.live calibrate             # threshold from the enrolled voices
python -m spectre.live identify              # who is speaking right now? (Ctrl+C to stop)
```

- Uses the most recent phase-2 model (`--ckpt` to pick another) and runs in real time on GPU or CPU.
- Every 0.5 s, the current turn — the speech since the last pause, up to 3 s — is embedded and compared with each enrolled person; below the threshold the voice is reported as **unknown**. Scores are smoothed within a turn and reset at pauses, so a new speaker is picked up within about a second.
- `enroll` and `identify` also accept `--file`, which runs recordings through exactly the same pipeline (handy for testing without a microphone). `devices`, `list`, `remove` and `--device` manage microphones and people.
- **Privacy:** only voice embeddings are stored (`enrollments/speakers.json`), never audio. They are biometric data, so `enrollments/` is git-ignored.
- Expect lower scores than on LibriSpeech: the model was trained on English audiobooks, while a live demo has another microphone, room and language. Enrolling with the same microphone used for identification helps. With two or more people enrolled, `calibrate` sets the threshold from how similar the enrolled voices are to each other (otherwise the model's own EER threshold is used).

## Phase 4 — who is who, from what is said

Live captions of a conversation, each line labelled with who said it. Nobody has to be enrolled: the names come from the conversation itself. If Joana says *"Olá João!"* and another voice answers *"Olá Joana, tudo bem?"*, that voice is João (labelled as soon as he starts talking) and the first one is Joana.

```bash
pip install transformers rich sounddevice scipy   # Whisper, live panel, microphone (or: pip install -e ".[conversation]")
# local LLM: install Ollama (https://ollama.com), then
ollama pull gemma3:12b
python -m spectre.conv_eval                                       # once per model: calibrate the voices (a few minutes)
python -m spectre.conversation                                    # live, from the microphone (Ctrl+C to stop)
python -m spectre.conversation --file conversa.wav --realtime     # a recording, played as if it were live
python -m spectre.conversation --file conversa.wav --save conversa.txt
```

Each turn ends at a pause, and is also cut where the voice changes (quick replies). Then:

1. **Voice** — the SPECTRE embedding, with the room compensated (below), is clustered online: close enough to a known voice means the same speaker, otherwise a new one; speakers that turn out to be the same voice are merged. Turns with less than 1 s of speech are too short to trust: they follow the closest voice, and move if a closer one appears later. People enrolled in phase 3 are recognised by voice and keep their names.
2. **Words** — Whisper (`openai/whisper-large-v3-turbo`, Portuguese by default, `--language auto` to detect) transcribes the turn.
3. **Names** — a local LLM (Ollama, `gemma3:12b` by default) reads the new line and only says *how* each name appears in it: the speaker's own name ("Eu sou o Pedro"), someone addressed ("Olá João"), introduced ("apresento-te o Pedro") or only mentioned ("A Rita chega mais tarde"). The code then checks the answer against the words: names not written in the line are dropped (small models copy them from the context), "own name" needs a self-introduction such as *sou*, *chamo-me* or *o meu nome é* ("Olá Joana" calls Joana), and a name after a greeting or set off by commas is someone being addressed.
4. **Who is who** — turn-taking rules turn those cues into evidence for each voice: their own name +3; the next *other* voice after a name is called or introduced +2; the previous other voice when a reply uses a name ("Olá Joana, tudo bem?") +2 — a name in an answer ("Obrigado, Maria", "O prazer é meu, Pedro") only points back; saying a name to or about someone else −3 for that name. A voice needs 2 points and each name goes to one voice. Names are applied to earlier lines too, and the end of the session lists the evidence for each name.

The LLM only reads and the code decides, so a small local model is enough, each call is one short line and every name can be explained. While someone is still talking, the panel already shows who it is (*▶ João is speaking…*), with the same rules.

- Everything runs locally and audio is never written to disk. `--save` writes the transcript; `--remember` adds the voices that got a name (≥ 8 s of speech) to `enrollments/speakers.json`, so next time they are recognised by voice.
- `--llm none` gives captions and voices without names; `--llm rules` reads the names with rules instead of an LLM (common first names and the words around them: instant, nothing to install, what the Android app uses); `--llm hf:<model>` uses a Hugging Face chat model instead of Ollama; `ollama:gemma3:4b` is a lighter option. On a 16 GB GPU, Whisper turbo and `gemma3:12b` fit side by side.
- Tuning: `--cluster-threshold` (default: the calibration, or 0.65 without one) — raise it if two people share a label, lower it if one person shows up as two; `--gap` (0.6 s) — the pause that ends a turn.
- Limits: overlapping speech is not separated, very short turns ("Sim.") are harder to attribute, and someone is only named once their name is said. Similar voices can still be merged, mostly when people answer each other very quickly; enrolling them (phase 3) avoids that.

### Voices around one microphone

Every LibriSpeech speaker reads in their own room with their own microphone, so a model trained on it learns that the channel is part of the voice. Around one microphone everyone shares the channel, and different people suddenly look alike: on the same simulated conversations, two *different* speakers typically score 0.04 when each voice is played as recorded, but 0.36–0.47 once they all go through one room (reverberation, microphone colour, background noise). The same person barely changes. A threshold taken from LibriSpeech (0.27 at the EER point) therefore merges people.

`python -m spectre.conv_eval` measures and corrects this for each model, with voices it never heard:

1. **Room compensation** — dev-clean clips are played in simulated rooms; the directions in which a room moves the embeddings are learned and removed from every embedding afterwards (nuisance attribute projection), from the first turn on.
2. **Calibration** — 150 one-microphone conversations between 2–4 test-clean speakers (other people than in step 1; answers 0.15–1 s after the other person stops, some within one turn) go through exactly the live pipeline. The settings that give the most speech to the right person, without inventing extra speakers, are saved next to the model (`runs/<run>/conversation.json`) and used automatically.

| Voice rules (1,172-speaker model) | Speech given to the right person | Conversations with two people merged |
|---|---|---|
| First version: threshold from LibriSpeech (EER + 0.1) | 56.6 % | 88 % |
| Calibrated threshold + rule for short turns | 85.8 % | 39 % |
| + room compensation | **89.1 %** | **31 %** |

The remaining errors come from the model itself, which still carries the room; training it with reverberation and microphone augmentation is the natural next step.

## Phase 5 — on the phone (Android)

`android/` is the same live captions as an Android app in Kotlin, with everything on the phone: no PC, no server, and the audio never leaves it. The screen is a terminal like the PC panel. It only reads the captions, so another display (Meta's glasses) can show them later.

```bash
pip install onnx onnxruntime
python -m spectre.export_android      # your model → android/app/src/main/assets/ (spectre_ecapa.onnx + voices.json)
cd android
./gradlew assembleRelease             # → app/build/outputs/apk/release/app-arm64-v8a-release.apk (or open android/ in Android Studio)
```

The first build downloads and compiles whisper.cpp (a few minutes). The APK (about 27 MB) is signed with the debug key, so it installs directly: copy it to the phone and open it (allow installing from unknown sources), or `adb install`.

| | PC | Phone |
|---|---|---|
| Voices | SPECTRE in PyTorch | the same model on ONNX Runtime (the STFT written as a convolution, the weights stored as float16: 13 MB, embeddings within 0.001 of PyTorch), with the PC's calibration and room compensation (`voices.json`) |
| Words | Whisper large-v3-turbo (transformers) | Whisper on whisper.cpp, quantised: *small* by default (190 MB), *base* (60 MB, fastest) or *turbo* (574 MB, best and slowest), downloaded once on first use |
| Names | local LLM, or `--llm rules` | the same rules as `--llm rules` |

- `android/core` is plain Kotlin: turns, voices, names and the conversation, ported line by line from `spectre.conversation` and checked against the Python with shared fixtures (`python scripts/make_android_golden.py`, then `./gradlew :core:test`). It has no Android code, so it can move to Kotlin Multiplatform for an iPhone version.
- `android/app` is the Android part: the microphone (16 kHz, no automatic gain), ONNX Runtime, whisper.cpp through JNI, the model download and the screen.
- Whisper encodes only the turn (with at least 10 s of context) instead of a 30-s window: about 3× faster, about as accurate on the test conversation. On a 2-core x86 test machine *small* runs at 1.3× real time and *base* at 0.4×. A phone with four fast cores should do better, but this hasn't been measured on a phone yet: if the captions fall behind (the status line counts the turns waiting), choose *base*.
- Needs Android 10+ and a 64-bit ARM processor with ARMv8.2 dot-product instructions (from about 2018; aimed at flagships from 2022 on). The screen stays on while listening. *Share* sends the transcript with how the names were found.
- Next, Meta's glasses: with the Wearables Device Access Toolkit the app stays on the phone and the glasses are a display (600 × 600; each update replaces the whole screen). The audio should still come from the phone, because the glasses' Bluetooth microphone is 8 kHz and beamformed towards the wearer, which suppresses exactly the other voices. Listening also moves to a foreground service, so it goes on with the screen off.

## Structure

```
configs/baseline.yaml          phase 1 hyper-parameters
configs/ecapa.yaml             phase 2 hyper-parameters
configs/ecapa_460.yaml         phase 2b: same model, 1,172 training speakers
scripts/download_librispeech.py
src/spectre/
  data.py       manifest, session-aware split, waveform dataset + augmentation
  model.py      log-mel front-end (+SpecAugment), timm CNN (phase 1), embedding net (phase 2)
  ecapa.py      ECAPA-TDNN encoder and AAM-softmax head
  train.py      training loop (AdamW, OneCycle, bf16 AMP, resumable)
  evaluate.py   sliding-window full-utterance classification (top-1 / top-5)
  embed_eval.py EER / minDCF, enroll-and-identify, closed-set by embedding
  live.py       live demo: enrollment, voice activity detection, real-time identification
  conversation.py  live captions: turns, voice clustering, Whisper, names from context (local LLM)
  room.py       simulated rooms and one-microphone conversations
  conv_eval.py  room compensation and voice calibration for conversations
  export_android.py  the model and voice rules for the Android app (ONNX + voices.json)
tests/        fast tests for phases 3–4 (no data, GPU, microphone or Ollama needed): pytest -q
scripts/make_android_golden.py  fixtures that check the Kotlin port against the Python
android/      Android app (phase 5): core/ = conversation logic in plain Kotlin, app/ = microphone, models, screen
```

## Results

Phase 1 — closed-set classification (251 known speakers, test = unseen recording sessions):

| Data | Model | Test top-1 | Test top-5 |
|---|---|---|---|
| train-clean-100 | EfficientNet-B0 (ImageNet init) | 84.3 % | 91.1 % |

Phase 2 — ECAPA-TDNN speaker embeddings (C = 512, 6.2 M parameters, trained from scratch with AAM-softmax):

| Training data | Train speakers | Known speakers · top-1 | Unseen speakers · EER ↓ | Unseen · minDCF (p = 0.01) ↓ | Unseen · enroll 10 s → top-1 |
|---|---|---|---|---|---|
| train-clean-100 | 251 | 94.3 % | 6.75 % | 0.432 | 87.4 % |
| train-clean-100 + 360 | 1,172 | 90.2 % | **6.23 %** | **0.308** | **91.9 %** |

- **More training speakers generalise better to new people.** With 4.7× more speakers, enroll-and-identify errors on unseen speakers fall from 12.6 % to 8.1 % (−36 %), minDCF by 29 % and EER from 6.75 % to 6.23 %.
- **Known speakers**: full test utterances from recording sessions never seen in training (top-5: 97.4 % with 251 speakers, 94.6 % with 1,172). Identifying them by the nearest enrolled embedding instead of the classifier head gives practically the same accuracy (94.1 % and 90.2 %), so the 192-d embedding alone carries the identity. Compared with phase 1, the 251-speaker closed-set error drops from 15.7 % to 5.7 %. This number depends on how many speakers are known, so it is not comparable across rows.
- **Unseen speakers**: the 80 LibriSpeech dev-clean + test-clean speakers, never heard in training; enroll-and-identify uses the 60 with ≥ 2 sessions (top-5: 94.3 % → 95.2 %). Every row is evaluated on these same speakers, so these columns are directly comparable across training sets.
- The 1,172-speaker model trains in under 10 minutes on an RTX 5070 Ti (20 epochs × 25 s). Its cosine threshold at the EER operating point is **0.27**, the starting value for the demo's "unknown speaker" decision.

## License

MIT
