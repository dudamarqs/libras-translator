# Real-Time Libras Translator

Fingerspelling recognition (the manual alphabet of Libras, Brazilian Sign
Language) from a webcam, translated to text in real time — with the prediction
confidence shown alongside, and with the system **refusing to answer** when it
does not recognize what it sees.

> **Status:** in progress — stage 11/19. The full circuit already runs live:
> camera -> landmarks -> features -> classifier -> word assembler -> corrector
> -> Portuguese caption on screen.
> The current model recognizes the **20 static letters** of the alphabet
> (A B C D E F G I L M N O P Q R S T U V W), trained on 9,271 samples
> across 5 capture sessions.

![Live demo: fingerspelling L-I-B-R-A-S letter by letter, the caption resolving to
"libras", and the system answering "?" when shown a hand that is not a letter](docs/demo.gif)

*Recorded from the demo itself (`scripts/reconhecer.py --gravar`). Yellow is the raw
spelling, green is the corrected caption. The red `?` near the end is the system
refusing to guess.*

---

## How the system works

```
  webcam  ──►  MediaPipe  ──►  normalization ──►  classifier  ──►  "C" (0.98)
  (frame)      (landmarks)     (63-d vector)      (LogReg)         letter + confidence
                                                       │                     │
                                                       ▼                     ▼
                                                novelty detection       assembler
                                             (far from training? "?")  (letters -> word)
                                                                             │
                                                                             ▼
                                                                    corrector (thread)
                                                                    "CAURO" -> "carro"
```

The central architectural point: **we do not classify pixels.** MediaPipe — a
network pre-trained by Google — converts each frame into the 3D coordinates of
the hand joints. A 921,600-byte frame becomes a vector of **63 numbers**
(21 points x 3 axes) describing only the **geometry** of the gesture, already
free of background, lighting and the signer's appearance.

That is transfer learning, and it is what makes the project feasible: ~100-200
samples per sign are enough, and the model trains in seconds on a CPU.

The full reasoning — including what was discarded and why — is in
[docs/DECISOES.md](docs/DECISOES.md) (21 ADRs, in Portuguese).

---

## The most interesting problem this project solves

A **closed-set** classifier knows N hand shapes and nothing else. Given any
hand, it is *forced* to pick one of the N — there is no "none of them" output.

In the first live test, the system saw a hand **scratching a head** and called
it `C` with **100% confidence**. And this is not a training bug: in a linear
model, the farther a point falls from the decision boundary, the **more
confident** the softmax becomes. Raising the confidence threshold does not fix
it — the model is 100% confident in the wrong answer.

The solution was two independent defenses:

| defense | what it attacks |
| ------- | --------------- |
| **Novelty detection** — distance to the K nearest training neighbors; too far ⇒ `unknown` | the confident guess on hands that are not letters |
| **Stillness gate** — only classifies when the hand *shape* has stopped changing | letters captured mid-transition |

The novelty threshold is not a magic number: it comes from the data's own
distribution (99th percentile of intra-training distances). Verification:
**0%** of real letters blocked, **100%** of invalid geometry blocked.

Details in [ADR-016](docs/DECISOES.md).

---

## Results

Validation with **`LeaveOneGroupOut` over the capture sessions** — train on N-1
sessions, test on the one left out. Each fold answers: *"I trained under two
lighting conditions; do I get it right under a third one I have never seen?"*

*(20 static letters, 9,271 samples, 5 sessions on different days and lighting.)*

**The honest number is 88.4%.** The mean across the 5 folds is 95.9%, but that
mean is misleading: three of those folds are older sessions containing only
A, B, C, L and O — five easy letters, where the model scores 98-100% and
inflates the average. Only two folds test the full alphabet:

| test session | letters in the fold | accuracy |
| ------------ | ------------------- | -------- |
| 20260715 | 5 | 98.4% |
| 20260716 | 5 | 100% |
| 20260717 | 5 | 99.7% |
| **20260723** | **20** | **93.1%** |
| **20260728** | **20** | **88.4%** |

Model comparison under the same validation:

| model | accuracy (LOGO mean) | note |
| ----- | -------------------- | ---- |
| **LogisticRegression** | **95.9%** | production model |
| RandomForest | 93.2% | less stable across sessions |
| MLP (PyTorch) | 96.0% | ties on the mean and **loses** on the hard session (87.8% vs 88.4%) |

> **Honest caveat.** The sessions come from the **same person and the same
> hand**. The number proves generalization across **lighting and position**,
> not across **people**. A hand with different proportions is territory the
> model has never seen.

### The neural network was built — and lost

The MLP (63->128->64->20) reached 100% on training and 87.8% on test: a
**12.3-point gap**, memorization without generalization. The linear model stays
in production.

This answers the question that matters: if model capacity were the bottleneck,
the network — which proved it had more than enough capacity to memorize the
training set — would have generalized better. It did not. **The bottleneck is
the data**, specifically how consistent the signing is across sessions in the
R↔U↔V and T↔F clusters.

The diagnosis confirmed it: R/U and T/F are 99-100% separable *within* a session
and drop to 55-80% *across* sessions. And the attempt to fix it with rotation
augmentation made things **worse** (92.5% -> 90.9%), because R, U and V are
distinguished precisely by finger orientation — teaching rotation invariance
erases the very signal that separates the classes.

See [ADR-002](docs/DECISOES.md), [ADR-015](docs/DECISOES.md),
[ADR-017](docs/DECISOES.md) and [ADR-018](docs/DECISOES.md).

---

## Running

After [Setup](#setup):

```powershell
# 1. collect the dataset (one session = one lighting condition, full alphabet)
python training\collect.py

#    re-record specific letters of a saved session, without redoing the rest
python training\collect.py --corrigir sessao_20260721_232645 --letras "A B C"

# 2. raw -> features
python training\preprocess.py

# 3. honest baseline (LeaveOneGroupOut + confusion matrix)
python training\train.py

# 4. train the final model on everything and export it for production
python training\exportar.py

# 5. the live demo (webcam -> letters -> words -> corrected caption)
python scripts\reconhecer.py

# 5b. the text layer without a camera ("CAURO" -> "carro"), to see both correctors
python scripts\demo_legenda.py

# 6. same live demo, recording itself to docs/demo.gif ('g' starts and stops)
python scripts\reconhecer.py --gravar
```

The recorder writes the demo's own frames, so the GIF carries the HUD and
nothing else — no cursor, no window chrome, no desktop behind it. The REC
indicator is drawn *after* the capture, so it stays on screen and out of the
file. Sensor noise is the expensive part of a GIF, not motion: pixels that
change less than the noise threshold are frozen, which took a simulated 15 s
clip from 12.89 MB to 0.32 MB.

Correction **does not run inside the video loop**. A frame lasts 33 ms and one
correction costs 582 ms offline (or 1-3 s via LLM): calling it directly would
freeze the video for 17 to 90 frames. It runs on a thread, with a single-request
slot — last one wins — and a generation counter that invalidates in-flight
results, otherwise the caption "resurrects" text the user has already deleted.
Measured cost inside the loop: **0.03 ms**. See [ADR-021](docs/DECISOES.md).

In the demo, every frame falls into one of four states — and **only green
types**:

| state | when |
| ----- | ---- |
| `desconhecido` (`?`) | hand far from anything the model has seen |
| `movendo...` | the hand shape is still changing |
| `incerto` | still, but confidence below the threshold |
| `ok` | recognized + still + confident + stable |

The HUD shows `mov:` and `dist:` — the raw numbers behind those decisions, so
the thresholds can be recalibrated by looking at data instead of guessing.

---

## Layout

| Folder | Role |
| ------ | ---- |
| `libras/` | **Shared package.** Landmark extraction and normalization, dataset, classifier and the sign catalog. Imported by *both* training *and* inference — that is what prevents training/serving skew ([ADR-004](docs/DECISOES.md)). |
| `training/` | ML lifecycle scripts: `collect.py`, `preprocess.py`, `train.py`, `exportar.py`. |
| `scripts/` | Runnable entry points: `reconhecer.py` (live demo), `doctor.py`, previews and benchmarks. |
| `tests/` | 104 tests (pytest), including deterministic concurrency tests (no `sleep`). |
| `docs/` | `DECISOES.md` — 21 ADRs with the reason behind each choice. |
| `datasets/` | `raw/` (what the webcam captured) and `processed/` (features). **Outside Git.** |
| `models/` | Trained model + `meta.json` (classes, metrics, commit). **Outside Git.** |
| `backend/` | *Empty — planned.* FastAPI service with an inference WebSocket. |
| `frontend/` | *Empty — planned.* React + Vite + Tailwind interface. |
| `notebooks/` | *Empty — planned.* Exploratory experimentation. |

`datasets/` and `models/` stay **outside Git**: Git keeps the full history of
every file, so a dataset re-recorded 10 times turns into a huge repository —
forever, even after you delete the file.

---

## Setup

Requires **Python 3.12** (MediaPipe does not publish wheels for 3.13/3.14).

```powershell
# 1. venv with Python 3.12 explicitly
py -3.12 -m venv .venv

# 2. activate
.venv\Scripts\Activate.ps1

# 3. dependencies (use -dev to develop; requirements.txt alone is the runtime)
python -m pip install -r requirements-dev.txt

# 4. make the `libras` package importable from anywhere
python -m pip install -e .

# 5. download the MediaPipe model
python scripts\download_models.py

# 6. verify that everything actually works
python scripts\doctor.py
```

`doctor.py` validates what `pip install` does **not**: whether numpy, OpenCV and
MediaPipe exchange arrays without blowing up the C ABI, whether the webcam
delivers a real frame, and whether the `libras` package is importable.

---

## Scope

**What is done:** **static** fingerspelling — the letters that fit in a single
frame. One-handed recognition, in real time, with abstention.

**Also done:** the text layer — the assembler joins letters into words and
sentences (pure logic, no AI), and the corrector fixes the result. There are two
implementations behind the same interface: one via LLM (Claude) and one
**offline and free** (dictionary + edit distance). A factory picks the best
available one, so the project runs for free by default and the LLM is an
optional upgrade — and the offline one becomes the fallback when the API fails.

**What is not done:** the letters **with movement** (H, J, K, X, Y, Z) require a
temporal model (LSTM) and belong to the dynamic phase — they are declared as
`dinamico` in [`sinais.yaml`](libras/registry/sinais.yaml) and the collector
skips them automatically. Two-handed signs are also a later phase.

There is no off-the-shelf Libras model: the public models are for **ASL**, which
is a different language. MediaPipe delivers landmarks, not meaning — the meaning
comes from the data you label.

---

## Stack

**Computer vision:** OpenCV (capture) + MediaPipe (landmarks).
**ML:** scikit-learn (baseline and production model) + PyTorch (MLP evaluated in
[ADR-017](docs/DECISOES.md) — built, measured and discarded).
**Text layer:** pure assembler + corrector via Claude (`anthropic`) or offline
(`pyspellchecker`), behind the same interface.
**Quality:** pytest + ruff.
**Planned:** FastAPI + Uvicorn (WebSocket at 30 FPS) and React + Vite + Tailwind.
