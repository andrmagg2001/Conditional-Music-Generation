# Conditional Music Generation from Guitar Performances using Deep Neural Networks

> **Thesis Project — Sapienza Università di Roma, A.Y. 2025/26**  
> **Author:** Andrea Maggiore — Mat. 1947898  
> **Supervisor:** Prof. Marco Raoul Marini

---

## Abstract

This project implements an end-to-end pipeline that takes a **raw guitar audio recording** as input and generates a coherent **multi-instrument MIDI accompaniment** (drums, bass, harmony) conditioned on the musical content of the input.

The system is composed of two cooperating subsystems:

1. **Audio Analysis Module** — A CNN-based classifier (`ResMelNet`) for genre prediction, combined with signal-processing routines for BPM estimation and chord recognition.
2. **Conditional MIDI Generator** — A causal Transformer language model trained on structured MIDI data, conditioned via token prepending on genre and tempo metadata extracted from the audio.

The key architectural insight is that **conditioning requires no modification to the Transformer itself**: by prepending discrete conditioning tokens (`<GENRE_rock>`, `<BPM_120>`, etc.) to the training sequences, the model learns genre- and tempo-dependent generation patterns through standard causal attention.

---

## System Architecture

```
                        ┌──────────────────────────────────┐
                        │         Input: guitar.wav        │
                        └──────────────┬───────────────────┘
                                       │
                        ┌──────────────▼───────────────────┐
                        │     Audio Analysis Pipeline      │
                        │                                  │
                        │  ┌────────────┐  ┌────────────┐  │
                        │  │  ResMelNet │  │  librosa   │  │
                        │  │   Genre    │  │   BPM      │  │
                        │  └─────┬──────┘  └─────┬──────┘  │
                        │        │               │         │
                        │  ┌─────▼───────────────▼──────┐  │
                        │  │   CQT Chroma → Chords      │  │
                        │  └─────┬──────────────────────┘  │
                        └────────┼─────────────────────────┘
                                 │
                        ┌────────▼─────────────────────────┐
                        │  Conditioning: meta.json         │
                        │  { genre: "rock", bpm: 120, ...} │
                        └────────┬─────────────────────────┘
                                 │
                        ┌────────▼─────────────────────────┐
                        │  Prompt Construction             │
                        │  <GENRE_rock> <BPM_120> <LEN_128>│
                        │  Generate one loop. Output only  │
                        │  tokens. <LOOP>                  │
                        └────────┬─────────────────────────┘
                                 │
                        ┌────────▼─────────────────────────┐
                        │  Causal Transformer LM (25.4M)   │
                        │  d=512, 8 layers, 8 heads        │
                        │  vocab: 427 sub-tokens           │
                        └────────┬─────────────────────────┘
                                 │
                        ┌────────▼─────────────────────────┐
                        │  Output: accompaniment.mid       │
                        │  (drums + bass + harmony)        │
                        └──────────────────────────────────┘
```

---

## Key Design Decisions

### Sub-token Decomposition

A critical challenge was vocabulary size. The naive approach of encoding each MIDI event as a single token (e.g., `D:s=0,e=1,p=42,v=110`) produced **~419,000 unique tokens**, making the embedding table alone consume >400 MB and preventing the use of meaningful model architectures on consumer hardware.

The solution adopted decomposes each event into **5 atomic sub-tokens**:

| Component | Token Example | Range |
|-----------|--------------|-------|
| Role      | `<R_D>`, `<R_B>`, `<R_H>` | 3 values |
| Start step | `<S_0>` ... `<S_127>` | 128 values |
| End step  | `<E_1>` ... `<E_128>` | 128 values |
| Pitch     | `<P_0>` ... `<P_127>` | 128 values |
| Velocity  | `<V_8>` ... `<V_120>` | 16 values (quantized) |

This reduces the vocabulary from **419,255 → 427 tokens**, a **~1000× reduction**. The embedding table drops from 430 MB to <1 MB, enabling a 25M-parameter Transformer to train comfortably on a MacBook Pro with 24 GB unified memory.

### Conditioning via Token Prepending

Instead of modifying the Transformer architecture (e.g., cross-attention, FiLM layers), conditioning is achieved by prepending structured tokens to the input sequence:

```
<GENRE_rock> <BPM_120> <LEN_128> Generate one loop. Output only tokens. <LOOP>
<R_D> <S_0> <E_2> <P_36> <V_96> <R_D> <S_0> <E_1> <P_42> <V_112> ...
</LOOP>
```

The model learns genre→pattern correlations through standard causal attention — no architectural changes required. This approach is consistent with techniques used in MusicLM, MusicGen, and other state-of-the-art generative music models.

---

## Repository Structure

```
Conditional-Music-Generation/
│
├── code/
│   ├── python/
│   │   ├── header.py                   # Constants, MelDataset, ResMelNet, LSCELoss
│   │   ├── feature_extraction.py       # Audio → genre + BPM + chords → meta.json
│   │   ├── enrich_manifest.py          # Adds genre/BPM labels to MIDI manifest
│   │   ├── manifest_to_jsonl.py        # MIDI → sub-token JSONL training data
│   │   ├── split_dataset.py            # Train/val/test split by song_id
│   │   ├── build_vocab.py              # Builds vocabulary from training split
│   │   ├── dataset_loader.py           # PyTorch Dataset + DataLoader
│   │   ├── train_baseline.py           # Transformer LM training loop
│   │   ├── pipeline_generate.py        # End-to-end inference pipeline
│   │   ├── validate_tokenizer_roundtrip.py  # Tokenizer integrity test
│   │   ├── convert_transformer_onnx.py      # PyTorch → ONNX export
│   │   └── preprocess.yaml             # Preprocessing configuration
│   │
│   └── jupyter/
│       ├── dataset_creation.ipynb      # Audio dataset curation + visualization
│       └── music_classificator.ipynb   # ResMelNet training + evaluation
│
├── data/
│   ├── dataset/
│   │   ├── audio/                      # 33 classes of guitar loops (Apple Loops)
│   │   └── midi/                       # 6,843 curated MIDI files
│   │
│   ├── json/
│   │   ├── manifest.jsonl              # Raw MIDI manifest (song_id, paths, BPM)
│   │   ├── manifest_enriched.jsonl     # + genre labels + BPM buckets
│   │   ├── vocab.json                  # 427-token vocabulary
│   │   ├── meta.json                   # Runtime audio analysis output
│   │   └── processed/                  # train.jsonl, val.jsonl, test.jsonl
│   │
│   ├── checkpoints/
│   │   ├── conditioned_transformer/    # Trained Transformer (best.pth, latest.pth)
│   │   ├── guitar_macro_classifier.pth # Trained ResMelNet
│   │   └── guitar_macro_classifier.onnx
│   │
│   └── samples/                        # Generated MIDI outputs
│
└── README.md
```

---

## Models

### ResMelNet — Audio Genre Classifier

| Property | Value |
|----------|-------|
| Architecture | Residual CNN with SE blocks |
| Input | Mel-spectrogram patches (256 bins) |
| Output | Macro-genre class (33 audio categories) |
| Checkpoint | `guitar_macro_classifier.pth` |
| Export | ONNX (`guitar_macro_classifier.onnx`) |

### Conditional Transformer LM — MIDI Generator

| Property | Value |
|----------|-------|
| Architecture | Causal Transformer Encoder |
| Parameters | **25.4M** |
| `d_model` | 512 |
| Layers | 8 |
| Attention heads | 8 |
| FFN dimension | 2,048 |
| Max sequence length | 1,024 |
| Vocabulary | **427 sub-tokens** |
| Conditioning | Genre (11 classes) + BPM (15 buckets) |
| Weight tying | Embedding ↔ LM head |
| Activation | GELU |
| Normalization | Pre-LayerNorm |

---

## Datasets

### Audio Dataset

- **Source:** Apple Loops (Logic Pro)
- **Classes:** 33 guitar styles
- **Preprocessing:** Mel-spectrograms (FFT=4096, hop=256, 256 bins, 20 Hz–Nyquist)
- **Augmentation:** Random time/frequency masking

### MIDI Dataset

| Split | Samples |
|-------|---------|
| Train | 4,089 |
| Val | 511 |
| Test | 512 |
| **Total** | **5,112** |

- **Source:** 6,843 curated MIDI files, filtered to 4/4 time signature
- **Genre distribution (train):** rock 1,680 · pop 935 · unknown 737 · country 163 · funk/disco 158 · jazz 110 · electronic 88 · latin 88 · blues 78 · reggae 42 · classical 10
- **Quantization:** 1/16 grid, 4 steps/beat, 480 TPB
- **Loop length:** 128 steps (8 bars)

---

## Training Results

| Metric | Value |
|--------|-------|
| Training steps | 6,350 |
| Epochs | 50 |
| Train loss | **0.7265** |
| Val loss | **0.9314** |
| Test loss | **0.8589** |
| Best val loss | 0.9261 |
| Effective batch size | 32 (4 × 8 accum) |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, wd=0.1) |
| Learning rate | 3×10⁻⁴ → 3×10⁻⁵ (cosine decay) |
| Hardware | Apple M4 Pro, 24 GB unified memory |

---

## Quick Start

### Prerequisites

```bash
pip install torch torchaudio librosa miditoolkit pyyaml numpy tqdm
```

### 1. Audio Analysis

Extract genre, BPM, and chords from an input guitar recording:

```bash
python3 code/python/feature_extraction.py --audio Test.wav
```

Output saved to `data/json/meta.json`:

```json
{
  "BPM": 83,
  "GENRE": "blues_garage",
  "CHORDS": [[0.725, "A"], [3.136, "D"], [6.261, "A"], ...]
}
```

### 2. Generate MIDI Accompaniment

Run the full inference pipeline — audio analysis → conditioning → generation:

```bash
python3 code/python/pipeline_generate.py \
  --audio Test.wav \
  --ckpt data/checkpoints/conditioned_transformer/best.pth \
  --vocab data/json/vocab.json \
  --out-dir data/samples/output \
  --num-candidates 10 \
  --temperature 0.4 \
  --top-k 12 \
  --max-new-tokens 2000
```

This generates 10 candidate MIDI files in `data/samples/output/`, automatically selects the best one, and validates each output for structural correctness.

#### Generation parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--temperature` | 0.6 | Sampling temperature (lower = more conservative) |
| `--top-k` | 24 | Top-k filtering (lower = less random) |
| `--num-candidates` | 5 | Number of candidates to generate |
| `--max-new-tokens` | 420 | Maximum tokens to generate per candidate |
| `--min-notes` | 16 | Minimum note count for validity |

### 3. Retrain the Model

To retrain from scratch on the conditioned dataset:

```bash
# Step 1: Enrich manifest with genre/BPM metadata
python3 code/python/enrich_manifest.py

# Step 2: Build sub-token training data
python3 code/python/manifest_to_jsonl.py

# Step 3: Split into train/val/test
python3 code/python/split_dataset.py

# Step 4: Build vocabulary
python3 code/python/build_vocab.py

# Step 5: Train
python3 code/python/train_baseline.py \
  --max-steps 8000 \
  --epochs 50 \
  --batch-size 4 \
  --grad-accum 8 \
  --d-model 512 \
  --layers 8 \
  --heads 8 \
  --max-len 1024 \
  --output-dir data/checkpoints/conditioned_transformer
```

### 4. Export to ONNX

```bash
python3 code/python/convert_transformer_onnx.py \
  --ckpt data/checkpoints/conditioned_transformer/best.pth \
  --out data/checkpoints/conditioned_transformer/best.onnx
```

---

## Technical Notes

### Hardware Requirements

- **Training:** Apple Silicon Mac with ≥16 GB unified memory (tested on M4 Pro 24 GB)
- **Inference:** Any machine with Python 3.10+ and PyTorch ≥2.0
- **ONNX runtime:** CPU inference supported via exported `.onnx` model

### Reproducibility

All random seeds are fixed (`seed=42` for training, `seed=1337` for dataset splitting). The dataset split is performed at the song level (no data leakage between train/val/test).

### Limitations and Future Work

- **Genre coverage:** The MIDI dataset has uneven genre distribution (rock and pop dominate). Balanced sampling or class-weighted loss could improve minority-genre generation.
- **Chord conditioning:** Chord information is extracted at inference time but not yet injected as conditioning tokens. Adding `<CHORD_Am>` tokens is a natural next step.
- **Audio-MIDI alignment:** The current system does not perform temporal alignment between input audio and generated MIDI. The output is a stylistically coherent loop, not a synchronized accompaniment.
- **Evaluation metrics:** Formal perceptual evaluation (e.g., listening tests, Fréchet Audio Distance on synthesized MIDI) remains as future work.

---

## References

- Dhariwal, P. et al. *Jukebox: A Generative Model for Music.* arXiv:2005.00341, 2020.
- Agostinelli, A. et al. *MusicLM: Generating Music from Text.* arXiv:2301.11325, 2023.
- Copet, J. et al. *Simple and Controllable Music Generation.* (MusicGen) arXiv:2306.05284, 2023.
- Huang, C.-Z. A. et al. *Music Transformer.* arXiv:1809.04281, 2018.

---

## Author & Supervision

**Author:** Andrea Maggiore — Mat. 1947898  
**Supervisor:** Prof. Marco Raoul Marini  
**Institution:** Sapienza Università di Roma  
**Academic Year:** 2025/26
