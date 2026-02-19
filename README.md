# Condition Music Generation from Guitar Performances using Deep Neural Networks

This project implements a complete pipeline for **audio analysis** and **genre‑conditioned music generation**.  
It includes dataset creation, feature extraction, deep‑learning‑based classification, and extraction of musical attributes such as BPM and chord progressions.  
The long‑term objective is to provide the conditioning signals necessary for a generative model capable of producing musical accompaniments based on an input guitar performance.

---

## 1. Project Purpose

The goal is to build a system that:
- Analyzes an input audio track.
- Estimates musical attributes (genre, BPM, chord sequence).
- Produces a structured JSON description used to control the future generator.
- Trains a CNN-based classifier (ResMelNet) on curated audio data.
- Extracts mel‑spectrograms and metadata for training and inference.

All components are fully implemented using Python, PyTorch, torchaudio, and librosa.

---

## 2. Dataset and Preprocessing

The dataset is built from **Apple Loops (Logic Pro)**, manually organized into subfolders representing musical styles.  
Two Jupyter notebooks document the entire dataset creation process:

- `code/jupyter/dataset_creation.ipynb`  
  (mel‑spectrogram caching, augmentation, indexing, visualization)

- `code/jupyter/music_classificator.ipynb`  
  (model training, metrics, plots, and evaluation)

### Preprocessing steps
- Loading and resampling audio.
- Converting audio to mel‑spectrograms using:
  - FFT size = 4096  
  - Hop length = 256  
  - 256 mel bins  
  - Frequency range 20 Hz – Nyquist  
- Applying amplitude‑to‑dB conversion.
- Normalizing to the `[0,1]` range.
- Random time/frequency masking for data augmentation.
- JSON index creation containing:
  - File path
  - Mel shape
  - Class index
  - Loop name stem
  - Global preprocessing parameters

For further information click [here](code/Jupyter/dataset_creation.ipynb)

---

## 3. Model: ResMelNet Classifier

The classifier receives a fixed‑size mel‑spectrogram patch and outputs a macro‑genre label.

### Architecture
- Convolutional stem.
- Four residual stages with optional downsampling.
- Squeeze‑and‑Excitation blocks.
- Projection layers between stages.
- Fully‑connected classification head.

### Training
- Optimizer: AdamW  
- Loss: Label‑Smoothed Cross Entropy  
- Scheduler: Cosine Annealing Warm Restarts  
- Gradient accumulation for stability  
- Early stopping based on validation accuracy  
- Automatic GPU selection (CUDA / MPS / CPU)

During training, mel crops are downsampled to reduce temporal and spectral resolution using average pooling.

A checkpoint is saved in:

```
data/checkpoints/guitar_macro_classifier.pth
```
For further information click [here](code/Jupyter/music_classificator.ipynb)

---

## 4. Feature Extraction

The `feature_extraction.py` module provides a unified interface to analyze a raw audio file.

### Extracted information
1. **Genre prediction**  
   - Loads the trained classifier.
   - Builds mel‑spectrogram frontend on the fly.
   - Produces softmax probabilities and genre selection.

2. **BPM estimation**  
   - Uses `librosa.beat.beat_track`.
   - Post‑processes the tempo to reduce octave errors (×2, ÷2).

3. **Chord recognition**  
   - Uses CQT‑based chroma.
   - Beat‑synchronous chroma averaging.
   - Cosine‑similarity matching against 24 chord templates  
     (12 major + 12 minor triads).

### Output format

The analysis results are automatically saved to:

```
data/json/meta.json
```

With the structure:

```json
{
  "BPM": <int>,
  "CHORDS": [
    [<timestamp_sec>, "<chord_label>"],
    ...
  ],
  "GENRE": "<genre_name>"
}
```

---

## 5. Usage

### Running the analysis

```
python code/python/feature_extraction.py
```

This:
- Loads the model if not already in memory.
- Computes mel‑spectrograms.
- Performs prediction and signal analysis.
- Saves metadata as JSON.

---

## Author & Supervision

**Author:** Andrea Maggiore  
**Thesis Supervisor:** Prof. Marco Raoul Marini  
**Department:** Computer Vision Laboratory  
**Institution:** Sapienza University of Rome  
**Academic Year:** 2025/26
# Conditional Music Generation from Guitar Performances

This repository is building an end‑to‑end pipeline that turns an **input guitar performance** into **conditioning signals** (genre, BPM, chords, etc.) and ultimately enables a **generative model** to produce coherent multi‑instrument accompaniments.

At the moment, the project contains:
- A complete **audio analysis + genre classification** pipeline (mel frontend + ResNet‑style model).
- A robust **MIDI preprocessing pipeline** to create **instruction‑tuning style JSONL** examples for loop generation, including variable loop length control.

---

## 1. Project Goal

Given a guitar audio track, the system aims to:
1. **Analyze** the audio and estimate musical attributes (macro‑genre, BPM, chord progression).
2. Produce a structured representation that can be used as **conditioning** for generation.
3. Train / fine‑tune a generative model that can **generate a loop** (drums, bass, harmony) consistent with the conditioning.

---

## 2. Audio Dataset & Preprocessing (Apple Loops)

The audio dataset is built from **Apple Loops (Logic Pro)**, manually organized into subfolders representing musical styles.

Two notebooks document dataset creation and training:
- `code/jupyter/dataset_creation.ipynb` — mel caching, augmentation, indexing, visualization
- `code/jupyter/music_classificator.ipynb` — training, metrics, evaluation

### Mel‑spectrogram frontend
- FFT size: **4096**
- Hop length: **256**
- Mel bins: **256**
- Frequency range: **20 Hz – Nyquist**
- Amplitude → dB
- Normalization to **[0, 1]**
- Augmentation: random time/frequency masking

The preprocessing also produces a JSON index with:
- File path
- Mel shape
- Class index
- Loop name stem
- Global preprocessing parameters

---

## 3. Genre Classifier: ResMelNet

The classifier takes a fixed‑size mel patch and outputs a **macro‑genre** label.

### Training setup
- Optimizer: **AdamW**
- Loss: **Label‑Smoothed Cross Entropy**
- Scheduler: **Cosine Annealing Warm Restarts**
- Gradient accumulation
- Early stopping on validation accuracy
- Automatic device selection (**CUDA / MPS / CPU**)

A checkpoint is saved to:

```text
data/checkpoints/guitar_macro_classifier.pth
```

---

## 4. Feature Extraction (Audio → Conditioning)

The `feature_extraction.py` module provides a unified interface to analyze a raw audio file.

### Extracted information
1. **Genre prediction**
   - Loads the trained classifier.
   - Builds the mel frontend on the fly.
   - Produces softmax probabilities and a predicted genre.

2. **BPM estimation**
   - Uses `librosa.beat.beat_track`.
   - Post‑processing to reduce octave errors (×2 / ÷2).

3. **Chord recognition**
   - CQT‑based chroma.
   - Beat‑synchronous chroma averaging.
   - Cosine similarity vs 24 templates (12 major + 12 minor triads).

### Output
Analysis results are saved to:

```text
data/json/meta.json
```

Example format:

```json
{
  "BPM": 120,
  "CHORDS": [
    [0.0, "Am"],
    [1.0, "F"],
    [2.0, "C"],
    [3.0, "G"]
  ],
  "GENRE": "rock"
}
```

---

## 5. MIDI → JSONL Loop Dataset (Done)

To train a loop generator, we convert curated MIDI files (with drums/bass/harmony tracks) into a **JSONL** format compatible with instruction‑tuning style training.

### What the preprocessing guarantees
- **Timebase normalization**: every MIDI is rescaled to a common `ticks_per_beat` (TPB).
- **Quantization**: events are snapped to a fixed grid (e.g., **1/16**) via `steps_per_beat`.
- **4/4 filtering**:
  - If the file has time signatures and any is not **4/4**, the sample is **discarded**.
  - If the file has no time signatures, it is treated as **4/4**.
- **Robust event encoding**:
  - Each note becomes a token containing start/end step, pitch, velocity.
  - Tracks are serialized role‑by‑role in a stable order.

### Variable loop length control
Each training example includes a **length token** in the user prompt:

```text
<LEN_128>
Generate one loop. Output only tokens.
```

The target is wrapped as:

```text
<LOOP>
... tokens ...
</LOOP>
```

The length (`LEN_N`) is derived from:
- `loop_steps` in the per‑song manifest entry (if present), otherwise
- a global default from `preprocess.yaml` (if set), otherwise
- a fallback computed from `steps_per_beat` (e.g., 4 beats).

### Outputs
The preprocessing produces:
- `processed.jsonl` — valid training rows
- `errors.jsonl` — exceptions with song id + path
- `discarded.jsonl` — discarded samples with reason (e.g., not_4_4)
- `summary.json` — totals and reason counts

---

## 6. Usage

### Run audio analysis

```bash
python code/python/feature_extraction.py
```

This:
- Loads the classifier (if available)
- Computes the mel spectrogram
- Predicts genre, estimates BPM, extracts chords
- Saves the conditioning JSON

### Run MIDI → JSONL preprocessing

Run the manifest→JSONL script after setting the paths in `preprocess.yaml`.

---

## Author & Supervision

**Author:** Andrea Maggiore  
**Thesis Supervisor:** Prof. Marco Raoul Marini  
**Institution:** Sapienza University of Rome  
**Academic Year:** 2025/26