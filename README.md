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