# Condition Music Generation from Guitar Performances using Deep Neural Networks

> **Goal:** Given a guitar performance (audio), extract musical descriptors —  
> tempo, beat/downbeat, chords, key — and generate coherent multi-track accompaniments  
> (bass, drums, keys) using deep generative models.

The current stage focuses on **dataset preparation** for the genre classification module.

- The **GTZAN Genre Collection** dataset was downloaded from Kaggle:  
  [GTZAN Dataset](https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection) with over 1000 samples audio with different genre.

- Each `.au` audio file from the dataset has been processed to generate a corresponding  
  **high-resolution Mel-spectrogram** stored as a `.npy` file (same filename and folder structure).

- All Mel-spectrograms are computed with **torchaudio** using GPU/MPS acceleration  
  for faster STFT computation, and cached in the directory `classifier_mel/`.

## Genre Classification

On top of the cached Mel-spectrograms, the project includes a **CNN-based genre classifier** (`ResMelNet`) that operates directly on log-Mel patches.

- Input: high-resolution log-Mel spectrograms (1024 mel bands, 44.1 kHz).
- Model: residual 2D CNN with Squeeze-and-Excitation (SE) blocks and a small MLP head.
- Training:
  - Optimizer: AdamW with weight decay.
  - Loss: Cross-Entropy with label smoothing.
  - Scheduler: CosineAnnealingWarmRestarts.
  - Supports CUDA (NVIDIA), MPS (Apple Silicon) and CPU, with AMP on CUDA.
- Output: a 10-way softmax over GTZAN genres, with validation accuracy around **70–75%** in the current configuration.

This classifier will be used as one of the conditioning signals for the **future accompaniment generation** stage (together with tempo, beat, chords and key).

---

## Author & Supervision

**Author:** Andrea Maggiore  
**Thesis Supervisor:** Prof. Marco Raoul Marini  
**Department:** Computer Vision Laboratory  
**Institution:** Sapienza University of Rome  
**Academic Year:** 2025/26