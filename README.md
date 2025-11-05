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


---

## Author & Supervision

**Author:** Andrea Maggiore  
**Thesis Supervisor:** Prof. Marco Raoul Marini  
**Department:** Computer Vision Laboratory  
**Institution:** Sapienza University of Rome  
**Academic Year:** 2025/26