import json
import librosa
import torch
import numpy as np
from pathlib import Path

NOTES       = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']



def build_chord_templates():
    """
    Build normalized chroma templates for all 12 major and 12 minor triads.

    This function creates simple binary chroma templates for:
      - C major triad:   [C, E, G]
      - C minor triad:   [C, Eb, G]

    and then circularly shifts (rolls) these base templates across the
    12 chroma positions to obtain templates for every possible root:
      C, C#, D, ..., B (both major and minor).

    Each template is L2-normalized so it can be compared directly with
    chroma vectors using a dot product (cosine similarity).

    Returns
    -------
    templates : np.ndarray, shape (24, 12)
        Array of chord chroma templates. The 24 rows correspond to:
        [C, Cm, C#, C#m, ..., B, Bm], each row being a 12-D chroma vector.

    names : list of str, length 24
        Chord names in the same order as `templates`. Major chords are
        named as "C", "C#", ..., "B"; minor chords as "Cm", "C#m", ..., "Bm".
    """
    
    maj_note = np.array([1,0,0,0,1,0,0,1,0,0,0,0], dtype=float)
    min_note = np.array([1,0,0,1,0,0,0,1,0,0,0,0], dtype=float)

    templates = []
    names     = []

    for root in range(12):
        maj_root = np.roll(maj_note, root)
        min_root = np.roll(min_note, root)

        templates.append(maj_root)
        names.append(f"{NOTES[root]}")

        templates.append(min_root)
        names.append(f"{NOTES[root]}")

    templates = np.stack(templates, axis=0)
    templates /= np.linalg.norm(templates, axis=1, keepdims=True) + 1e-8
    return templates, names


CHORD_TEMPLATES, CHORD_NAMES = build_chord_templates()



def estimate_chords(y, sr=48000, hop_length=512):
    """
    Estimate a beat-synchronous chord sequence from an audio track.

    This function:
      - Loads the input audio (if a filepath is provided).
      - Computes a chroma representation over time with `librosa.feature.chroma_cqt`.
      - Runs beat tracking (`librosa.beat.beat_track`) to get beat times in seconds.
      - For each beat interval [t_i, t_{i+1}):
        * Averages the chroma frames in that interval.
        * Normalizes the resulting 12-D chroma vector.
        * Computes similarity against pre-defined major/minor chord templates
          (`CHORD_TEMPLATES`, `CHORD_NAMES`).
        * Picks the chord with the highest similarity.

    Parameters
    ----------
    y : str or np.ndarray
        If str: path to an audio file readable by `librosa.load`.
        If np.ndarray: mono waveform array of shape (n_samples,).
    sr : int, optional
        Target sampling rate. If `y` is a filepath, the audio will be
        resampled to this rate. Default is 48000.
    hop_length : int, optional
        Hop length used for both chroma computation and beat tracking.
        Default is 512.

    Returns
    -------
    bpm : int
        Global tempo estimate in beats per minute (BPM), rounded to integer.
    chord_seq : list of tuple
        Sequence of `(t0, chord_name)`:
          - `t0` : float
              Beat start time in seconds.
          - `chord_name` : str
              Estimated chord label (e.g., "C", "Am", "F#", "G#m"),
              taken from `CHORD_NAMES`.

    Notes
    -----
    Assumes that:
      - `CHORD_TEMPLATES` has shape [24, 12] and contains normalized
        major/minor templates.
      - `CHORD_NAMES` is a list of 24 chord labels aligned with the rows
        of `CHORD_TEMPLATES`.
    """
    if isinstance(y, (str, Path)):
        y, sr = librosa.load(y, sr=sr, mono=True)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    bpm, beats = librosa.beat.beat_track(
        y=y,
        sr=sr,
        hop_length=hop_length,
        units="time"
    )
    bpm = int(np.round(bpm).item())
    chord_seq = []

    for i in range(len(beats) - 1):
        t0, t1 = round(beats[i],3), round(beats[i + 1],3)
        f0 = librosa.time_to_frames(t0, sr=sr, hop_length=hop_length)
        f1 = librosa.time_to_frames(t1, sr=sr, hop_length=hop_length)

        if f1 <= f0:
            continue

        seg = chroma[:, f0:f1]
        if seg.shape[1] == 0:
            continue

        chroma_mean = seg.mean(axis=1)
        chroma_norm = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-8)

        sims = CHORD_TEMPLATES @ chroma_norm
        idx = np.argmax(sims)
        chord_name = CHORD_NAMES[idx]
        chord_seq.append((t0, chord_name))

    return bpm, chord_seq

print(estimate_chords("Test.wav"))