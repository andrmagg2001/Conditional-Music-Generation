import librosa
from header import *

NOTES       = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

mel_tf, db_tf = build_mel_frontend(DEVICE)
_model: ResMelNet | None = None


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


def classify_audio_file(path: str | Path) -> str:
    """
    Classify a single audio file into one of the macro-genre classes.

    This function:
      - Lazily loads the trained `ResMelNet` model checkpoint (if not already loaded).
      - Loads and mono-resamples the input audio.
      - Computes a mel-spectrogram + dB conversion using the same frontend
        used during training.
      - Normalizes, centers and crops a fixed-length time window.
      - Optionally downsamples the time/frequency resolution.
      - Runs the classifier and returns the most likely macro-genre label.

    Parameters
    ----------
    path : str or Path
        Path to the input audio file (e.g., "Test.wav") to classify.

    Returns
    -------
    str
        Predicted macro-genre name, taken from `CLASSES[class_idx]`.
    """
    global _model

    if _model is None:
        ckpt_path = Path(CHECKPOINTS) / "guitar_macro_classifier.pth"
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model = ResMelNet(n_classes=len(CLASSES)).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()
        _model = model

    wav = load_mono_resampled(path)

    with torch.inference_mode():
        S = mel_tf(wav.to(DEVICE))
        S_db = db_tf(S)[0]

    X = S_db.cpu().numpy().astype(np.float32)
    X = np.nan_to_num(X, nan=DB_LO, neginf=DB_LO, posinf=DB_HI)
    X = norm_db(X)
    X = np.nan_to_num(X, nan=0.0, neginf=0.0, posinf=1.0)

    M, T = X.shape
    s = max(0, T // 2 - CROP_F // 2)
    e = min(T, T // 2 + CROP_F // 2)
    X = X[:, s:e]

    if X.shape[1] < CROP_F:
        pad = CROP_F - X.shape[1]
        X = np.pad(X, ((0, 0), (0, pad)), mode="edge")

    X_t = torch.from_numpy(X).unsqueeze(0).unsqueeze(0).to(torch.float32)

    if FREQ_D > 1 or TIME_D > 1:
        X_t = torch.nn.functional.avg_pool2d(
            X_t,
            kernel_size=(FREQ_D, TIME_D),
            stride=(FREQ_D, TIME_D),
        )

    X_t = X_t.to(DEVICE, non_blocking=True)

    with torch.no_grad():
        logits = _model(X_t)
        logits = torch.nan_to_num(logits, nan=0.0, neginf=-1e4, posinf=1e4)
        probs = torch.softmax(logits, dim=1)[0]

    probs_np = probs.cpu().numpy()
    class_idx = int(probs_np.argmax())
    class_name = CLASSES[class_idx]

    return class_name


bpm, chords = estimate_chords("Test.wav")
genre = classify_audio_file("Test.wav")

with open("data/json/meta.json", "w") as f:
    json.dump({"BPM": bpm, "CHORDS" : chords, "GENRE" : genre }, f)