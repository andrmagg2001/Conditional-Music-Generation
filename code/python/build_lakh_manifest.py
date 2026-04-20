import os
import json
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import miditoolkit
import warnings
warnings.filterwarnings("ignore")


LAKH_DIR = "data/dataset/lakh/lmd_full"
OUTPUT_MANIFEST = "data/json/manifest_lakh.jsonl"
VALID_BASS_PROGRAMS = set(range(32, 40))

def analyze_midi_file(midi_path):
    """
    Analize a single MIDI file and return a dictionary with the song information.
    """
    try:
        midi = miditoolkit.midi.parser.MidiFile(midi_path)
    except Exception:
        return None

    drum_idx = -1
    bass_idx = -1
    harm_idx = -1
    
    max_drum_notes = 0
    max_bass_notes = 0
    max_harm_notes = 0

    for idx, inst in enumerate(midi.instruments):
        num_notes = len(inst.notes)
        if num_notes == 0:
            continue
            
        if inst.is_drum:
            if num_notes > max_drum_notes:
                drum_idx = idx
                max_drum_notes = num_notes
        else:
            if inst.program in VALID_BASS_PROGRAMS:
                if num_notes > max_bass_notes:
                    bass_idx = idx
                    max_bass_notes = num_notes
            else:
                if num_notes > max_harm_notes:
                    harm_idx = idx
                    max_harm_notes = num_notes


    if drum_idx == -1 or bass_idx == -1 or harm_idx == -1:
        return None

    try:
        bpm = midi.tempo_changes[0].tempo if len(midi.tempo_changes) > 0 else 120.0
    except Exception:
        bpm = 120.0

    song_id = os.path.basename(midi_path).replace(".mid", "")
    
    rel_path = f"../../{midi_path}"
    
    return {
        "song_id": song_id,
        "midi_path": rel_path,
        "bpm": bpm,
        "drum_idx": drum_idx,
        "bass_idx": bass_idx,
        "harm_idx": harm_idx
    }

def main():
    print("Loading", LAKH_DIR, "...")
    midi_files = glob.glob(os.path.join(LAKH_DIR, "**", "*.mid"), recursive=True)
    print(f"Found {len(midi_files)} MIDI files. Starting analysis...")

    valid_songs = []
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(analyze_midi_file, midi_files), total=len(midi_files)))
        
    for res in results:
        if res is not None:
            valid_songs.append(res)
            
    print(f"\nAnalysis completed! Valid MIDI files found: {len(valid_songs)}")
    
    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        for song in valid_songs:
            f.write(json.dumps(song) + "\n")
            
    print(f"Manifest Lakh saved in: {OUTPUT_MANIFEST}")

if __name__ == "__main__":
    main()
