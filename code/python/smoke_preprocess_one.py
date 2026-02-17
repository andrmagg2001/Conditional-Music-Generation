import os
import yaml
import miditoolkit

import json as js

YAML_PATH = "code/python/preprocess.yaml"
JSON_MANIFEST   = ""
JSON_OUTPUT_DIR = ""
JSON_LOG_DIR    = ""
QUANT_GRID      = ""
TARGET_TPD      = 0
STEP_PER_BEAT   = 0
STEP_TICKS      = 0



with open(YAML_PATH) as stream:
    
    try:
        temp = yaml.safe_load(stream)
        JSON_MANIFEST   = temp['paths']['manifest_path']
        JSON_OUTPUT_DIR = temp['paths']['output_json_dir']
        JSON_LOG_DIR    = temp['paths']['report_dir']
        TARGET_TPD      = temp['normalize']['target_tpb']
        STEP_PER_BEAT   = temp['quantization']['steps_per_beat']
        QUANT_GRID      = temp['quantization']['grid']
        STEP_TICKS      = TARGET_TPD // STEP_PER_BEAT


    except yaml.YAMLError as exc:
        print(exc)


    with open(JSON_MANIFEST) as manifest:
        try:
            line = js.loads(manifest.readline().strip())
            
        except:
            print("Error in json reading")


manifest_dir = os.path.dirname(JSON_MANIFEST)
midi_abs     = os.path.normpath(os.path.join(manifest_dir, line['midi_path']))

midi    = miditoolkit.MidiFile(midi_abs)
tracks  = midi.instruments

drum_idx = line["drum_idx"]
bass_idx = line["bass_idx"]
harm_idx = line["harm_idx"]

drums_out   = []
bass_out    = []
harm_out    = []

old_tpd     = midi.ticks_per_beat
scale       = TARGET_TPD/old_tpd

drum_notes  = tracks[drum_idx].notes
bass_notes  = tracks[bass_idx].notes
harm_notes  = tracks[harm_idx].notes



for instrument in [drum_notes, bass_notes, harm_notes]:
    for note in instrument:
        note.start  = round(note.start * scale)
        note.end    = round(note.end * scale)

        if note.end <= note.start:
            note.end = note.start + 1

        note.start  = round(note.start / STEP_TICKS) * STEP_TICKS
        note.end    = round(note.end   / STEP_TICKS) * STEP_TICKS

        if note.end <= note.start:
            note.end = note.start + STEP_TICKS

        if instrument == drum_notes:
            drums_out.append([note.start, note.end, note.pitch, note.velocity])

        elif instrument == bass_notes:
            bass_out.append([note.start, note.end, note.pitch, note.velocity])

        elif instrument == harm_notes:
            harm_out.append([note.start, note.end, note.pitch, note.velocity])


midi.ticks_per_beat = TARGET_TPD

print(len(drums_out), len(bass_out), len(harm_out))
if len(drums_out) > 0:
    print(drums_out[0])

row = {
    "song_id": line["song_id"],
    "bpm": line.get("bpm"),
    "tpb": TARGET_TPD,
    "grid": QUANT_GRID,
    "step_ticks": STEP_TICKS,
    "source_midi_path": line.get("midi_path"),
    "roles": {
        "drums": drums_out,
        "bass": bass_out,
        "harmony": harm_out,
    },
}

os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(JSON_OUTPUT_DIR, "debug_one.jsonl")
with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(js.dumps(row, ensure_ascii=False) + "\n")

print(out_path)
