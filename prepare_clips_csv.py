import os
import pandas as pd
import csv

# Input CSV and output directory for clips
# INPUT_CSV = "train_dash_0.csv"
# OUTPUT_CLIPS_DIR = "output_clips"
# OUTPUT_CSV = "train_dash_clips.csv"
INPUT_CSV = "sampled_seconds_clips.csv"
OUTPUT_CLIPS_DIR = "sampled_frames"
OUTPUT_CSV = "sampled_seconds_clips.csv"

# Adjust these column names if your CSV uses different ones
VIDEO_COL = 'video_path'
START_COL = 'time_start'
END_COL = 'time_end'
LABEL_COL = 'label'  # Change if your label column is named differently

# Load class definitions
CLASS_DEF_PATH = "Distracted_Activity_Class_definition.txt"
class_map = {}
with open(CLASS_DEF_PATH, encoding="utf-8") as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)  # skip header
    for row in reader:
        if len(row) >= 2:
            class_map[str(row[0]).strip()] = row[1].strip()

df = pd.read_csv(INPUT_CSV)

rows = []
for idx, row in df.iterrows():
    video_file = row[VIDEO_COL]
    # Check if the clip_path already exists
    if not os.path.exists(video_file):
        print(f"Clip path does not exist: {video_file}, skipping.")
        continue
    label = str(row[LABEL_COL]) if LABEL_COL in row else ''
    class_name = class_map.get(label, '')
    rows.append({
        'video_path': video_file,
        'label': label,
        'class_name': class_name
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV} with class names.")
