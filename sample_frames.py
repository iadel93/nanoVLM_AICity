import os
import subprocess
import pandas as pd
import csv

CLIPS_CSV = "train_dash_clips.csv"
FRAMES_DIR = "sampled_frames"
FPS = 10  # Change to 5 for lower rate
KEYFRAMES_PER_SEC = 1
UNIFORM_FRAMES_PER_SEC = 4

os.makedirs(FRAMES_DIR, exist_ok=True)

df = pd.read_csv(CLIPS_CSV)

sampled_video = []

for idx, row in df.iterrows():
    print(f"Processing clip {idx+1}/{len(df)}: {row['video_path']}")
    clip_path = row['video_path']
    video_name = clip_path.split('/')[-1].replace('.mp4', '')
    base_name = clip_path.split('/')[-2]
    out_dir = os.path.join(FRAMES_DIR, base_name)
    os.makedirs(out_dir, exist_ok=True)

    import json
    import shlex
    probe_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=duration -of json {shlex.quote(clip_path)}"
    probe_result = subprocess.run(probe_cmd, shell=True, capture_output=True, text=True)
    duration = None
    try:
        duration = float(json.loads(probe_result.stdout)['streams'][0]['duration'])
    except Exception:
        print(f"Could not get duration for {clip_path}")
        continue

    # For each second, extract a 1-second video at 10 fps
    extracted_clips = []
    for sec in range(int(duration)):
        start_time = sec
        end_time = min(sec + 1, duration)
        out_clip_name = f"{video_name}_sec{sec:03d}.mp4"
        out_clip_path = os.path.join(out_dir, out_clip_name)
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", clip_path,
            "-ss", str(start_time), "-to", str(end_time),
            "-vf", "fps=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            out_clip_path
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True)
        if os.path.exists(out_clip_path):
            extracted_clips.append({
                'video_path': out_clip_path,
                'start_time': start_time,
                'end_time': end_time,
                'orig_video': clip_path,
                'label': base_name
            })
        else:
            print(f"Failed to extract {out_clip_path}")

    # break  # Remove to process all clips

# Create a new DataFrame for the extracted clips
all_extracted_clips = []
for idx, row in df.iterrows():
    clip_path = row['video_path']
    video_name = clip_path.split('/')[-1].replace('.mp4', '')
    base_name = clip_path.split('/')[-2]
    out_dir = os.path.join(FRAMES_DIR, base_name)
    # For each second, check for extracted clips
    for sec in range(10000):  # Arbitrary large number, will break when not found
        out_clip_name = f"{video_name}_sec{sec:03d}.mp4"
        out_clip_path = os.path.join(out_dir, out_clip_name)
        if os.path.exists(out_clip_path):
            all_extracted_clips.append({
                'video_path': out_clip_path,
                'start_time': sec,
                'end_time': sec+1,
                'orig_video': clip_path,
                'label': base_name
            })
        else:
            break

extracted_df = pd.DataFrame(all_extracted_clips)
extracted_df.to_csv('sampled_seconds_clips.csv', index=False)
print("Saved: sampled_seconds_clips.csv with extracted 1-second clips.")
