import os
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip

CSV_PATH = "train_dash_0.csv"
OUTPUT_DIR = "output_clips"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Adjust these column names if your CSV uses different ones
VIDEO_COL = 'video_path'
START_COL = 'time_start'
END_COL = 'time_end'

not_found_videos = []
for idx, row in df.iterrows():
    video_file = row[VIDEO_COL]
    input_path = video_file

    # Get the folder name of the video file (e.g., '8' from '8/Dashboard_user_id_13522_NoAudio_5_11_31.MP4')
    video_folder = str(video_file.split('/')[-2])  # Assuming the folder is the first part of the path
    output_subdir = os.path.join(OUTPUT_DIR, video_folder)
    os.makedirs(output_subdir, exist_ok=True)
    output_filename = f"{os.path.splitext(os.path.basename(video_file))[0]}_clip_{idx}.mp4"
    output_path = os.path.join(output_subdir, output_filename)

    if not os.path.exists(input_path):
        print(f"Video not found: {input_path}")
        not_found_videos.append(input_path)
        continue

    try:
        clip = VideoFileClip(input_path)
        duration = clip.duration
        print(f"Loaded video: {input_path}, duration: {duration}s")
        # Determine start and end times for middle 5 seconds (or whole video if < 5s)
        if duration <= 5:
            start_time = 0
            end_time = duration
        else:
            mid_point = duration / 2
            start_time = max(0, mid_point - 2.5)
            end_time = min(duration, mid_point + 2.5)
        new_clip = clip.subclipped(start_time, end_time)
        new_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        clip.close()
        new_clip.close()
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")


if not_found_videos:
    print(f"{len(not_found_videos)} videos were not found:")
    for video in not_found_videos:
        print(video)
