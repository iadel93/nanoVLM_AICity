import argparse
import os
import torch
from PIL import Image
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

def parse_args():
    parser = argparse.ArgumentParser(description="Classify video frames with nanoVLM")
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory containing video frames")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a local checkpoint")
    parser.add_argument("--hf_model", type=str, default="lusxvr/nanoVLM-222M", help="HuggingFace repo ID")
    parser.add_argument("--prompt", type=str, default="is driver distracted?", help="Prompt for classification")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Max tokens per output")
    return parser.parse_args()

def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")


    if args.checkpoint and args.checkpoint.endswith('.pt'):
        print(f"Loading PyTorch checkpoint from: {args.checkpoint}")
        from models.config import VLMConfig
        vlm_cfg = VLMConfig()
        model = VisionLanguageModel(vlm_cfg).to(device)
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
    else:
        source = args.checkpoint if args.checkpoint else args.hf_model
        print(f"Loading weights from: {source}")
        model = VisionLanguageModel.from_pretrained(source).to(device)
        model.eval()

    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)

    # check if directory exists
    if not os.path.isdir(args.frames_dir):
        video_files = [args.frames_dir]
    else:
        video_files = [f for f in os.listdir(args.frames_dir) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        video_files.sort()

    def extract_and_average_frames(video_path, image_processor, frames_per_video=10):
        import cv2
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(total_frames // frames_per_video, 1)
        for i in range(frames_per_video):
            frame_idx = i * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = image_processor(img)
            frames.append(img)
        cap.release()
        if len(frames) == 0:
            img_shape = (3, model.cfg.vit_img_size, model.cfg.vit_img_size)
            return torch.zeros(img_shape)
        while len(frames) < frames_per_video:
            frames.append(torch.zeros_like(frames[0]))
        # Average frames to get a single image tensor [3, H, W]
        return torch.mean(torch.stack(frames), dim=0)

    for video_file in video_files:
        video_path = os.path.join(args.frames_dir, video_file)
        avg_img = extract_and_average_frames(video_path, image_processor, frames_per_video=10).unsqueeze(0).to(device)

        template = f"Question: {args.prompt} Answer:"
        encoded = tokenizer.batch_encode_plus([template], return_tensors="pt")
        tokens = encoded["input_ids"].to(device)

        gen = model.generate(tokens, avg_img, max_new_tokens=args.max_new_tokens)
        out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        print(f"{video_file}: {out}")

if __name__ == "__main__":
    main()
