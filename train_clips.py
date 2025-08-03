import os
import csv
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor
import models.config as config
import torch.optim as optim
from tqdm import tqdm
import torch

class ClipsVQADataset(Dataset):
    def __init__(self, csv_path, image_processor, frames_per_clip=10):
        self.samples = []
        self.image_processor = image_processor
        self.frames_per_clip = frames_per_clip
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

    def __len__(self):
        return len(self.samples)

    def extract_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(total_frames // self.frames_per_clip, 1)
        for i in range(self.frames_per_clip):
            frame_idx = i * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = self.image_processor(img)
            frames.append(img)
        cap.release()
        if len(frames) == 0:
            img_shape = (3, 224, 224)
            return torch.zeros(img_shape)
        while len(frames) < self.frames_per_clip:
            try:
                frames.append(torch.zeros_like(frames[0]))
            except Exception as e:
                print(f"Error occurred while padding frames: {video_path}, {e}")
        # Average frames to get a single image tensor [3, H, W]
        return torch.mean(torch.stack(frames), dim=0)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = os.path.join('/home/iadel/data_A1', sample['video_path'])
        image = self.extract_frames(video_path)
        question = "What is the activity in this clip?"
        answer = sample['class_name']
        return {
            'images': image,
            'texts': {
                'user': question,
                'assistant': answer
            }
        }

from data.collators import VQACollator

def train(args):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    vlm_cfg = config.VLMConfig()
    vlm_cfg.vit_hidden_dim = 128
    vlm_cfg.vit_n_blocks = 2
    vlm_cfg.vit_n_heads = 4
    vlm_cfg.vit_img_size = 224

    vlm_cfg.lm_hidden_dim = 64
    vlm_cfg.lm_n_blocks = 2
    vlm_cfg.lm_n_heads = 2

    image_processor = get_image_processor(vlm_cfg.vit_img_size)
    dataset = ClipsVQADataset(args.csv_path, image_processor, frames_per_clip=args.frames_per_clip)
    # Limit to first 100 samples
    from torch.utils.data import Subset
    # subset = Subset(dataset, range(10))
    vqa_collator = VQACollator(get_tokenizer(vlm_cfg.lm_tokenizer), vlm_cfg.lm_max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=vqa_collator)
    model = VisionLanguageModel(vlm_cfg).to(device)
    # model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    import matplotlib.pyplot as plt
    epoch_losses = []
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in tqdm(dataloader):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
        torch.cuda.empty_cache()
    # Plot training loss over epochs
    plt.figure()
    plt.plot(range(1, args.epochs+1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.savefig('training_loss_curve.png')
    plt.show()
    # torch.save(model.state_dict(), args.output)
    model.save_pretrained(args.output)
    print(f"Model saved to {args.output}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train nanoVLM on sampled_seconds_clips.csv")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to sampled_seconds_clips.csv")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--frames_per_clip", type=int, default=10, help="Number of frames per clip")
    parser.add_argument("--output", type=str, default="nanoVLM_clips", help="Output model file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
