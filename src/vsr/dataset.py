import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
import json

from src.vsr.degrade import DegradationPipeline
from src.vsr.flow import get_flow_estimator

class VSRDataset(Dataset):
    def __init__(self, data_dir: str, sequence_length: int = 5, mode: str = 'train'):
        """
        Loads pre-processed clip sequences (HR, LR, Flow).
        Expected directory structure:
        data_dir/
            clip_0001/
                hr_0.png, hr_1.png...
                lr_0.png, lr_1.png...
                flow_1.npy, flow_2.npy... (flow from t to t-1)
            ...
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        
        # Gather all clips
        self.clips = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
    def __len__(self):
        return len(self.clips)
        
    def __getitem__(self, idx):
        clip_path = self.clips[idx]
        
        hr_seq = []
        lr_seq = []
        flow_seq = [] # flow_seq[0] is dummy (no previous frame)
        
        for t in range(self.sequence_length):
            # Load HR and LR
            hr = cv2.imread(str(clip_path / f"hr_{t}.png"), cv2.IMREAD_GRAYSCALE)
            lr = cv2.imread(str(clip_path / f"lr_{t}.png"), cv2.IMREAD_GRAYSCALE)
            
            hr = hr.astype(np.float32) / 255.0
            lr = lr.astype(np.float32) / 255.0
            
            hr_seq.append(torch.from_numpy(hr).unsqueeze(0))
            lr_seq.append(torch.from_numpy(lr).unsqueeze(0))
            
            if t == 0:
                # Dummy flow for t=0
                flow = np.zeros((lr.shape[0], lr.shape[1], 2), dtype=np.float32)
            else:
                flow = np.load(str(clip_path / f"flow_{t}.npy"))
                
            # Flow needs to be (2, H, W)
            flow = flow.transpose(2, 0, 1)
            flow_seq.append(torch.from_numpy(flow))
            
        return {
            'hr': torch.stack(hr_seq),       # (T, 1, H*s, W*s)
            'lr': torch.stack(lr_seq),       # (T, 1, H, W)
            'flow': torch.stack(flow_seq)    # (T, 2, H, W)
        }

def build_dataset_from_video(video_path: str, output_dir: str, config, clip_length: int = 7):
    """
    Cuts a video into clips, creates LR versions, computes flow, and saves to disk.
    """
    print(f"Building dataset from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    degrader = DegradationPipeline(scale=config.scale, config=config)
    flow_estimator = get_flow_estimator(config.motion_estimator)
    
    clip_idx = 0
    frame_buffer = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to Y channel only for training
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_channel, _, _ = cv2.split(ycrcb)
        
        # Resize to an even multiple of scale to avoid dimension issues
        h, w = y_channel.shape
        new_h, new_w = h - (h % config.scale), w - (w % config.scale)
        if new_h != h or new_w != w:
            y_channel = y_channel[:new_h, :new_w]
            
        frame_buffer.append(y_channel)
        
        if len(frame_buffer) == clip_length:
            clip_path = out_dir / f"clip_{clip_idx:04d}"
            clip_path.mkdir(exist_ok=True)
            
            prev_lr = None
            
            for t in range(clip_length):
                hr = frame_buffer[t]
                lr = degrader.apply(hr)
                
                cv2.imwrite(str(clip_path / f"hr_{t}.png"), hr)
                cv2.imwrite(str(clip_path / f"lr_{t}.png"), lr)
                
                if t > 0:
                    flow_lr = flow_estimator.estimate(lr, prev_lr)
                    np.save(str(clip_path / f"flow_{t}.npy"), flow_lr)
                    
                prev_lr = lr.copy()
                
            frame_buffer = [] # non-overlapping clips
            clip_idx += 1
            if clip_idx % 10 == 0:
                print(f"Processed {clip_idx} clips")
                
    cap.release()
    print(f"Done. Saved {clip_idx} clips to {output_dir}")

def build_dataset_from_vimeo90k(vimeo_dir: str, output_dir: str, config, split: str = 'train', sequence_length: int = 7):
    """
    Converts the Vimeo-90K dataset into our preprocessed format.
    Expects vimeo_dir to contain 'sequences' and the corresponding list files.
    """
    vimeo_dir = Path(vimeo_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect list file prefix (tri_ for triplet, sep_ for septuplet)
    prefix = "tri_" if sequence_length == 3 else "sep_"
    list_file = vimeo_dir / f"{prefix}{split}list.txt"
    
    # Fallback in case they renamed it or only have one type
    if not list_file.exists():
        fallback_file = vimeo_dir / f"sep_{split}list.txt" if sequence_length == 3 else vimeo_dir / f"tri_{split}list.txt"
        if fallback_file.exists():
            list_file = fallback_file
            print(f"Warning: using {list_file.name} instead.")
        else:
            raise FileNotFoundError(f"List file {list_file} not found in {vimeo_dir}")
        
    with open(list_file, 'r') as f:
        seq_paths = [line.strip() for line in f.readlines() if line.strip()]
        
    degrader = DegradationPipeline(scale=config.scale, config=config)
    flow_estimator = get_flow_estimator(config.motion_estimator)
    
    clip_idx = 0
    print(f"Processing {len(seq_paths)} sequences from Vimeo-90K ({split}, {sequence_length} frames)...")
    
    for seq_path in tqdm(seq_paths):
        # seq_path is like '00001/0001'
        seq_dir = vimeo_dir / "sequences" / seq_path
        if not seq_dir.exists():
            continue
            
        clip_out_path = out_dir / f"vimeo_{seq_path.replace('/', '_')}"
        clip_out_path.mkdir(exist_ok=True)
        
        prev_lr = None
        
        for t in range(sequence_length):
            frame_path = seq_dir / f"im{t+1}.png"
            hr_bgr = cv2.imread(str(frame_path))
            if hr_bgr is None:
                # If some frames are missing, just break early
                break
                
            # Convert to Y channel
            ycrcb = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2YCrCb)
            hr = ycrcb[:, :, 0]
            
            # Crop to multiple of scale
            h, w = hr.shape
            new_h, new_w = h - (h % config.scale), w - (w % config.scale)
            if new_h != h or new_w != w:
                hr = hr[:new_h, :new_w]
                
            lr = degrader.apply(hr)
            
            cv2.imwrite(str(clip_out_path / f"hr_{t}.png"), hr)
            cv2.imwrite(str(clip_out_path / f"lr_{t}.png"), lr)
            
            if t > 0:
                flow_lr = flow_estimator.estimate(lr, prev_lr)
                np.save(str(clip_out_path / f"flow_{t}.npy"), flow_lr)
                
            prev_lr = lr.copy()
            
        clip_idx += 1
        
    print(f"Done. Saved {clip_idx} Vimeo-90K clips to {output_dir}")

if __name__ == "__main__":
    import argparse
    from src.vsr.utils import Config
    parser = argparse.ArgumentParser(description="Preprocess video or Vimeo-90K to training format.")
    parser.add_argument("--video", type=str, help="Path to local HD video to slice into clips")
    parser.add_argument("--vimeo", type=str, help="Path to Vimeo-90K root directory")
    parser.add_argument("--out", type=str, required=True, help="Output directory for processed dataset")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Vimeo-90K split to process")
    parser.add_argument("--frames", type=int, default=7, help="Number of frames per sequence (7 for septuplet, 3 for triplet)")
    args = parser.parse_args()
    
    config = Config()
    if args.video:
        build_dataset_from_video(args.video, args.out, config, clip_length=args.frames)
    elif args.vimeo:
        build_dataset_from_vimeo90k(args.vimeo, args.out, config, split=args.split, sequence_length=args.frames)
    else:
        print("Must provide either --video or --vimeo")
