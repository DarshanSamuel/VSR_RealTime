import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import csv
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import numpy as np

from src.vsr.dataset import VSRDataset
from src.vsr.model import RecurrentVSR
from src.vsr.losses import VSRLoss
from src.vsr.utils import load_config
from src.vsr.fusion_classical import decimate

def torch_warp(image, flow):
    """
    Differentiable image warping in PyTorch using grid_sample.
    image: (B, C, H, W)
    flow: (B, 2, H, W) - displacement mapping current to previous
    """
    B, C, H, W = image.size()
    # Create normalized grid [-1, 1]
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(image.device)
    
    # Add flow to grid
    vgrid = grid + flow
    
    # Normalize to [-1, 1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    
    vgrid = vgrid.permute(0, 2, 3, 1) # (B, H, W, 2)
    output = torch.nn.functional.grid_sample(image, vgrid, mode='bilinear', padding_mode='border', align_corners=True)
    return output

def compute_confidence_map(y_t, warped_x_prev, scale, tau=12.0/255.0):
    """Compute confidence map in LR space."""
    # Decimate warped HR back to LR
    B, C, H, W = warped_x_prev.size()
    # Simple average pooling for decimation
    d_w_x_prev = torch.nn.functional.avg_pool2d(warped_x_prev, kernel_size=scale, stride=scale)
    
    # Error in LR space
    e_t = torch.abs(d_w_x_prev - y_t)
    
    # Blur error (using a simple fixed 3x3 conv or avg pool as approximation)
    e_t = torch.nn.functional.avg_pool2d(e_t, kernel_size=3, stride=1, padding=1)
    
    # Confidence
    a_t = 0.8 * torch.exp(- (e_t / tau)**2)
    return a_t

def train(config_path, data_dir, output_dir, stage='B'):
    config = load_config(config_path)
    device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training on {device}, Stage {stage}")
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect sequence length by counting hr_*.png files in the first clip
    data_path = Path(data_dir)
    first_clip = next(data_path.iterdir())
    detected_length = len(list(first_clip.glob("hr_*.png")))
    seq_length = detected_length if stage == 'B' else 1
    
    print(f"Auto-detected sequence length: {detected_length} frames (Using {seq_length} for Stage {stage})")
    
    dataset = VSRDataset(data_dir, sequence_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    
    model = RecurrentVSR(scale=config.scale, channels=config.channels, num_blocks=config.blocks).to(device)
    criterion = VSRLoss(lambda_temporal=config.lambda_temporal).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    log_file = open(out_dir / f"train_log_stage_{stage}.csv", "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["epoch", "loss", "loss_spatial", "loss_temporal"])
    
    losses = []
    
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_loss_spatial = 0
        epoch_loss_temporal = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.epochs}")
        for batch in pbar:
            hr_seq = batch['hr'].to(device)
            lr_seq = batch['lr'].to(device)
            flow_seq = batch['flow'].to(device) # Flow from curr to prev in LR space
            
            B, T, C, H, W = lr_seq.size()
            
            optimizer.zero_grad()
            batch_loss = 0
            
            # Recurrent state
            x_hat_prev = None
            
            for t in range(T):
                lr_t = lr_seq[:, t]
                hr_t = hr_seq[:, t]
                flow_lr_t = flow_seq[:, t]
                
                # Drift control or Stage A (simulate no history)
                if stage == 'A' or x_hat_prev is None or random.random() < 0.15:
                    # No history: use bicubic upsampled current frame
                    warped_x_prev = torch.nn.functional.interpolate(lr_t, scale_factor=config.scale, mode='bicubic', align_corners=False)
                    conf_map_lr = torch.ones_like(lr_t)
                else:
                    # Upscale flow to HR
                    flow_hr_t = torch.nn.functional.interpolate(flow_lr_t, scale_factor=config.scale, mode='bilinear', align_corners=True) * config.scale
                    
                    # Warp previous output
                    warped_x_prev = torch_warp(x_hat_prev, flow_hr_t)
                    
                    # Confidence map in LR
                    conf_map_lr = compute_confidence_map(lr_t, warped_x_prev, config.scale)
                
                # Forward pass
                x_hat_t = model(lr_t, warped_x_prev, conf_map_lr)
                
                if stage == 'A' or x_hat_prev is None:
                    loss, l_sp, l_temp = criterion(x_hat_t, hr_t)
                else:
                    # Confidence map in HR space for loss masking
                    conf_map_hr = torch.nn.functional.interpolate(conf_map_lr, scale_factor=config.scale, mode='nearest')
                    loss, l_sp, l_temp = criterion(x_hat_t, hr_t, warped_x_prev, conf_map_hr)
                    
                batch_loss = batch_loss + loss
                epoch_loss_spatial += l_sp.item()
                epoch_loss_temporal += l_temp.item()
                
                # Update recurrent state
                x_hat_prev = x_hat_t
                
            # Truncated backprop for the whole sequence
            batch_loss = batch_loss / T
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            pbar.set_postfix({'loss': batch_loss.item()})
            
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        avg_sp = epoch_loss_spatial / (len(dataloader) * T)
        avg_temp = epoch_loss_temporal / (len(dataloader) * T)
        
        csv_writer.writerow([epoch, avg_loss, avg_sp, avg_temp])
        log_file.flush()
        losses.append(avg_loss)
        
        # Save checkpoint
        if epoch % 10 == 0 or epoch == config.epochs:
            torch.save(model.state_dict(), out_dir / f"model_stage_{stage}_epoch_{epoch}.pth")
            
    log_file.close()
    
    # Plot losses
    plt.figure()
    plt.plot(range(1, config.epochs + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Stage {stage}')
    plt.savefig(out_dir / f"loss_plot_stage_{stage}.png")
    plt.close()
    print("Training complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", required=True, help="Dataset directory")
    parser.add_argument("--out", default="outputs/checkpoints")
    parser.add_argument("--stage", choices=['A', 'B'], default='A')
    args = parser.parse_args()
    
    train(args.config, args.data, args.out, args.stage)
