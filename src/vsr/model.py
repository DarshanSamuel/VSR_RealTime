import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res

class RecurrentVSR(nn.Module):
    def __init__(self, scale=2, channels=16, num_blocks=2):
        super().__init__()
        self.scale = scale
        
        # Input channels: LR (1) + space-to-depth HR history (scale^2) + confidence map (1)
        in_channels = 1 + scale**2 + 1
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
        
        self.tail = nn.Sequential(
            nn.Conv2d(channels, scale**2, 3, padding=1),
            nn.PixelShuffle(scale)
        )
        
        self.unshuffle = nn.PixelUnshuffle(scale)
        
    def forward(self, lr, warped_hr_prev, conf_map):
        """
        lr: (B, 1, H, W)
        warped_hr_prev: (B, 1, H*s, W*s)
        conf_map: (B, 1, H, W) - confidence map in LR space
        """
        # Space-to-depth on the warped HR history
        history_lr = self.unshuffle(warped_hr_prev) # (B, s^2, H, W)
        
        # Concatenate inputs
        x = torch.cat([lr, history_lr, conf_map], dim=1) # (B, 1 + s^2 + 1, H, W)
        
        # Network forward pass
        features = self.head(x)
        features = self.body(features)
        residual = self.tail(features) # (B, 1, H*s, W*s)
        
        # Global skip connection (bicubic upsample)
        # Note: align_corners=False is usually standard for SR
        base = F.interpolate(lr, scale_factor=self.scale, mode='bicubic', align_corners=False)
        
        # Final output
        out = base + residual
        return out
