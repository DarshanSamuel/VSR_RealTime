import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        
    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss

class VSRLoss(nn.Module):
    def __init__(self, lambda_temporal=0.1, eps=1e-3):
        super().__init__()
        self.charbonnier = CharbonnierLoss(eps)
        self.lambda_temporal = lambda_temporal
        
    def forward(self, x_hat_t, hr_t, x_hat_prev_warped=None, conf_map_hr=None):
        """
        x_hat_t: Current predicted HR frame
        hr_t: Current ground truth HR frame
        x_hat_prev_warped: Previous predicted HR frame warped to current frame (optional, for temporal loss)
        conf_map_hr: Confidence map in HR space (optional, to mask temporal loss)
        """
        # Spatial reconstruction loss
        loss_spatial = self.charbonnier(x_hat_t, hr_t)
        
        loss_temporal = torch.tensor(0.0, device=x_hat_t.device)
        
        if x_hat_prev_warped is not None and conf_map_hr is not None:
            # Temporal consistency loss, masked by confidence map
            diff = x_hat_t - x_hat_prev_warped
            temp_loss_map = torch.sqrt(diff * diff + self.charbonnier.eps**2)
            
            # Mask with confidence map
            masked_temp_loss = temp_loss_map * conf_map_hr
            loss_temporal = torch.mean(masked_temp_loss)
            
        total_loss = loss_spatial + self.lambda_temporal * loss_temporal
        
        return total_loss, loss_spatial, loss_temporal
