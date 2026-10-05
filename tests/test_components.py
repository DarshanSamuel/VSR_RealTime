import numpy as np
import torch
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.vsr.fusion_classical import decimate, upsample
from src.vsr.warp import get_warp_interpolation
from src.vsr.model import RecurrentVSR
from src.vsr.losses import CharbonnierLoss
from src.vsr.utils import Config

def test_decimate():
    img = np.ones((10, 10), dtype=np.uint8) * 255
    out = decimate(img, 2)
    assert out.shape == (5, 5)
    assert out[0, 0] == 255

def test_upsample():
    img = np.ones((5, 5), dtype=np.uint8) * 255
    out = upsample(img, 2)
    assert out.shape == (10, 10)
    assert out[0, 0] == 255

def test_warp_interpolation():
    assert get_warp_interpolation('bilinear') == cv2.INTER_LINEAR
    assert get_warp_interpolation('bicubic') == cv2.INTER_CUBIC

def test_model_shapes():
    scale = 2
    model = RecurrentVSR(scale=scale, channels=16, num_blocks=2)
    
    # Batch=2, Channel=1
    lr = torch.randn(2, 1, 64, 64)
    history_hr = torch.randn(2, 1, 128, 128)
    conf_map = torch.randn(2, 1, 64, 64)
    
    out = model(lr, history_hr, conf_map)
    
    assert out.shape == (2, 1, 128, 128)

def test_charbonnier_loss():
    loss_fn = CharbonnierLoss(eps=1e-3)
    x = torch.zeros(1, 1, 10, 10)
    y = torch.ones(1, 1, 10, 10)
    loss = loss_fn(x, y)
    
    # Should be close to 1.0
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-2)
