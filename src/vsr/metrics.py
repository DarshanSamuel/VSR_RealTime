import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Structural Similarity Index."""
    # Assuming images are grayscale or Y channel (H, W)
    return ssim(img1, img2, data_range=255)

def calculate_warping_error(curr_frame: np.ndarray, warped_prev_frame: np.ndarray) -> float:
    """Calculate Mean Absolute Error between current frame and warped previous frame."""
    return np.mean(np.abs(curr_frame.astype(np.float64) - warped_prev_frame.astype(np.float64)))
