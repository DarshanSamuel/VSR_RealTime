import cv2
import numpy as np

def decimate(image: np.ndarray, scale: int) -> np.ndarray:
    """Downsample an image by area averaging (simulating camera integration)."""
    h, w = image.shape[:2]
    return cv2.resize(image, (w // scale, h // scale), interpolation=cv2.INTER_AREA)

def upsample(image: np.ndarray, scale: int, interpolation=cv2.INTER_CUBIC) -> np.ndarray:
    """Upsample an image."""
    h, w = image.shape[:2]
    return cv2.resize(image, (w * scale, h * scale), interpolation=interpolation)

class ClassicalFusion:
    def __init__(self, scale: int, alpha: float = 0.8, tau: float = 12.0, use_back_projection: bool = True):
        self.scale = scale
        self.alpha = alpha
        self.tau = tau
        self.use_back_projection = use_back_projection
        
    def fuse(self, y_t: np.ndarray, warped_x_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform classical recursive fusion.
        Args:
            y_t: Current LR frame (H, W).
            warped_x_prev: Previous HR output warped to current frame (H*s, W*s).
        Returns:
            x_hat_t: Current HR output.
            conf_map: Confidence map in HR space (for visualization/debugging).
        """
        # Ensure inputs are float32
        y_t_f32 = y_t.astype(np.float32)
        warped_x_prev_f32 = warped_x_prev.astype(np.float32)
        
        # Upsample current LR frame
        u_y_t = upsample(y_t_f32, self.scale, cv2.INTER_CUBIC)
        
        # Calculate error in LR space
        d_w_x_prev = decimate(warped_x_prev_f32, self.scale)
        e_t_lr = np.abs(d_w_x_prev - y_t_f32)
        
        # Blur the error map slightly to make it robust
        e_t_lr = cv2.GaussianBlur(e_t_lr, (3, 3), 1.0)
        
        # Upsample error to HR space
        e_t_hr = upsample(e_t_lr, self.scale, cv2.INTER_LINEAR)
        
        # Calculate confidence map a_t
        a_t = self.alpha * np.exp(- (e_t_hr / self.tau)**2)
        
        # Fuse
        x_hat_t = a_t * warped_x_prev_f32 + (1.0 - a_t) * u_y_t
        
        # Back-projection step
        if self.use_back_projection:
            d_x_hat_t = decimate(x_hat_t, self.scale)
            residual_lr = y_t_f32 - d_x_hat_t
            residual_hr = upsample(residual_lr, self.scale, cv2.INTER_CUBIC)
            x_hat_t += residual_hr
            
        # Clip to valid range
        x_hat_t = np.clip(x_hat_t, 0, 255)
        
        return x_hat_t.astype(np.uint8), a_t
