import cv2
import numpy as np
import random

class DegradationPipeline:
    def __init__(self, scale=2, config=None):
        self.scale = scale
        # Defaults
        self.methods = ['blur_decimate'] # can include 'bicubic', 'noise', 'jpeg'
        self.blur_sigma_range = (0.5, 2.0)
        self.noise_sigma = 5.0
        self.jpeg_quality_range = (40, 95)
        
        if config:
            self.methods = getattr(config, 'degradation_methods', self.methods)
            
    def apply(self, hr_frame: np.ndarray) -> np.ndarray:
        """Apply degradations to an HR frame to generate an LR frame."""
        lr_frame = hr_frame.copy()
        
        for method in self.methods:
            if method == 'bicubic':
                h, w = lr_frame.shape[:2]
                lr_frame = cv2.resize(lr_frame, (w // self.scale, h // self.scale), interpolation=cv2.INTER_CUBIC)
            
            elif method == 'blur_decimate':
                # Gaussian blur
                sigma = random.uniform(*self.blur_sigma_range)
                ksize = int(np.ceil(sigma * 3)) * 2 + 1
                blurred = cv2.GaussianBlur(lr_frame, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
                
                # Naive decimation
                h, w = blurred.shape[:2]
                lr_frame = cv2.resize(blurred, (w // self.scale, h // self.scale), interpolation=cv2.INTER_NEAREST)
                
            elif method == 'noise':
                noise = np.random.normal(0, self.noise_sigma, lr_frame.shape).astype(np.float32)
                lr_frame = np.clip(lr_frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                
            elif method == 'jpeg':
                quality = random.randint(*self.jpeg_quality_range)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, encimg = cv2.imencode('.jpg', lr_frame, encode_param)
                lr_frame = cv2.imdecode(encimg, 1)
                
        return lr_frame
