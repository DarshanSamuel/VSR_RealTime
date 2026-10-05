import cv2
import numpy as np

class SceneCutDetector:
    def __init__(self, history_size=10, error_threshold_multiplier=3.0, hist_threshold=0.5):
        self.history_size = history_size
        self.error_threshold_multiplier = error_threshold_multiplier
        self.hist_threshold = hist_threshold
        
        self.error_history = []
        
    def detect(self, prev_frame: np.ndarray, curr_frame: np.ndarray, warping_error: float) -> bool:
        """
        Detect scene cut based on warping error and histogram difference.
        Returns True if a scene cut is detected.
        """
        # 1. Histogram difference
        hist_prev = cv2.calcHist([prev_frame], [0], None, [256], [0, 256])
        hist_curr = cv2.calcHist([curr_frame], [0], None, [256], [0, 256])
        
        cv2.normalize(hist_prev, hist_prev, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_curr, hist_curr, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        hist_diff = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_BHATTACHARYYA)
        
        if hist_diff > self.hist_threshold:
            self.error_history = [] # Reset history on cut
            return True
            
        # 2. Warping error spike
        if len(self.error_history) == self.history_size:
            mean_error = np.mean(self.error_history)
            std_error = np.std(self.error_history)
            
            threshold = mean_error + self.error_threshold_multiplier * std_error
            
            if warping_error > threshold and warping_error > mean_error * 2.0:
                self.error_history = []
                return True
                
        # Update history
        self.error_history.append(warping_error)
        if len(self.error_history) > self.history_size:
            self.error_history.pop(0)
            
        return False
