import cv2
import numpy as np

class OpticalFlowEstimator:
    """Base class for optical flow estimation."""
    def estimate(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        """
        Estimate dense optical flow from prev_frame to curr_frame.
        Returns: flow field of shape (H, W, 2) where flow[..., 0] is X (dx) and flow[..., 1] is Y (dy).
        """
        raise NotImplementedError

class PhaseCorrelationFlow(OpticalFlowEstimator):
    """Global translation estimated via phase correlation."""
    def estimate(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        prev_f32 = np.float32(prev_frame)
        curr_f32 = np.float32(curr_frame)
        # Apply hanning window to reduce edge artifacts
        window = cv2.createHanningWindow((prev_f32.shape[1], prev_f32.shape[0]), cv2.CV_32F)
        shift, response = cv2.phaseCorrelate(prev_f32, curr_f32, window=window)
        # shift is (dx, dy)
        dx, dy = shift
        # Create a dense flow field with this constant global shift
        flow = np.zeros((*prev_frame.shape, 2), dtype=np.float32)
        flow[..., 0] = dx
        flow[..., 1] = dy
        return flow

class FarnebackFlow(OpticalFlowEstimator):
    """Dense optical flow using Gunnar Farneback's algorithm."""
    def estimate(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        return cv2.calcOpticalFlowFarneback(
            prev_frame, curr_frame, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

class DISFlow(OpticalFlowEstimator):
    """Dense Inverse Search (DIS) optical flow."""
    def __init__(self, preset="fast"):
        if preset == "ultrafast":
            self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        elif preset == "fast":
            self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        elif preset == "medium":
            self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        else:
            raise ValueError(f"Unknown DIS preset: {preset}")

    def estimate(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        return self.dis.calc(prev_frame, curr_frame, None)

class PyrLKDenseFlow(OpticalFlowEstimator):
    """Pyramidal Lucas-Kanade dense optical flow (computed on a grid)."""
    def __init__(self, grid_step=8):
        self.grid_step = grid_step

    def estimate(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        h, w = prev_frame.shape
        # Create a grid of points
        y, x = np.mgrid[self.grid_step//2:h:self.grid_step, self.grid_step//2:w:self.grid_step]
        pts_prev = np.column_stack((x.ravel(), y.ravel())).astype(np.float32).reshape(-1, 1, 2)
        
        # Calculate sparse flow
        pts_curr, status, err = cv2.calcOpticalFlowPyrLK(
            prev_frame, curr_frame, pts_prev, None, 
            winSize=(15, 15), maxLevel=2, 
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Interpolate to dense flow
        flow_sparse = (pts_curr - pts_prev).reshape(x.shape[0], x.shape[1], 2)
        
        # Resize to original resolution
        flow_dense = cv2.resize(flow_sparse, (w, h), interpolation=cv2.INTER_LINEAR)
        return flow_dense

def get_flow_estimator(name: str) -> OpticalFlowEstimator:
    if name == "phase_correlation":
        return PhaseCorrelationFlow()
    elif name == "lucas_kanade":
        return PyrLKDenseFlow()
    elif name == "farneback":
        return FarnebackFlow()
    elif name == "dis_ultrafast":
        return DISFlow("ultrafast")
    elif name == "dis_fast":
        return DISFlow("fast")
    elif name == "dis_medium":
        return DISFlow("medium")
    else:
        raise ValueError(f"Unknown flow estimator: {name}")
