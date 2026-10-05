import cv2
import numpy as np

def get_warp_interpolation(interp_str: str) -> int:
    if interp_str == 'bilinear':
        return cv2.INTER_LINEAR
    elif interp_str == 'bicubic':
        return cv2.INTER_CUBIC
    elif interp_str == 'lanczos':
        return cv2.INTER_LANCZOS4
    else:
        raise ValueError(f"Unknown interpolation: {interp_str}")

def warp_image(image: np.ndarray, flow: np.ndarray, interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    """
    Warp an image based on optical flow.
    The flow should map pixels from the target (current) frame to the source (previous) frame.
    flow[..., 0] is X displacement, flow[..., 1] is Y displacement.
    """
    h, w = image.shape[:2]
    # Create grid
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Add flow to grid
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    
    # Remap
    warped = cv2.remap(image, map_x, map_y, interpolation, borderMode=cv2.BORDER_REPLICATE)
    return warped

def upscale_flow(flow: np.ndarray, scale: int) -> np.ndarray:
    """
    Upscale an optical flow field by a scale factor.
    The vectors are also multiplied by the scale factor.
    """
    h, w = flow.shape[:2]
    new_h, new_w = h * scale, w * scale
    
    # Resize flow field
    flow_hr = cv2.resize(flow, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Scale vectors
    flow_hr *= scale
    return flow_hr
