import argparse
import time
import cv2
import numpy as np
from pathlib import Path
import sys

# Ensure src is in path for scripts
sys.path.append(str(Path(__file__).parent.parent))

from src.vsr.capture import VideoSource
from src.vsr.flow import get_flow_estimator
from src.vsr.warp import warp_image, upscale_flow, get_warp_interpolation
from src.vsr.fusion_classical import ClassicalFusion
from src.vsr.utils import load_config
from src.vsr.infer_onnx import ONNXInferencer
from src.vsr.fusion_classical import upsample, decimate

def compute_confidence_map_numpy(y_t_f32, warped_x_prev_f32, scale, tau):
    # LR error
    d_w_x_prev = decimate(warped_x_prev_f32, scale)
    e_t_lr = np.abs(d_w_x_prev - y_t_f32)
    e_t_lr = cv2.GaussianBlur(e_t_lr, (3, 3), 1.0)
    a_t = 0.8 * np.exp(- (e_t_lr / tau)**2)
    return a_t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Webcam index, RTSP URL, or video file path")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--onnx", default="", help="Path to ONNX model for neural mode")
    args = parser.parse_args()

    config = load_config(args.config)
    
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
        
    try:
        cap = VideoSource(source)
    except Exception as e:
        print(f"Error opening source: {e}")
        return

    flow_estimator = get_flow_estimator(config.motion_estimator)
    fusion_classical = ClassicalFusion(scale=config.scale, alpha=config.alpha, tau=config.tau, use_back_projection=config.use_back_projection)
    warp_interp = get_warp_interpolation(config.warp_interpolation)
    
    use_neural = False
    inferencer = None
    if args.onnx:
        if Path(args.onnx).exists():
            inferencer = ONNXInferencer(args.onnx)
            use_neural = True
            print(f"Loaded ONNX model from {args.onnx}")
        else:
            print(f"ONNX model {args.onnx} not found, falling back to classical.")
            
    mode = "Neural" if use_neural else "Classical"
    
    prev_y_t = None
    x_hat_prev = None
    
    print(f"Starting {mode} real-time demo.")
    print("Press 'q' to quit, 'r' to reset recurrent state, 'm' to toggle mode.")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        t0 = time.time()
        ret, bgr_frame = cap.read()
        if not ret or bgr_frame is None:
            if not cap.is_live:
                break
            continue
            
        # Simulate Low-Resolution input by downsizing the camera feed
        # We target a width of 320 (e.g. 320x240) to ensure CPU real-time speeds (25+ FPS)
        target_lr_width = 320
        h_orig, w_orig = bgr_frame.shape[:2]
        target_lr_height = int(h_orig * (target_lr_width / w_orig))
        
        # Ensure dimensions are multiples of config.scale (2)
        target_lr_width = target_lr_width - (target_lr_width % config.scale)
        target_lr_height = target_lr_height - (target_lr_height % config.scale)
        
        bgr_lr = cv2.resize(bgr_frame, (target_lr_width, target_lr_height), interpolation=cv2.INTER_AREA)
            
        ycrcb = cv2.cvtColor(bgr_lr, cv2.COLOR_BGR2YCrCb)
        y_t, cr_t, cb_t = cv2.split(ycrcb)
        
        h_lr, w_lr = y_t.shape
        h_hr, w_hr = h_lr * config.scale, w_lr * config.scale
        
        y_t_bicubic = cv2.resize(y_t, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
        cr_hr = cv2.resize(cr_t, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
        cb_hr = cv2.resize(cb_t, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
        
        t_net = 0
        if prev_y_t is None or x_hat_prev is None:
            x_hat_t = y_t_bicubic.copy()
            flow_ms = 0
        else:
            t_flow0 = time.perf_counter()
            flow_lr = flow_estimator.estimate(y_t, prev_y_t)
            t_flow1 = time.perf_counter()
            flow_ms = (t_flow1 - t_flow0) * 1000
            
            flow_hr = upscale_flow(flow_lr, config.scale)
            warped_x_prev = warp_image(x_hat_prev, flow_hr, warp_interp)
            
            if mode == "Classical":
                x_hat_t, _ = fusion_classical.fuse(y_t, warped_x_prev)
            else:
                t_net0 = time.perf_counter()
                y_t_f32 = y_t.astype(np.float32) / 255.0
                warped_x_prev_f32 = warped_x_prev.astype(np.float32) / 255.0
                
                # compute confidence map in LR
                conf_map_lr = compute_confidence_map_numpy(y_t_f32, warped_x_prev_f32, config.scale, config.tau / 255.0)
                
                # The fast ONNX model now outputs the residual (from -1.0 to 1.0 approx)
                residual_f32 = inferencer.infer(y_t_f32, warped_x_prev_f32, conf_map_lr)
                
                # Add the residual to our OpenCV bicubic upsample
                y_t_bicubic_f32 = y_t_bicubic.astype(np.float32) / 255.0
                out_f32 = y_t_bicubic_f32 + residual_f32
                
                x_hat_t = np.clip(out_f32 * 255.0, 0, 255).astype(np.uint8)
                t_net1 = time.perf_counter()
                t_net = (t_net1 - t_net0) * 1000
                
        prev_y_t = y_t.copy()
        x_hat_prev = x_hat_t.copy()
        
        ycrcb_hr = cv2.merge([x_hat_t, cr_hr, cb_hr])
        bgr_hr = cv2.cvtColor(ycrcb_hr, cv2.COLOR_YCrCb2BGR)
        
        ycrcb_bicubic = cv2.merge([y_t_bicubic, cr_hr, cb_hr])
        bgr_bicubic = cv2.cvtColor(ycrcb_bicubic, cv2.COLOR_YCrCb2BGR)
        
        t1 = time.time()
        fps = 1.0 / (t1 - t0 + 1e-6)
        
        # 4th panel: the original Low-Res frame scaled up with nearest-neighbor to show true pixelation
        lr_vis = cv2.resize(bgr_lr, (w_hr, h_hr), interpolation=cv2.INTER_NEAREST)
        
        if mode == "Neural" and 'residual_f32' in locals():
            res_vis = np.clip(np.abs(residual_f32) * 5.0 * 255.0, 0, 255).astype(np.uint8)
            res_bgr = cv2.applyColorMap(res_vis, cv2.COLORMAP_JET)
        else:
            # Blank screen for classical mode
            res_bgr = np.zeros_like(bgr_hr)
            
        top_row = np.hstack((bgr_bicubic, bgr_hr))
        bottom_row = np.hstack((lr_vis, res_bgr))
        grid = np.vstack((top_row, bottom_row))
        
        cv2.putText(grid, f"Bicubic {config.scale}x", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(grid, f"{mode} VSR {config.scale}x", (w_hr + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(grid, "Original Input (Nearest)", (10, h_hr + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if mode == "Neural":
            cv2.putText(grid, "Neural 'Brain' (Edges)", (w_hr + 10, h_hr + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.putText(grid, f"Flow: {flow_ms:.1f}ms", (w_hr + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if mode == "Neural":
            cv2.putText(grid, f"Net: {t_net:.1f}ms", (w_hr + 10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(grid, f"Total FPS: {fps:.1f}", (w_hr + 10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(grid, f"Total FPS: {fps:.1f}", (w_hr + 10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Scale the grid down to fit on laptop screens
        max_height = 720
        if grid.shape[0] > max_height:
            scale_ratio = max_height / grid.shape[0]
            grid = cv2.resize(grid, (int(grid.shape[1] * scale_ratio), int(grid.shape[0] * scale_ratio)))
            
        cv2.imshow("VSR Comparison", grid)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            prev_y_t = None
            x_hat_prev = None
            print("Resetting recurrent state.")
        elif key == ord('m'):
            if inferencer is not None:
                mode = "Classical" if mode == "Neural" else "Neural"
                print(f"Switched to {mode} mode.")
            else:
                print("Cannot switch to Neural mode: no ONNX model loaded.")
            
        frame_count += 1
        
    cap.stop()
    cv2.destroyAllWindows()
    total_time = time.time() - start_time
    print(f"Processed {frame_count} frames in {total_time:.2f}s ({frame_count/total_time:.2f} FPS avg).")

if __name__ == "__main__":
    main()
