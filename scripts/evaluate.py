import argparse
import cv2
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.vsr.degrade import DegradationPipeline
from src.vsr.flow import get_flow_estimator
from src.vsr.warp import warp_image, upscale_flow
from src.vsr.fusion_classical import ClassicalFusion
from src.vsr.infer_onnx import ONNXInferencer
from src.vsr.metrics import calculate_psnr, calculate_ssim, calculate_warping_error
from src.vsr.utils import load_config
from src.vsr.fusion_classical import upsample

def run_evaluation(video_path, config_path, onnx_model_path, out_dir, num_frames=100):
    config = load_config(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening {video_path}")
        return
        
    degrader = DegradationPipeline(scale=config.scale, config=config)
    flow_estimator = get_flow_estimator(config.motion_estimator)
    fusion_class = ClassicalFusion(scale=config.scale, alpha=config.alpha, tau=config.tau, use_back_projection=config.use_back_projection)
    
    net = None
    if onnx_model_path and Path(onnx_model_path).exists():
        net = ONNXInferencer(onnx_model_path)
        
    methods = ['Bicubic', 'Lanczos', 'Classical']
    if net:
        methods.append('Neural')
        
    results = {m: {'psnr': [], 'ssim': [], 'warp_err': []} for m in methods}
    
    prev_lr = None
    prev_hr_class = None
    prev_hr_net = None
    
    print(f"Evaluating {num_frames} frames...")
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        hr_y, _, _ = cv2.split(ycrcb)
        
        # Crop to multiple of scale
        h, w = hr_y.shape
        new_h, new_w = h - (h % config.scale), w - (w % config.scale)
        hr_y = hr_y[:new_h, :new_w]
        
        lr_y = degrader.apply(hr_y)
        
        # Baselines
        bicubic = cv2.resize(lr_y, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        lanczos = cv2.resize(lr_y, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Evaluate Bicubic
        results['Bicubic']['psnr'].append(calculate_psnr(hr_y, bicubic))
        results['Bicubic']['ssim'].append(calculate_ssim(hr_y, bicubic))
        
        # Evaluate Lanczos
        results['Lanczos']['psnr'].append(calculate_psnr(hr_y, lanczos))
        results['Lanczos']['ssim'].append(calculate_ssim(hr_y, lanczos))
        
        # VSR methods
        if prev_lr is None:
            hr_class = bicubic.copy()
            if net: hr_net = bicubic.copy()
        else:
            flow_lr = flow_estimator.estimate(lr_y, prev_lr)
            flow_hr = upscale_flow(flow_lr, config.scale)
            
            # Classical
            warped_class = warp_image(prev_hr_class, flow_hr, cv2.INTER_CUBIC)
            hr_class, _ = fusion_class.fuse(lr_y, warped_class)
            results['Classical']['warp_err'].append(calculate_warping_error(hr_y, warped_class))
            
            # Neural
            if net:
                warped_net = warp_image(prev_hr_net, flow_hr, cv2.INTER_CUBIC)
                from src.vsr.fusion_classical import decimate
                
                lr_y_f32 = lr_y.astype(np.float32) / 255.0
                warped_net_f32 = warped_net.astype(np.float32) / 255.0
                
                # Confidence map in LR
                d_w = decimate(warped_net_f32, config.scale)
                e_t = np.abs(d_w - lr_y_f32)
                e_t = cv2.GaussianBlur(e_t, (3, 3), 1.0)
                conf = 0.8 * np.exp(- (e_t / (config.tau/255.0))**2)
                
                out_net = net.infer(lr_y_f32, warped_net_f32, conf)
                hr_net = np.clip(out_net * 255.0, 0, 255).astype(np.uint8)
                results['Neural']['warp_err'].append(calculate_warping_error(hr_y, warped_net))
                
        results['Classical']['psnr'].append(calculate_psnr(hr_y, hr_class))
        results['Classical']['ssim'].append(calculate_ssim(hr_y, hr_class))
        
        if net:
            results['Neural']['psnr'].append(calculate_psnr(hr_y, hr_net))
            results['Neural']['ssim'].append(calculate_ssim(hr_y, hr_net))
            
        prev_lr = lr_y.copy()
        prev_hr_class = hr_class.copy()
        if net: prev_hr_net = hr_net.copy()
        
    # Summarize
    log_path = out_dir / "evaluation_results.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Mean PSNR', 'Mean SSIM', 'Mean Warp Error'])
        for m in methods:
            m_psnr = np.mean(results[m]['psnr'])
            m_ssim = np.mean(results[m]['ssim'])
            m_warp = np.mean(results[m]['warp_err']) if len(results[m]['warp_err']) > 0 else 0
            writer.writerow([m, f"{m_psnr:.2f}", f"{m_ssim:.4f}", f"{m_warp:.2f}"])
            print(f"{m}: PSNR={m_psnr:.2f}, SSIM={m_ssim:.4f}")
            
    # Plot PSNR over time (drift check)
    plt.figure(figsize=(10, 5))
    for m in methods:
        plt.plot(results[m]['psnr'], label=m)
    plt.xlabel('Frame')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.title('PSNR over time (Drift check)')
    plt.savefig(out_dir / "psnr_drift.png")
    
    print(f"Evaluation complete. Results saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--onnx", default="")
    parser.add_argument("--out", default="outputs/eval")
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()
    
    run_evaluation(args.video, args.config, args.onnx, args.out, args.frames)
