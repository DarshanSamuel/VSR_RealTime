import argparse
import time
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.vsr.flow import get_flow_estimator
from src.vsr.warp import warp_image, upscale_flow
from src.vsr.fusion_classical import ClassicalFusion
from src.vsr.infer_onnx import ONNXInferencer
from src.vsr.utils import load_config

def benchmark_speed(config_path, onnx_model_path=None):
    config = load_config(config_path)
    
    # 320x240 is default
    h, w = 240, 320
    
    lr1 = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    lr2 = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    hr_prev = np.random.randint(0, 255, (h * config.scale, w * config.scale), dtype=np.uint8)
    
    flow_est = get_flow_estimator(config.motion_estimator)
    fusion_class = ClassicalFusion(scale=config.scale)
    
    net = None
    if onnx_model_path and Path(onnx_model_path).exists():
        net = ONNXInferencer(onnx_model_path)
        
    iters = 100
    
    print(f"Benchmarking speed (Scale: {config.scale}x, Input: {w}x{h})")
    print("-" * 40)
    
    # 1. Flow
    t0 = time.perf_counter()
    for _ in range(iters):
        flow = flow_est.estimate(lr2, lr1)
    t1 = time.perf_counter()
    ms_flow = (t1 - t0) * 1000 / iters
    print(f"Optical Flow ({config.motion_estimator}): {ms_flow:.2f} ms")
    
    # 2. Upscale flow + Warp
    t0 = time.perf_counter()
    for _ in range(iters):
        flow_hr = upscale_flow(flow, config.scale)
        warped = warp_image(hr_prev, flow_hr, cv2.INTER_CUBIC)
    t1 = time.perf_counter()
    ms_warp = (t1 - t0) * 1000 / iters
    print(f"Upscale Flow + Warp: {ms_warp:.2f} ms")
    
    # 3. Classical Fusion
    t0 = time.perf_counter()
    for _ in range(iters):
        fusion_class.fuse(lr2, warped)
    t1 = time.perf_counter()
    ms_class = (t1 - t0) * 1000 / iters
    print(f"Classical Fusion: {ms_class:.2f} ms")
    
    total_class = ms_flow + ms_warp + ms_class
    print(f"-> Total Classical Pipeline: {total_class:.2f} ms ({1000/total_class:.1f} FPS)")
    
    # 4. Neural Fusion
    if net:
        lr_f32 = lr2.astype(np.float32) / 255.0
        warped_f32 = warped.astype(np.float32) / 255.0
        conf_map = np.ones((h, w), dtype=np.float32)
        
        t0 = time.perf_counter()
        for _ in range(iters):
            net.infer(lr_f32, warped_f32, conf_map)
        t1 = time.perf_counter()
        ms_net = (t1 - t0) * 1000 / iters
        print(f"Neural Inference (ONNX CPU): {ms_net:.2f} ms")
        
        # approximate conf map time
        ms_conf = 1.0 # ~1ms for conf map computation
        
        total_net = ms_flow + ms_warp + ms_conf + ms_net
        print(f"-> Total Neural Pipeline: {total_net:.2f} ms ({1000/total_net:.1f} FPS)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--onnx", default="")
    args = parser.parse_args()
    benchmark_speed(args.config, args.onnx)
