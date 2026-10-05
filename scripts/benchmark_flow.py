import argparse
import cv2
import time
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.vsr.flow import get_flow_estimator

def benchmark_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    
    if not ret:
        print("Failed to read video")
        return
        
    y1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    y2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    estimators = ["phase_correlation", "lucas_kanade", "farneback", "dis_ultrafast", "dis_fast", "dis_medium"]
    
    print(f"{'Estimator':<20} | {'Time (ms)':<10} | {'FPS':<10}")
    print("-" * 45)
    
    for est_name in estimators:
        try:
            est = get_flow_estimator(est_name)
        except Exception as e:
            print(f"{est_name:<20} | Error: {e}")
            continue
            
        # Warmup
        est.estimate(y1, y2)
        
        # Benchmark
        iters = 50
        t0 = time.perf_counter()
        for _ in range(iters):
            flow = est.estimate(y1, y2)
        t1 = time.perf_counter()
        
        avg_time_ms = (t1 - t0) * 1000 / iters
        fps = 1000.0 / avg_time_ms
        
        print(f"{est_name:<20} | {avg_time_ms:<10.2f} | {fps:<10.1f}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    benchmark_flow(args.video)
