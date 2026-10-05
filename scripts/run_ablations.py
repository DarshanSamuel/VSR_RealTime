import argparse
import yaml
import subprocess
import os
import csv
from pathlib import Path

def run_ablations(ablations_dir, video_path, out_dir):
    ablations_dir = Path(ablations_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    configs = list(ablations_dir.glob("*.yaml"))
    
    if not configs:
        print(f"No ablation configs found in {ablations_dir}")
        return
        
    results_csv = out_dir / "ablation_results.csv"
    
    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Config", "Method", "PSNR", "SSIM", "Warp_Error"])
        
    for cfg in configs:
        print(f"Running ablation: {cfg.name}")
        
        cfg_out_dir = out_dir / cfg.stem
        
        # Run evaluate.py with this config
        cmd = [
            "python", "scripts/evaluate.py",
            "--video", str(video_path),
            "--config", str(cfg),
            "--out", str(cfg_out_dir),
            "--frames", "50"
        ]
        
        subprocess.run(cmd)
        
        # Parse results
        res_file = cfg_out_dir / "evaluation_results.csv"
        if res_file.exists():
            with open(res_file, 'r') as f:
                reader = csv.reader(f)
                next(reader) # skip header
                for row in reader:
                    method, psnr, ssim, warp_err = row
                    with open(results_csv, 'a', newline='') as out_f:
                        writer = csv.writer(out_f)
                        writer.writerow([cfg.name, method, psnr, ssim, warp_err])
                        
    print(f"Ablations complete. Summary in {results_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations", default="configs/ablations")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="outputs/ablations")
    args = parser.parse_args()
    
    run_ablations(args.ablations, args.video, args.out)
