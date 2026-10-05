# Real-Time Multi-Frame Video Super-Resolution (VSR)

A CPU-only real-time video super-resolution system combining classical DSP with a tiny neural network.

## Installation

1. Requires Python 3.11+.
2. Create virtual env and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Or .\venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

## Running the Live Demo

### 1. Classical Mode (No Training Required)
You can run the live demo immediately using the classical recursive fusion baseline.

```bash
# Use webcam (index 0)
python scripts/live_demo.py --source 0

# Or use a local video file
python scripts/live_demo.py --source path/to/video.mp4
```
Controls:
- `q`: Quit
- `r`: Reset recurrent state (use this if the scene changes abruptly)
- `m`: Toggle Neural mode (if ONNX model is loaded)

### 2. Neural Network Mode

**Step A: Build a small dataset from a local video or Vimeo-90K**
If you want to train on a local video file:
```bash
python src/vsr/dataset.py --video path/to/video.mp4 --out outputs/dataset
```
If you downloaded the **Vimeo-90K Septuplet** dataset (recommended for better generalization), you can process it like this:
```bash
python src/vsr/dataset.py --vimeo path/to/vimeo_septuplet --out outputs/dataset --split train
```
*(This will read `sep_trainlist.txt`, load the frames, apply degradations, compute optical flow offline, and save them in the expected `outputs/dataset` structure).*

**Step B: Train the model**
```bash
# Stage A (Single-frame pretrain)
python src/vsr/train.py --data outputs/dataset --out outputs/checkpoints --stage A

# Stage B (Recurrent finetune)
python src/vsr/train.py --data outputs/dataset --out outputs/checkpoints --stage B
```

**Step C: Export to ONNX**
```bash
python src/vsr/export_onnx.py --checkpoint outputs/checkpoints/model_stage_B_epoch_100.pth --out outputs/vsr_model.onnx
```
*(Add `--quantize` for experimental INT8 quantization).*

**Step D: Run Live Demo in Neural Mode**
```bash
python scripts/live_demo.py --source 0 --onnx outputs/vsr_model.onnx
```
Press `m` while running to toggle between Classical and Neural modes live.

## Evaluation & Benchmarks

Run the evaluation suite to generate a CSV report and PSNR drift plots:
```bash
python scripts/evaluate.py --video path/to/video.mp4 --onnx outputs/vsr_model.onnx
```

Compare optical flow estimators:
```bash
python scripts/benchmark_flow.py --video path/to/video.mp4
```

## Known Limitations
- The system heavily relies on optical flow accuracy. Complex non-rigid motion or heavy occlusion may cause artifacts.
- The recurrent hidden state can accumulate errors over long sequences ("drift"). The training script mitigates this by randomly zeroing the history input, but it's not perfect. Scene cuts will corrupt the output unless the `scene_cut.py` logic automatically resets the state.
