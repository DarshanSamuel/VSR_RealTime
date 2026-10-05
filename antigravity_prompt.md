# Prompt for Gemini 3.1 Pro (Antigravity) — Real-Time Multi-Frame Video Super-Resolution

> Copy everything below the line into Antigravity as the task/mission prompt.

---

## Role

You are acting as a senior Python / computer-vision engineer pair-programming inside Antigravity. Build a complete, working, real-time multi-frame video super-resolution (VSR) system as a university Digital Signal Processing course project. Work autonomously, create a proper repository structure, write clean documented code, and run/test what you build as you go rather than producing it all in one untested dump. After each major module, run it (or a smoke test) and fix errors before moving on.

## Project context (read fully before starting)

**Course:** Digital Signal Processing. **Goal:** combine classical DSP with a small AI model to do a real-time image/video task, run entirely on a CPU-only laptop (no CUDA GPU).

**Target hardware for inference:** AMD Ryzen 7 5700U, 8 cores / 16 threads, AVX2, 16 GB RAM, integrated GPU only (not usable by PyTorch). All real-time code must run acceptably on CPU alone. Training the neural network should use a free cloud GPU (Colab/Kaggle) in a separate notebook, but it must also be possible — just slower — to train the tiny model on this CPU, so keep the training script hardware-agnostic (`torch.device` auto-detect).

**The chosen project:** Real-time multi-frame video super-resolution. Low-resolution video (CCTV, video calls, legacy footage) is upscaled live using temporal information from multiple frames, not just a single frame. This is a causal, streaming system (it must work live, so it cannot look at future frames — no bidirectional methods like BasicVSR).

### Why multi-frame, and the imaging model

A video observes the same scene many times with slightly different sub-pixel offsets. If the low-resolution (LR) frames contain aliasing, each one carries different samples of the same underlying high-resolution (HR) scene, and a multi-frame method can recover real detail rather than inventing it (which is what single-image SR has to do). The assumed observation model is:

```
y_k = D · H · F_k · x + n_k
```

where `x` is the HR frame, `F_k` is the motion warp of frame k, `H` is blur, `D` is decimation, and `n_k` is noise. This underlies both the classical baseline and the training-data degradation pipeline.

### System architecture to implement

```
Source (webcam / RTSP / file) --thread--> latest-frame buffer
   -> BGR to YCrCb, split luma (Y) from chroma (Cr, Cb)
   -> LR luma y_t
   -> optical flow estimation at LR resolution (y_t -> y_{t-1})
   -> upscale flow field by scale factor s, multiply vectors by s
   -> warp previous HR output x_hat_{t-1} to current frame (bicubic remap)
   -> confidence map = blurred |downsample(warped) - y_t|  (flags occlusion / bad flow)
   -> FUSION (two interchangeable implementations):
        A) classical recursive fusion + one back-projection step
        B) tiny recurrent CNN (space-to-depth(warped) + y_t + confidence -> residual)
   -> x_hat_t (HR luma)
   -> chroma upscaled separately with plain bicubic (never super-resolved)
   -> merge YCrCb -> BGR -> display / stream out
   -> x_hat_t becomes x_hat_{t-1} for the next frame (the recurrence)
```

Only the luma channel is super-resolved (saves ~3x compute; chroma SR is visually unnecessary). Flow, warping and recurrence state live **outside** the neural network — only the small feed-forward fusion network should be exportable to ONNX.

### Classical baseline (must be implemented and must work standalone, no AI)

- Bicubic and Lanczos interpolation as the quality floor.
- Shift-and-add fusion + deblurring (Elad & Hel-Or style) as an optional extra baseline.
- **Recursive fusion** (Farsiu-style, dynamic/video SR):
  ```
  x_hat_t = a_t ⊙ W(x_hat_{t-1}) + (1 - a_t) ⊙ U(y_t)
  a_t(p)  = alpha * exp( -(e_t(p) / tau)^2 ),   e_t = |D(W(x_hat_{t-1})) - y_t|
  ```
  `W` = flow-based warp, `U` = bicubic upsample, `D` = decimation (area downsample). `alpha` and `tau` are tunable (defaults: alpha=0.8, tau=12.0).
- One iterative back-projection correction step after fusion:
  `x_hat_t += U(y_t - D(x_hat_t))`.
- This classical pipeline must run in real time and produce a complete, demonstrable result **on its own**, independent of whether the neural network is trained yet. Treat it as Milestone 1.

### Motion estimation (implement all, make them swappable via a common interface)

1. Phase correlation with sub-pixel refinement (global translation only — fast, good for camera shake).
2. Pyramidal Lucas-Kanade (sparse, interpolated to a dense field, or use OpenCV's dense variant).
3. Farnebäck dense optical flow (OpenCV).
4. OpenCV DISOpticalFlow (ULTRAFAST / FAST / MEDIUM presets) — this should be the real-time default.

Provide a benchmarking script that measures each estimator's speed (ms/frame) and warping error (difference between the current frame and the flow-warped previous frame) on a short test clip, and prints/plots a speed-vs-accuracy comparison.

### The neural network (lightweight recurrent fusion model)

Design, in the spirit of FRVSR (Sajjadi et al.) but far smaller and using **classical, not learned, optical flow**:

- **Input:** concatenation of (1) LR luma frame, (2) `space_to_depth(warped_prev_hr_output, s)` giving `s^2` channels, (3) the confidence map. Total input channels = `1 + s^2 + 1`.
- **Architecture:** `Conv3x3(in_ch -> 16) + ReLU` head → 2 residual blocks (`Conv3x3(16->16), ReLU, Conv3x3(16->16)`, with a skip add) → `Conv3x3(16 -> s^2)` tail → `PixelShuffle(s)` to go from LR to HR resolution. Output = `bicubic_upsample(LR) + residual` (global skip connection). Target size: roughly 10-15k parameters for `s=2`.
- Make channel count (default 16), number of residual blocks (default 2), and scale factor `s` (default 2, support 3 and 4) configurable via a config file or CLI flags — these are ablation knobs.
- **Training stages:**
  - Stage A (single-frame pretrain): feed `bicubic(current LR)` in place of the warped-history input (i.e. simulate "no history") and train on single LR/HR pairs. This doubles as the single-image SR baseline for comparison.
  - Stage B (recurrent fine-tune): train on 5-7 frame clips with truncated backpropagation through time. Precompute optical flow for training clips offline (don't recompute it every epoch) and cache it.
  - **Loss:** Charbonnier loss (`sqrt((x - y)^2 + eps^2)`) on the output, plus a temporal-consistency term: `lambda * Charbonnier(x_hat_t, warp(x_hat_{t-1}))`, masked by the confidence map. Make `lambda` configurable (default 0.1-0.5, let me tune it).
  - **Drift control:** during training, occasionally corrupt or zero out the history input (random probability per step, e.g. 10-20%) so the model doesn't over-rely on a perfect history, and evaluate on sequences much longer than the training clip length to check for drift / color shift over time.
  - Optimizer: Adam with cosine LR decay. Log training curves (loss vs. step) to TensorBoard or a simple CSV + matplotlib plot.
- **Data:**
  - Support training from a small subset of Vimeo-90K septuplets (document the expected directory layout, don't require the full dataset).
  - Also support building a training set from **any local HD video file**: cut it into clips, generate LR versions via the degradation pipeline below. This is the easiest path to get started without downloading a large external dataset.
  - **Degradation pipeline** (must implement at least these, selectable): (a) bicubic downsample (clean baseline, least aliasing), (b) Gaussian blur (sigma in [0.5, 2]) followed by naive decimation with no anti-alias filtering (deliberately aliased — this is where multi-frame SR should help most), (c) additive Gaussian noise, (d) JPEG re-compression at a random quality in [40, 95]. Let these be combined/stacked and controlled by config.
- **Test data:** support evaluating on any local video by self-degrading it (downscale, then try to recover, compare against the real frame) — this gives free ground truth without needing a benchmark dataset, and should be the primary way I demo and sanity-check things.

### Real-time streaming requirements (this is the deliverable that gets demoed live)

- **Sources:** webcam index (`0`, `1`, ...), RTSP/HTTP URL (IP camera, phone running an "IP Webcam" app), and local video files (paced at native FPS so files play at realistic speed, not as fast as the CPU can process them).
- **A background capture thread** must keep only the single newest frame for live sources, so a slow consumer drops frames instead of building up latency. Files are read sequentially in the main loop (no frames skipped) since there's no "live" deadline to protect.
- RTSP should prefer TCP transport with low-delay/no-buffer flags to minimize latency.
- **Side-by-side display:** bicubic vs. multi-frame SR output, with an on-screen overlay showing processing time per frame (ms), estimated max FPS, and — when running in self-degraded "simulate" mode against a known source — live PSNR for both methods so quality can be judged quantitatively while watching, not just qualitatively.
- Keyboard controls: quit, reset recurrent state (for manual scene-cut recovery), toggle between classical-fusion mode and neural-fusion mode live, cycle through motion estimators live.
- **Scene-cut detection**: if the flow warping error spikes well above a rolling baseline, or a frame-to-frame histogram difference is large, automatically reset the recurrent state rather than warping garbage into the new scene.
- **Inference path for the neural network:** export the trained PyTorch model to ONNX and run it with ONNX Runtime (CPU execution provider) in the live loop — do not run raw PyTorch inference in the hot loop. Provide the export script. Try dynamic or static INT8 quantization as an optional, clearly-labeled experimental mode, and warn in logs/docs that quantization error can accumulate through the recurrence and should be checked on long sequences before trusting it.
- Multithread capture vs. processing vs. display sensibly; avoid blocking the capture thread on slow inference.

### Evaluation / benchmarking requirements

Build scripts (not just ad hoc notebook cells) that produce, and save to disk (CSV + matplotlib PNG plots):

1. **Quality ladder:** bicubic vs. Lanczos vs. single-frame network (Stage A) vs. classical recursive fusion vs. full recurrent hybrid (Stage B) — Y-channel PSNR and SSIM, on at least one real or self-degraded test clip, ideally across a few degradation types.
2. **Temporal consistency:** warping error over time (per-frame), to show flicker/stability, comparing single-frame vs. recurrent outputs.
3. **Long-sequence drift check:** run 500-1000+ frames and plot PSNR vs. frame index to check for accumulated drift.
4. **Speed/feasibility report:** FPS and per-stage latency breakdown (capture, flow, warp, network, merge/display) measured on this machine, plus parameter count and estimated MACs/FLOPs for the network — this replaces/validates the rough numbers below.
5. **Ablation runner:** a config-driven script able to sweep: motion estimator choice, warp interpolation kernel (bilinear/bicubic/Lanczos), confidence map on/off, back-projection on/off, fixed-alpha classical fusion vs. learned fusion, network width/depth, and scale factor — logging results into one comparison table automatically rather than requiring manual re-running.

Rough compute expectations to sanity-check against (treat as estimates to verify, not ground truth): a ~16-channel, 2-residual-block network at 320x240 -> 640x480 (x2) is expected to need roughly 1.5-2 GFLOPs/frame, which should allow somewhere in the ballpark of real-time (~15-30+ FPS) on this CPU depending on thread tuning — measure this directly with the benchmark script rather than assuming it.

## Tech stack

- Python 3.11+
- OpenCV (`opencv-python`) for capture, color conversion, optical flow, remap/warping
- NumPy
- PyTorch (CPU build is fine; auto-detect CUDA/MPS if present but don't require it) for training
- ONNX + ONNX Runtime for real-time CPU inference
- scikit-image (for SSIM) or implement PSNR/SSIM manually with justification
- Matplotlib for plots
- PyYAML or a simple dataclass-based config system for experiment/ablation configs
- pytest for unit tests
- tqdm for progress bars
- Optional/stretch: Streamlit or Gradio for a simple control dashboard; TensorBoard for training curves

## Repository structure to create

```
vsr-project/
├── README.md                  # setup, how to run each script, reproduction instructions
├── requirements.txt
├── pyproject.toml / setup.cfg # packaging + lint config (ruff/black)
├── configs/
│   ├── default.yaml           # scale, channels, blocks, alpha, tau, lambda, paths...
│   └── ablations/*.yaml       # one file per ablation sweep
├── src/vsr/
│   ├── capture.py             # Source class: webcam/RTSP/file, threaded latest-frame buffer
│   ├── flow.py                # common interface + phase correlation / LK / Farneback / DIS
│   ├── warp.py                # remap-based warping, interpolation kernel options
│   ├── fusion_classical.py    # recursive fusion + confidence map + back-projection
│   ├── degrade.py             # the degradation pipeline (bicubic/aliased/noise/JPEG)
│   ├── dataset.py             # Vimeo-90K loader + "build dataset from any local video" tool
│   ├── model.py                # the recurrent network (PyTorch)
│   ├── losses.py               # Charbonnier + temporal consistency loss
│   ├── train.py                # Stage A + Stage B training loop, checkpointing, TB logging
│   ├── export_onnx.py          # PyTorch -> ONNX (+ optional INT8 quantization) export
│   ├── infer_onnx.py           # ONNX Runtime wrapper used by the live app
│   ├── metrics.py              # PSNR, SSIM, warping error
│   ├── scene_cut.py            # scene-cut detection logic
│   └── utils.py
├── scripts/
│   ├── live_demo.py            # the main real-time application (CLI flags for source, scale, mode)
│   ├── benchmark_flow.py       # motion estimator speed/accuracy comparison
│   ├── benchmark_speed.py      # per-stage latency + FPS + FLOPs report
│   ├── evaluate.py             # quality ladder + temporal + drift evaluation, saves CSV/plots
│   └── run_ablations.py        # sweeps configs/ablations/*.yaml and aggregates results
├── notebooks/
│   └── colab_train.ipynb       # same training code, ready to run on a free GPU
├── tests/
│   └── test_*.py               # unit tests for flow, warp, fusion math, model shapes, metrics
└── outputs/                    # checkpoints, onnx exports, plots, csv — gitignored
```

## What to build, in order (please actually execute/test each stage before moving to the next)

1. Repo scaffold, `requirements.txt`, config system, logging setup.
2. `capture.py` — test it opens a webcam (index 0) and a sample local video file; print measured FPS.
3. `flow.py` + `warp.py` — implement all four estimators behind one interface; write a quick script that loads two frames from a sample video, computes flow each way, warps, and reports warping error + timing, so correctness is checked before anything depends on it.
4. `fusion_classical.py` — wire capture → flow → warp → recursive fusion → back-projection into `scripts/live_demo.py` in **classical-only mode**. This should be a fully working, demoable real-time app before any neural network exists. Run it (headless mode against a sample file is fine if no camera is available in this environment) and confirm it produces sensible output and FPS numbers.
5. `degrade.py`, `dataset.py` — build the "make a training set from any local video" tool and the degradation pipeline; unit test shapes and value ranges.
6. `model.py`, `losses.py`, `train.py` — implement and train Stage A (single-frame) first on a tiny subset to confirm the loop runs and loss decreases, then Stage B (recurrent, truncated BPTT). Make it trivially runnable on CPU with a tiny config for a smoke test, and documented for a full run on Colab.
7. `export_onnx.py`, `infer_onnx.py` — export, verify ONNX output matches PyTorch output numerically within tolerance, then benchmark ONNX Runtime speed.
8. Wire the neural fusion path into `live_demo.py` as a mode toggle alongside the classical path.
9. `metrics.py`, `scene_cut.py`, `evaluate.py`, `benchmark_flow.py`, `benchmark_speed.py`, `run_ablations.py` — the evaluation suite described above.
10. `tests/` — reasonably thorough unit tests, not just happy-path.
11. `README.md` — clear setup and run instructions, including exact commands to reproduce every plot/table, and a short "known limitations" section.

## Additional features you may add if you think they improve the project

You have latitude to extend this beyond the literal spec where it clearly strengthens the project without jeopardizing the CPU real-time core deliverable. Good candidates, roughly in order of value — use your judgment on which are worth the time:

- **ROI-based super-resolution mode:** track a face or license-plate-like region (e.g. with a lightweight detector or simple tracker) and apply SR only inside that region at a higher effective scale, compositing it back — much cheaper than full-frame SR and arguably more realistic for a CCTV use case.
- **H.264/H.265 motion-vector extraction** (via ffmpeg) as a free, near-zero-cost alternative "motion estimator" for pre-recorded files, benchmarked against the optical-flow estimators.
- **A small Streamlit/Gradio control panel** to adjust alpha/tau/scale/estimator live without editing config files, useful for the demo.
- **Automatic report generation:** a script that runs the full evaluation suite and assembles the resulting CSVs/plots into a single Markdown (or PDF) summary report.
- **Docker/conda environment file** for reproducibility.
- **Experiment tracking** via a simple local TensorBoard setup (avoid requiring external accounts).
- **Graceful degradation mode:** if real-time budget is exceeded, automatically drop to a smaller scale factor or lower input resolution rather than falling behind, logging when this happens.
- **Recording mode:** save the side-by-side comparison video (bicubic vs. SR) to an MP4 file for later use in a demo video/report, since video recording of a live screen is otherwise fiddly.

Please don't add heavyweight dependencies (web frameworks beyond Streamlit/Gradio, cloud SDKs, databases) — keep everything runnable locally with `pip install -r requirements.txt`.

## Non-functional requirements

- Target 25+ FPS end-to-end at 320x240 → 640x480 (x2 scale) on the stated CPU in classical mode, and report honestly (don't hide it) if the neural mode falls short — the evaluation suite exists specifically to measure and expose this.
- Code must be typed where reasonable (type hints), documented with docstrings, formatted with `black`/`ruff`, and have no dead code left over from experimentation.
- Every number that ends up in a report or plot must be reproducible by running a specific documented script/command — no hand-edited results.
- Handle errors gracefully (camera not found, RTSP connection drop, missing checkpoint file) with clear error messages, not stack traces dumped on the user.
- Commit-sized, incremental changes if using git — don't produce one giant unreviewable diff.

## Definition of done

- `scripts/live_demo.py` runs against at least a local video file (and a webcam if available in this environment) in both classical and neural modes, with the on-screen overlay working.
- `scripts/evaluate.py` and `scripts/benchmark_speed.py` run end-to-end and produce CSV + PNG outputs in `outputs/`.
- `pytest` passes.
- `README.md` lets a new user go from clone to a running live demo in under 10 commands.

Start by scaffolding the repository and implementing steps 1-4 above (capture through the classical-only real-time demo), run it, show me the result, and then proceed to the neural network stages.
