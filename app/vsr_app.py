"""
VSR Real-Time — Desktop Application
====================================
A standalone PyQt5 GUI that wraps the entire Video Super-Resolution pipeline
into a clean Windows application.  No core source files are modified.

Usage (development):
    python app/vsr_app.py

Usage (packaged .exe):
    dist/VSR_RealTime/VSR_RealTime.exe
"""

import sys
import os
import time
import threading
import subprocess

# Critical: onnxruntime must be imported before PyQt5 on Windows to prevent
# CRT/TLS initialization conflict (0xC0000005 crash).
try:
    import onnxruntime as _ort
except Exception:
    pass

# On Windows, force OpenCV to prioritize DirectShow (DSHOW) over Media Foundation (MSMF).
# MSMF frequently deadlocks or stalls on external USB cameras (such as Logitech Brio).
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Path setup for both script mode and PyInstaller frozen mode
# ---------------------------------------------------------------------------
if getattr(sys, 'frozen', False):
    _APP_DIR = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    _PROJECT_ROOT = _APP_DIR
else:
    _APP_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(_APP_DIR)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import traceback

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QFileDialog, QGroupBox,
    QStatusBar, QSplitter, QFrame, QSizePolicy, QStyle, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon

from src.vsr.capture import VideoSource
from src.vsr.flow import get_flow_estimator
from src.vsr.warp import warp_image, upscale_flow, get_warp_interpolation
from src.vsr.fusion_classical import ClassicalFusion, upsample, decimate
from src.vsr.utils import load_config
from src.vsr.infer_onnx import ONNXInferencer


# ── Utility: detect available cameras ──────────────────────────────────────
def get_camera_names() -> list[str]:
    """Retrieve friendly hardware names of connected cameras via Windows PnP."""
    if os.name != 'nt':
        return []
    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE

        cmd = [
            'powershell', '-NoProfile', '-NonInteractive', '-Command',
            'Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -eq \'Camera\' } | Select-Object -ExpandProperty Name'
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3, startupinfo=startupinfo)
        if res.returncode == 0:
            lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
            return lines
    except Exception:
        pass
    return []


def detect_cameras(max_index: int = 5) -> list[tuple[int, str]]:
    """Probe camera indices 0..max_index using DirectShow (cv2.CAP_DSHOW).
    
    DirectShow reliably opens both integrated webcams and USB cameras (e.g. Logitech Brio)
    without hanging or deadlocking.
    """
    friendly_names = get_camera_names()
    cameras: list[tuple[int, str]] = []
    for idx in range(max_index):
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                name = friendly_names[idx] if idx < len(friendly_names) else f"Camera {idx}"
                cameras.append((idx, f"Camera {idx}: {name} ({w}×{h})"))
                cap.release()
        except Exception:
            pass
    return cameras


# ── Confidence map (same as live_demo.py) ──────────────────────────────────
def compute_confidence_map_numpy(y_t_f32, warped_x_prev_f32, scale, tau):
    d_w_x_prev = decimate(warped_x_prev_f32, scale)
    e_t_lr = np.abs(d_w_x_prev - y_t_f32)
    e_t_lr = cv2.GaussianBlur(e_t_lr, (3, 3), 1.0)
    a_t = 0.8 * np.exp(-(e_t_lr / tau) ** 2)
    return a_t


# ═══════════════════════════════════════════════════════════════════════════
#  Worker thread — runs the capture + inference loop off the GUI thread
# ═══════════════════════════════════════════════════════════════════════════
class VideoWorker(QThread):
    # Signals emitted to the GUI thread
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict)
    error = pyqtSignal(str)
    finished_playback = pyqtSignal()

    def __init__(self, source, config, onnx_path: str, mode: str):
        super().__init__()
        self.source = source
        self.config = config
        self.onnx_path = onnx_path
        self.mode = mode          # "Neural" or "Classical"
        self._running = True

    # -- public control from GUI thread --
    def stop(self):
        self._running = False

    def set_mode(self, mode: str):
        self.mode = mode

    def reset_state(self):
        self._reset_requested = True

    def run(self):
        self._reset_requested = False
        config = self.config

        # Open video source
        try:
            cap = VideoSource(self.source)
        except Exception as e:
            self.error.emit(str(e))
            return

        # Optical flow & fusion
        flow_estimator = get_flow_estimator(config.motion_estimator)
        fusion_classical = ClassicalFusion(
            scale=config.scale, alpha=config.alpha, tau=config.tau,
            use_back_projection=config.use_back_projection,
        )
        warp_interp = get_warp_interpolation(config.warp_interpolation)

        # ONNX model (optional)
        inferencer = None
        if self.onnx_path and os.path.isfile(self.onnx_path):
            try:
                inferencer = ONNXInferencer(self.onnx_path)
            except Exception as e:
                self.error.emit(f"Failed to load ONNX model: {e}")
                return

        prev_y_t = None
        x_hat_prev = None

        while self._running:
            if self._reset_requested:
                prev_y_t = None
                x_hat_prev = None
                self._reset_requested = False

            t0 = time.time()
            ret, bgr_frame = cap.read()
            if not ret or bgr_frame is None:
                if not cap.is_live:
                    self.finished_playback.emit()
                    break
                continue

            # Simulate LR by downsizing to width=320
            target_lr_width = 320
            h_orig, w_orig = bgr_frame.shape[:2]
            target_lr_height = int(h_orig * (target_lr_width / w_orig))
            target_lr_width -= target_lr_width % config.scale
            target_lr_height -= target_lr_height % config.scale

            bgr_lr = cv2.resize(bgr_frame, (target_lr_width, target_lr_height),
                                interpolation=cv2.INTER_AREA)

            ycrcb = cv2.cvtColor(bgr_lr, cv2.COLOR_BGR2YCrCb)
            y_t, cr_t, cb_t = cv2.split(ycrcb)

            h_lr, w_lr = y_t.shape
            h_hr, w_hr = h_lr * config.scale, w_lr * config.scale

            y_t_bicubic = cv2.resize(y_t, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
            cr_hr = cv2.resize(cr_t, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
            cb_hr = cv2.resize(cb_t, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)

            mode = self.mode
            t_net = 0.0
            flow_ms = 0.0
            residual_f32 = None

            if prev_y_t is None or x_hat_prev is None:
                x_hat_t = y_t_bicubic.copy()
            else:
                t_flow0 = time.perf_counter()
                flow_lr = flow_estimator.estimate(y_t, prev_y_t)
                flow_ms = (time.perf_counter() - t_flow0) * 1000

                flow_hr = upscale_flow(flow_lr, config.scale)
                warped_x_prev = warp_image(x_hat_prev, flow_hr, warp_interp)

                if mode == "Classical" or inferencer is None:
                    x_hat_t, _ = fusion_classical.fuse(y_t, warped_x_prev)
                else:
                    t_net0 = time.perf_counter()
                    y_t_f32 = y_t.astype(np.float32) / 255.0
                    warped_x_prev_f32 = warped_x_prev.astype(np.float32) / 255.0

                    conf_map_lr = compute_confidence_map_numpy(
                        y_t_f32, warped_x_prev_f32, config.scale, config.tau / 255.0
                    )
                    residual_f32 = inferencer.infer(y_t_f32, warped_x_prev_f32, conf_map_lr)
                    y_t_bicubic_f32 = y_t_bicubic.astype(np.float32) / 255.0
                    out_f32 = y_t_bicubic_f32 + residual_f32
                    x_hat_t = np.clip(out_f32 * 255.0, 0, 255).astype(np.uint8)
                    t_net = (time.perf_counter() - t_net0) * 1000

            prev_y_t = y_t.copy()
            x_hat_prev = x_hat_t.copy()

            # Reconstruct colour frames
            bgr_hr = cv2.cvtColor(cv2.merge([x_hat_t, cr_hr, cb_hr]),
                                  cv2.COLOR_YCrCb2BGR)
            bgr_bicubic = cv2.cvtColor(cv2.merge([y_t_bicubic, cr_hr, cb_hr]),
                                       cv2.COLOR_YCrCb2BGR)

            # Panel 3: nearest-neighbour blow-up of the raw LR input
            lr_vis = cv2.resize(bgr_lr, (w_hr, h_hr), interpolation=cv2.INTER_NEAREST)

            # Panel 4: residual heatmap
            if residual_f32 is not None:
                res_vis = np.clip(np.abs(residual_f32) * 5.0 * 255.0, 0, 255).astype(np.uint8)
                res_bgr = cv2.applyColorMap(res_vis, cv2.COLORMAP_JET)
            else:
                res_bgr = np.zeros_like(bgr_hr)

            t1 = time.time()
            fps = 1.0 / (t1 - t0 + 1e-6)

            stats = {
                "fps": fps,
                "flow_ms": flow_ms,
                "net_ms": t_net,
                "mode": mode,
                "lr_res": f"{w_lr}×{h_lr}",
                "hr_res": f"{w_hr}×{h_hr}",
            }

            self.frame_ready.emit(bgr_bicubic, bgr_hr, lr_vis, res_bgr, stats)

            # Throttle to ~60 fps max so we don't flood the GUI
            elapsed = time.time() - t0
            if elapsed < 1 / 60:
                time.sleep(1 / 60 - elapsed)

        cap.stop()


# ═══════════════════════════════════════════════════════════════════════════
#  Main Window
# ═══════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VSR Real-Time — Video Super-Resolution")
        self.setMinimumSize(960, 640)
        self.resize(1280, 780)

        self.worker: VideoWorker | None = None
        self.config = load_config(os.path.join(_PROJECT_ROOT, "configs", "default.yaml"))
        self.onnx_path = ""
        self._cameras = detect_cameras()

        self._build_ui()
        self._connect_signals()

    # ── UI construction ────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 4)
        root_layout.setSpacing(6)

        # ── Top toolbar ────────────────────────────────────────────────────
        toolbar = QFrame()
        toolbar.setFrameShape(QFrame.StyledPanel)
        toolbar.setStyleSheet("QFrame { background: #1e1e2e; border-radius: 8px; }")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(12, 8, 12, 8)
        tb_layout.setSpacing(12)

        # -- Input source --
        grp_input = self._make_group("Input Source")
        gi_layout = QHBoxLayout()
        gi_layout.setSpacing(6)

        self.combo_source = QComboBox()
        self.combo_source.setMinimumWidth(240)
        self.combo_source.setStyleSheet(self._combo_style())
        for idx, label in self._cameras:
            self.combo_source.addItem(label, idx)
        self.combo_source.addItem("📂  Video File…", "file")
        gi_layout.addWidget(self.combo_source)

        self.btn_refresh_cam = QPushButton("🔄")
        self.btn_refresh_cam.setToolTip("Refresh camera list")
        self.btn_refresh_cam.setFixedWidth(32)
        self.btn_refresh_cam.setStyleSheet(self._btn_secondary_style())
        gi_layout.addWidget(self.btn_refresh_cam)

        grp_input.layout().addLayout(gi_layout)
        tb_layout.addWidget(grp_input)

        # -- Model --
        grp_model = self._make_group("ONNX Model")
        gm_layout = QHBoxLayout()
        gm_layout.setSpacing(6)

        self.lbl_model_path = QLabel("No model loaded")
        self.lbl_model_path.setStyleSheet("color: #888; font-size: 12px;")
        self.lbl_model_path.setMinimumWidth(160)
        gm_layout.addWidget(self.lbl_model_path)

        self.btn_browse_model = QPushButton("Browse…")
        self.btn_browse_model.setStyleSheet(self._btn_secondary_style())
        gm_layout.addWidget(self.btn_browse_model)
        grp_model.layout().addLayout(gm_layout)
        tb_layout.addWidget(grp_model)

        # -- Mode toggle --
        grp_mode = self._make_group("Mode")
        gmo_layout = QHBoxLayout()
        gmo_layout.setSpacing(6)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Neural", "Classical"])
        self.combo_mode.setStyleSheet(self._combo_style())
        gmo_layout.addWidget(self.combo_mode)
        grp_mode.layout().addLayout(gmo_layout)
        tb_layout.addWidget(grp_mode)

        # -- Start / Stop --
        self.btn_start = QPushButton("▶  Start")
        self.btn_start.setFixedHeight(38)
        self.btn_start.setStyleSheet(self._btn_start_style())
        tb_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setFixedHeight(38)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(self._btn_stop_style())
        tb_layout.addWidget(self.btn_stop)

        tb_layout.addStretch()

        root_layout.addWidget(toolbar)

        # ── Video display (2×2 grid via QLabels) ──────────────────────────
        display_frame = QFrame()
        display_frame.setFrameShape(QFrame.NoFrame)
        display_frame.setStyleSheet("background: #0e0e16; border-radius: 6px;")
        display_layout = QVBoxLayout(display_frame)
        display_layout.setContentsMargins(4, 4, 4, 4)
        display_layout.setSpacing(2)

        # Top row
        top_row = QHBoxLayout()
        top_row.setSpacing(2)
        self.lbl_bicubic = self._make_video_label("Bicubic Upscale")
        self.lbl_neural = self._make_video_label("Neural / Classical VSR")
        top_row.addWidget(self.lbl_bicubic)
        top_row.addWidget(self.lbl_neural)
        display_layout.addLayout(top_row, stretch=1)

        # Bottom row
        bot_row = QHBoxLayout()
        bot_row.setSpacing(2)
        self.lbl_original = self._make_video_label("Original Input (Nearest)")
        self.lbl_residual = self._make_video_label("Neural Residual Heatmap")
        bot_row.addWidget(self.lbl_original)
        bot_row.addWidget(self.lbl_residual)
        display_layout.addLayout(bot_row, stretch=1)

        root_layout.addWidget(display_frame, stretch=1)

        # ── Status bar ────────────────────────────────────────────────────
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(
            "QStatusBar { background: #1e1e2e; color: #cdd6f4; font-size: 12px; }"
        )
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — select a source and model, then press Start.")

        # ── Global stylesheet ─────────────────────────────────────────────
        self.setStyleSheet("""
            QMainWindow { background: #11111b; }
            QGroupBox { 
                color: #89b4fa; font-size: 11px; font-weight: bold;
                border: 1px solid #313244; border-radius: 6px;
                margin-top: 10px; padding: 6px 8px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        """)

    # ── Signal wiring ──────────────────────────────────────────────────────
    def _connect_signals(self):
        self.btn_browse_model.clicked.connect(self._browse_model)
        self.btn_refresh_cam.clicked.connect(self.refresh_cameras)
        self.combo_source.activated.connect(self._on_source_changed)
        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.combo_mode.currentTextChanged.connect(self._on_mode_changed)

    def refresh_cameras(self):
        """Re-scan connected cameras and update the combo box."""
        curr_data = self.combo_source.currentData()
        self._cameras = detect_cameras()
        self.combo_source.blockSignals(True)
        self.combo_source.clear()
        selected_idx = 0
        for i, (idx, label) in enumerate(self._cameras):
            self.combo_source.addItem(label, idx)
            if idx == curr_data:
                selected_idx = i
        self.combo_source.addItem("📂  Video File…", "file")
        if curr_data == "file":
            selected_idx = self.combo_source.count() - 1
        self.combo_source.setCurrentIndex(selected_idx)
        self.combo_source.blockSignals(False)
        self.status_bar.showMessage(f"Detected {len(self._cameras)} camera(s).", 3000)

    # ── Callbacks ──────────────────────────────────────────────────────────
    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ONNX Model", _PROJECT_ROOT,
            "ONNX Models (*.onnx);;All Files (*)"
        )
        if path:
            self.onnx_path = path
            basename = os.path.basename(path)
            self.lbl_model_path.setText(basename)
            self.lbl_model_path.setToolTip(path)
            self.lbl_model_path.setStyleSheet("color: #a6e3a1; font-size: 12px;")

    def _on_source_changed(self, index):
        data = self.combo_source.currentData()
        if data == "file":
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Video File", "",
                "Video Files (*.mp4 *.avi *.mkv *.mov *.webm);;All Files (*)"
            )
            if path:
                # Insert the file entry before the "Video File…" placeholder
                file_label = f"📄  {os.path.basename(path)}"
                self.combo_source.insertItem(self.combo_source.count() - 1, file_label, path)
                self.combo_source.setCurrentIndex(self.combo_source.count() - 2)
            else:
                # User cancelled — revert to first camera
                if self.combo_source.count() > 1:
                    self.combo_source.setCurrentIndex(0)

    def _on_mode_changed(self, text):
        if self.worker is not None:
            self.worker.set_mode(text)

    def _start(self):
        data = self.combo_source.currentData()
        if data == "file":
            self.status_bar.showMessage("Please select a video file first.")
            return

        source = data  # int (camera index) or str (file path)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.combo_source.setEnabled(False)

        mode = self.combo_mode.currentText()
        self.worker = VideoWorker(source, self.config, self.onnx_path, mode)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.error.connect(self._on_error)
        self.worker.finished_playback.connect(self._on_finished)
        self.worker.start()

        self.status_bar.showMessage(f"Running — {mode} mode  |  Source: {self.combo_source.currentText()}")

    def _stop(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.combo_source.setEnabled(True)
        self.status_bar.showMessage("Stopped.")

    def _on_finished(self):
        """Called when a video file finishes playing."""
        self._stop()
        self.status_bar.showMessage("Playback finished.")

    def _on_error(self, msg):
        self._stop()
        self.status_bar.showMessage(f"Error: {msg}")

    def _on_frame(self, bicubic, neural, original, residual, stats):
        """Receive four BGR panels + stats dict from the worker thread."""
        self._set_panel(self.lbl_bicubic, bicubic, f"Bicubic {self.config.scale}×")
        label2 = f"{stats['mode']} VSR {self.config.scale}×"
        self._set_panel(self.lbl_neural, neural, label2)
        self._set_panel(self.lbl_original, original, "Original (Nearest)")
        hm_label = "Neural Residual" if stats["mode"] == "Neural" else ""
        self._set_panel(self.lbl_residual, residual, hm_label)

        self.status_bar.showMessage(
            f"FPS: {stats['fps']:.1f}  |  Flow: {stats['flow_ms']:.1f} ms  |  "
            f"Net: {stats['net_ms']:.1f} ms  |  LR: {stats['lr_res']}  →  "
            f"HR: {stats['hr_res']}  |  Mode: {stats['mode']}"
        )

    # ── Helpers ────────────────────────────────────────────────────────────
    def _set_panel(self, label: QLabel, bgr: np.ndarray, overlay_text: str = ""):
        """Convert a BGR numpy image to a QPixmap and display it scaled inside a QLabel."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Draw overlay text onto the image itself
        if overlay_text:
            cv2.putText(rgb, overlay_text, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 120), 2,
                        cv2.LINE_AA)

        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        # Scale to fit the label while preserving aspect ratio
        label.setPixmap(pix.scaled(label.size(), Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation))

    @staticmethod
    def _make_video_label(placeholder: str) -> QLabel:
        lbl = QLabel(placeholder)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lbl.setMinimumSize(160, 120)
        lbl.setStyleSheet(
            "QLabel { background: #181825; color: #585b70; "
            "font-size: 14px; border: 1px solid #313244; border-radius: 4px; }"
        )
        return lbl

    @staticmethod
    def _make_group(title: str) -> QGroupBox:
        grp = QGroupBox(title)
        grp.setLayout(QVBoxLayout())
        grp.layout().setContentsMargins(6, 14, 6, 4)
        return grp

    # ── Stylesheet helpers ─────────────────────────────────────────────────
    @staticmethod
    def _combo_style() -> str:
        return """
            QComboBox {
                background: #181825; color: #cdd6f4; border: 1px solid #45475a;
                border-radius: 4px; padding: 4px 8px; min-height: 24px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #1e1e2e; color: #cdd6f4; selection-background-color: #45475a;
            }
        """

    @staticmethod
    def _btn_secondary_style() -> str:
        return """
            QPushButton {
                background: #313244; color: #cdd6f4; border: none;
                border-radius: 4px; padding: 6px 14px; font-weight: bold;
            }
            QPushButton:hover { background: #45475a; }
        """

    @staticmethod
    def _btn_start_style() -> str:
        return """
            QPushButton {
                background: #a6e3a1; color: #1e1e2e; border: none;
                border-radius: 6px; padding: 6px 20px; font-weight: bold; font-size: 13px;
            }
            QPushButton:hover { background: #94e2d5; }
            QPushButton:disabled { background: #45475a; color: #6c7086; }
        """

    @staticmethod
    def _btn_stop_style() -> str:
        return """
            QPushButton {
                background: #f38ba8; color: #1e1e2e; border: none;
                border-radius: 6px; padding: 6px 20px; font-weight: bold; font-size: 13px;
            }
            QPushButton:hover { background: #eba0ac; }
            QPushButton:disabled { background: #45475a; color: #6c7086; }
        """

    # ── Window close ───────────────────────────────────────────────────────
    def closeEvent(self, event):
        self._stop()
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════
def main():
    try:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")

        # Dark palette for native widgets that don't get stylesheet overrides
        from PyQt5.QtGui import QPalette, QColor
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#11111b"))
        palette.setColor(QPalette.WindowText, QColor("#cdd6f4"))
        palette.setColor(QPalette.Base, QColor("#181825"))
        palette.setColor(QPalette.AlternateBase, QColor("#1e1e2e"))
        palette.setColor(QPalette.ToolTipBase, QColor("#1e1e2e"))
        palette.setColor(QPalette.ToolTipText, QColor("#cdd6f4"))
        palette.setColor(QPalette.Text, QColor("#cdd6f4"))
        palette.setColor(QPalette.Button, QColor("#313244"))
        palette.setColor(QPalette.ButtonText, QColor("#cdd6f4"))
        palette.setColor(QPalette.Highlight, QColor("#89b4fa"))
        palette.setColor(QPalette.HighlightedText, QColor("#1e1e2e"))
        app.setPalette(palette)

        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        err = traceback.format_exc()
        try:
            with open("crash_log.txt", "w", encoding="utf-8") as f:
                f.write(err)
        except Exception:
            pass
        try:
            QMessageBox.critical(None, "VSR Real-Time Error", f"Failed to start VSR Real-Time:\n\n{err}")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
