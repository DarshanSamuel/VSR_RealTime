import cv2
import threading
import time
import logging

class VideoSource:
    def __init__(self, source_path_or_index):
        self.source = source_path_or_index
        
        # Check if source is a live stream (webcam or RTSP/HTTP)
        self.is_live = False
        if isinstance(self.source, int):
            self.is_live = True
        elif isinstance(self.source, str):
            if self.source.startswith(('rtsp://', 'http://', 'https://')):
                self.is_live = True
                
        # Open video capture
        if self.is_live and isinstance(self.source, str) and self.source.startswith('rtsp'):
            # Prefer TCP for RTSP to minimize latency and dropped packets
            import os
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|analyzeduration;0|probesize;32"
            
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {self.source}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0 or self.fps != self.fps:
            self.fps = 30.0 # Default fallback
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.latest_frame = None
        self.ret = False
        self.running = True
        
        if self.is_live:
            # Start background thread to keep only the newest frame
            self.lock = threading.Lock()
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            # Wait for first frame
            time.sleep(0.1)
            
    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                if ret:
                    self.latest_frame = frame
            if not ret:
                break
                
    def read(self):
        if self.is_live:
            with self.lock:
                # Return a copy to avoid race conditions during processing
                if self.latest_frame is not None:
                    return self.ret, self.latest_frame.copy()
                return self.ret, None
        else:
            # For files, read sequentially without dropping frames
            return self.cap.read()
            
    def stop(self):
        self.running = False
        if self.is_live:
            self.thread.join(timeout=1.0)
        self.cap.release()

    def __del__(self):
        self.stop()

if __name__ == "__main__":
    # Test script
    import sys
    source_idx = 0 if len(sys.argv) == 1 else sys.argv[1]
    
    try:
        source_idx = int(source_idx)
    except ValueError:
        pass
        
    print(f"Opening source: {source_idx}")
    src = VideoSource(source_idx)
    print(f"Resolution: {src.width}x{src.height} @ {src.fps} FPS")
    
    start_time = time.time()
    frames = 0
    while True:
        ret, frame = src.read()
        if not ret or frame is None:
            if not src.is_live:
                break # end of file
            continue
            
        cv2.imshow("Capture Test", frame)
        frames += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    end_time = time.time()
    print(f"Processed {frames} frames in {end_time - start_time:.2f} seconds")
    print(f"Effective FPS: {frames / (end_time - start_time):.2f}")
    
    src.stop()
    cv2.destroyAllWindows()
