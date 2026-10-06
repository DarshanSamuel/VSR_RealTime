import onnxruntime as ort
import numpy as np

class ONNXInferencer:
    def __init__(self, model_path: str):
        # Enforce CPU execution provider as requested and optimize threading
        opts = ort.SessionOptions()
        # Prevent thread thrashing on many-core CPUs (like Ryzen) for small models
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        
    def infer(self, lr: np.ndarray, warped_hr_prev: np.ndarray, conf_map: np.ndarray) -> np.ndarray:
        """
        Inputs:
            lr: (H, W) float32
            warped_hr_prev: (H*s, W*s) float32
            conf_map: (H, W) float32
        Returns:
            hr: (H*s, W*s) float32
        """
        # Expand dims for batch and channel (1, 1, H, W)
        lr_in = np.expand_dims(lr, axis=(0, 1))
        warped_in = np.expand_dims(warped_hr_prev, axis=(0, 1))
        conf_in = np.expand_dims(conf_map, axis=(0, 1))
        
        inputs = {
            'lr': lr_in,
            'warped_hr_prev': warped_in,
            'conf_map': conf_in
        }
        
        # Run inference
        outputs = self.session.run(None, inputs)
        
        # Squeeze batch and channel
        hr_out = np.squeeze(outputs[0], axis=(0, 1))
        return hr_out
