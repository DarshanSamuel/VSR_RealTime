import torch
import argparse
from pathlib import Path

from src.vsr.model import RecurrentVSR
from src.vsr.utils import load_config

def export_to_onnx(config_path, checkpoint_path, output_path, quantize=False):
    config = load_config(config_path)
    
    # Instantiate model
    model = RecurrentVSR(scale=config.scale, channels=config.channels, num_blocks=config.blocks)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Warning: No valid checkpoint provided, exporting untrained model.")
        
    model.eval()
    
    # Dummy inputs
    # Shape: (Batch, Channels, H, W)
    # Let's assume a dummy resolution of 320x240 for export, 
    # though ONNX can support dynamic axes if configured.
    dummy_lr = torch.randn(1, 1, 240, 320)
    dummy_warped_hr = torch.randn(1, 1, 240 * config.scale, 320 * config.scale)
    dummy_conf_map = torch.randn(1, 1, 240, 320)
    
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Export
    dynamic_axes = {
        'lr': {0: 'batch_size', 2: 'height', 3: 'width'},
        'warped_hr_prev': {0: 'batch_size', 2: 'hr_height', 3: 'hr_width'},
        'conf_map': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'hr_height', 3: 'hr_width'}
    }
    
    torch.onnx.export(
        model, 
        (dummy_lr, dummy_warped_hr, dummy_conf_map), 
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['lr', 'warped_hr_prev', 'conf_map'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    print(f"Exported ONNX model to {output_path}")
    
    if quantize:
        print("Starting dynamic INT8 quantization (Experimental)...")
        try:
            import onnx
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            quantized_path = str(output_path).replace('.onnx', '_int8.onnx')
            quantize_dynamic(output_path, quantized_path, weight_type=QuantType.QUInt8)
            print(f"Exported quantized ONNX model to {quantized_path}")
            print("Note: Quantization error can accumulate through recurrence. Check on long sequences!")
        except ImportError:
            print("onnxruntime not found. Skipping quantization.")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--out", default="outputs/vsr_model.onnx")
    parser.add_argument("--quantize", action="store_true", help="Experimental INT8 dynamic quantization")
    args = parser.parse_args()
    
    export_to_onnx(args.config, args.checkpoint, args.out, args.quantize)
