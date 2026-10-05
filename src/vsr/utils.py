import yaml
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    scale: int = 2
    channels: int = 16
    blocks: int = 2
    alpha: float = 0.8
    tau: float = 12.0
    use_back_projection: bool = True
    lambda_temporal: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 16
    epochs: int = 100
    motion_estimator: str = "dis_fast"
    warp_interpolation: str = "bicubic"
    device: str = "cpu"

def load_config(path: str) -> Config:
    if not Path(path).exists():
        return Config() # defaults
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    return Config(**{k: v for k, v in data.items() if hasattr(Config, k)})
