from config.base import BaseConfig

class RunConfig(BaseConfig):
    checkpoint_path: str = ""
    version: str = "0.0"

    total_epoches: int = 1000
    batch_size: int = 64

    weight_norm: float = 1e-5
    moments: float = 0.9
    lr: float = 1e-4
    lr_decay_rate: float = 0.5
    lr_decay_steps: list = [50, 100]