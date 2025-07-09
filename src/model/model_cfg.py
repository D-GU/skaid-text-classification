from dataclasses import dataclass

from torch.cuda import is_available


@dataclass
class ModelCFG:
    device: str = "cuda" if is_available() else "cpu"
    device_id: int = 1 if device == "cuda" else 0
    accelerator: str = "gpu" if device == "cuda" else "cpu"
    precision: int = 16

    input_size: int = 117881
    hidden_size: int = 512
    dropout: float = 0.3
    lr: float = 1e-3
    batch_size: int = 32
    num_workers: int = 4
    num_epochs: int = 5
    num_cls: int = 20


cfg = ModelCFG()
