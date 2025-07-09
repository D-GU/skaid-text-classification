from dataclasses import dataclass, asdict

import torch


@dataclass
class ModelCFG:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_size: int = 128
    dropout: float = 0.2
    lr: float = 1e-2
    batch_size: int = 32
    num_workers: int = 4
    num_epochs: int = 20


cfg = ModelCFG()
cfg_dict = asdict(cfg)
