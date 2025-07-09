import torch
from pytorch_lightning import Trainer

from src.document_handler.parser import Parser
from src.document_handler.text_preprocessor import TextPreprocessor
from src.model.model import TextClassifier
from src.model.model_cfg import ModelCFG

cfg = ModelCFG()


def train(_model, _filename: str):
    # Перемножение матриц с уменьшенной точностью для увеличения производительности
    torch.set_float32_matmul_precision("high")

    trainer = Trainer(
        max_epochs=cfg.num_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.device_id,
        precision=cfg.precision,
        accumulate_grad_batches=2,
        enable_progress_bar=True
    )

    # Обучаем модель
    trainer.fit(_model)

    # Проверяем ее на тестовом датасете
    results = trainer.test(_model)
    print(results)

    # Сохраняем все параметры модели
    _model = _model.to(cfg.device)
    torch.save(_model.state_dict(), _filename)
