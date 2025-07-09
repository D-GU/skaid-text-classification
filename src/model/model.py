import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall

from src.document_handler.vectorizer import Vectorizer
from src.model.dataset import NewsgroupsDataset
from src.model.model_cfg import ModelCFG


class TextClassifier(pl.LightningModule):
    def __init__(self, cfg: ModelCFG):
        super().__init__()

        self.vectorizer = Vectorizer()

        # Создаем простую линейную модель
        self.net = nn.Sequential(
            nn.Linear(cfg.input_size, cfg.hidden_size),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size, cfg.num_cls),
        )

        # Определим метрики классификации
        self.accuracy = Accuracy(task="multiclass", num_classes=cfg.num_cls, average="macro")
        self.precision = Precision(task="multiclass", num_classes=cfg.num_cls, average='macro')
        self.recall = Recall(task="multiclass", num_classes=cfg.num_cls, average='macro')

        # Определим функцию потери для задачи мультиклассовой классификации
        self.loss = nn.CrossEntropyLoss()

        # Определим гиперпараметры
        self.lr = cfg.lr
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        pred = torch.argmax(logits, dim=1)

        accuracy = self.accuracy(pred, y)
        precision = self.precision(pred, y)
        recall = self.recall(pred, y)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_accuracy", accuracy, prog_bar=True, on_epoch=True)
        self.log("train_precision", precision, prog_bar=True, on_epoch=True)
        self.log("train_recall", recall, prog_bar=True, on_epoch=True)

        return loss

    def train_dataloader(self):
        dataset = NewsgroupsDataset(
            vectorizer=self.vectorizer,
            subset="train",
        )

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        dataset = NewsgroupsDataset(
            vectorizer=self.train_dataloader().dataset.vectorizer,
            subset="test",  # переключаемся на тестовый набор
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        prec = self.precision(preds, y)
        rec = self.recall(preds, y)

        # Логируем на эпоху
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_accuracy", acc, prog_bar=True, on_epoch=True)
        self.log("test_precision", prec, prog_bar=True, on_epoch=True)
        self.log("test_recall", rec, prog_bar=True, on_epoch=True)