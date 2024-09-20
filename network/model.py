from typing import Any
from lightning import LightningModule
from torch.optim import Adam, AdamW, SGD, lr_scheduler
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics import Accuracy, F1Score

import torch.nn as nn
import torch

class IntelImageClassificationModel(LightningModule):
    def __init__(
        self, 
        lr: float,
        num_class: int | None = None,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        class_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_class)
        
        print("Using class weights:", class_weights)
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights) if class_weights else None)
        
        self.tr_accu = Accuracy(task="multiclass", num_classes=num_class)
        self.val_accu = Accuracy(task="multiclass", num_classes=num_class)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_class, average='macro')
        self.test_accu = Accuracy(task="multiclass", num_classes=num_class)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_class, average='macro')

        self.save_hyperparameters()
        
    def forward(self, x):
        return self.backbone(x)
    
    def _share_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        return loss, logits
    
    def training_step(self, batch, batch_idx):
        loss, logits = self._share_step(batch, batch_idx)
        self.log_dict({
            "train_loss": loss, 
            "train_accu": self.tr_accu(logits, batch[1])
        }, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, logits = self._share_step(batch, batch_idx)
        self.log_dict({
            "val_loss": loss, 
            "val_accu": self.val_accu(logits, batch[1]),
            "val_f1": self.val_f1(logits, batch[1])
        }, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        self.log_dict({
            "test_accu": self.test_accu(logits, labels),
            "test_f1": self.test_f1(logits, labels)
        }, prog_bar=False)
        
    def predict_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        preds = logits.argmax(dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.hparams.lr,
                    total_steps=self.trainer.estimated_stepping_batches
                ),
                "interval": "step",
            }
        }