# Fardin Rastakhiz @ 2023

from typing import Any
import torch
import torchmetrics
import lightning as L
from abc import abstractmethod

from Scripts.Models.LossFunctions.HeteroLossFunctions import HeteroLossArgs, MulticlassHeteroLoss1

class BaseLightningModel(L.LightningModule):

    def __init__(self, model, optimizer=None, loss_func=None, learning_rate=0.01, batch_size=64, lr_scheduler=None, user_lr_scheduler=False, min_lr=0.0):
        super(BaseLightningModel, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = model
        self.min_lr = min_lr
        # self.save_hyperparameters(ignore=["model"])
        self.save_hyperparameters("model", logger=False)
        self.optimizer = self._get_optimizer(optimizer)
        self.lr_scheduler = self._get_lr_scheduler(lr_scheduler) if user_lr_scheduler else None
        self.loss_func = self._get_loss_func(loss_func)

    def forward(self, data_batch, *args, **kwargs):
        return self.model(data_batch)

    def on_train_epoch_start(self) -> None:
        param_groups = next(iter(self.optimizer.param_groups))
        if 'lr' in param_groups and param_groups['lr'] is not None:
            current_learning_rate = float(param_groups['lr'])
            self.log('lr', current_learning_rate, batch_size=self.batch_size, on_epoch=True, on_step=False)
    
    def training_step(self, data_batch, *args, **kwargs):
        data, labels = data_batch
        data = data.to(self.device)
        labels = labels.to(self.device)
        out_features = self(data)
        if type(out_features) is tuple:
            out_features = out_features[0]
        loss = self.loss_func(out_features, labels.view(out_features.shape))
        
        self.log('train_loss', loss, prog_bar=True, batch_size=self.batch_size, on_epoch=True, on_step=True)
        return loss, out_features

    def validation_step(self, data_batch, *args, **kwargs):
        data, labels = data_batch
        data = data.to(self.device)
        labels = labels.to(self.device)
        out_features = self(data)
        if type(out_features) is tuple:
            out_features = out_features[0]
        loss = self.loss_func(out_features, labels.view(out_features.shape))
        self.log('val_loss', loss, prog_bar=True, batch_size=self.batch_size, on_epoch=True, on_step=True)
        return out_features

    def predict_step(self, data_batch, *args: Any, **kwargs: Any) -> Any:
        data, labels = data_batch
        data = data.to(self.device)
        labels = labels.to(self.device)
        return self(data)

    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return self.optimizer
        
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

    def update_learning_rate(self, learning_rate: float):
        self.learning_rate = learning_rate
        for g in self.optimizer.param_groups:
            g['lr'] = learning_rate

    def _get_optimizer(self, optimizer):
        return optimizer if optimizer is not None else torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def _get_lr_scheduler(self, lr_scheduler):
        return lr_scheduler if lr_scheduler is not None else torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5, mode='min', min_lr=self.min_lr)
            

    @abstractmethod
    def _get_loss_func(self, loss_func):
        pass


class HeteroMultiClassLightningModel(BaseLightningModel):

    def __init__(self, model, num_classes, optimizer=None, loss_func=None, learning_rate=0.01, batch_size=64, lr_scheduler=None, user_lr_scheduler=False, min_lr=0.0):
        super(HeteroMultiClassLightningModel, self).__init__(model, optimizer, loss_func, learning_rate, batch_size=batch_size, lr_scheduler=lr_scheduler, user_lr_scheduler=user_lr_scheduler, min_lr=min_lr)
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def training_step(self, data_batch, *args, **kwargs):
        data, labels = data_batch
        data.to(self.device)
        labels = labels.to(self.device)
        out_features = self(data)
        h_out_features = HeteroLossArgs(out_features[0], out_features[1])
        label_features = HeteroLossArgs(labels, data.x_dict)
        loss = self.loss_func(h_out_features, label_features)
        self.log('train_loss', loss, batch_size=self.batch_size, prog_bar=True, on_epoch=True, on_step=True)
        self.train_acc(torch.argmax(out_features[0], dim=1), torch.argmax(labels, dim=1))
        self.log('train_acc', self.train_acc, prog_bar=True, on_epoch=True, on_step=True)
        
        data.to('cpu')
        return loss

    def validation_step(self, data_batch, *args, **kwargs):
        data, labels = data_batch
        data.to(self.device)
        labels = labels.to(self.device)
        out_features = self(data)
        h_out_features = HeteroLossArgs(out_features[0], out_features[1])
        label_features = HeteroLossArgs(labels, data.x_dict)
        loss = self.loss_func(h_out_features, label_features)
        self.log('val_loss', loss, batch_size=self.batch_size, on_epoch=True, on_step=False)
        self.val_acc(torch.argmax(out_features[0], dim=1), torch.argmax(labels, dim=1))
        self.log('val_acc', self.val_acc, prog_bar=True, on_epoch=True, on_step=False)
        data.to('cpu')

    def _get_loss_func(self, loss_func):
        return loss_func \
            if loss_func is not None else \
            MulticlassHeteroLoss1('word')