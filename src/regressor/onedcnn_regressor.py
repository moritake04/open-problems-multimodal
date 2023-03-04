import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


def correlation_score(y_true, y_pred):
    if type(y_true) == pd.DataFrame:
        y_true = y_true.values
    if type(y_pred) == pd.DataFrame:
        y_pred = y_pred.values
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)


def partial_correlation_score_torch_faster(y_true, y_pred):
    """Compute the correlation between each rows of the y_true and y_pred tensors.
    Compatible with backpropagation.
    """
    y_true_centered = y_true - torch.mean(y_true, dim=1)[:, None]
    y_pred_centered = y_pred - torch.mean(y_pred, dim=1)[:, None]
    cov_tp = torch.sum(y_true_centered * y_pred_centered, dim=1) / (y_true.shape[1] - 1)
    var_t = torch.sum(y_true_centered ** 2, dim=1) / (y_true.shape[1] - 1)
    var_p = torch.sum(y_pred_centered ** 2, dim=1) / (y_true.shape[1] - 1)
    return cov_tp / torch.sqrt(var_t * var_p)


def correl_loss(pred, tgt):
    """Loss for directly optimizing the correlation."""
    return -torch.mean(partial_correlation_score_torch_faster(tgt, pred))


class OneDCNN(pl.LightningModule):
    def __init__(self, input_size, output_size, cfg):
        super().__init__()
        print(f"input size: {input_size}, output size: {output_size}")
        # hidden_size = cfg["hidden_size"]
        self.cfg = cfg
        if cfg["criterion"] == "correl_loss":
            self.criterion = correl_loss
        else:
            self.criterion = nn.__dict__[cfg["criterion"]]()

        if cfg["batchnorm"]:
            self.dense1 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(input_size, 4096)),
                nn.BatchNorm1d(4096),
                nn.Mish(),
                nn.Dropout(cfg["dropout"]),
                nn.Unflatten(1, (256, 16)),
            )
            self.conv1 = nn.Sequential(
                nn.utils.weight_norm(
                    nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2, bias=False),
                    dim=None,
                ),
                nn.BatchNorm1d(512),
                nn.Mish(),
                nn.Dropout(cfg["dropout"]),
            )
            self.avg_po = nn.AdaptiveAvgPool1d(8)
            self.conv2 = nn.Sequential(
                nn.utils.weight_norm(
                    nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1), dim=None
                ),
                nn.BatchNorm1d(512),
                nn.Mish(),
                nn.Dropout(cfg["dropout"]),
            )
            self.conv2_1 = nn.Sequential(
                nn.utils.weight_norm(
                    nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1), dim=None
                ),
                nn.BatchNorm1d(512),
                nn.Mish(),
                nn.Dropout(cfg["dropout"]),
            )
            self.conv2_2 = nn.Sequential(
                nn.utils.weight_norm(
                    nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2), dim=None
                ),
                nn.BatchNorm1d(512),
                nn.Mish(),
                nn.Dropout(cfg["dropout"]),
            )
        else:
            self.dense1 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(input_size, 4096)),
                nn.Mish(),
                nn.Dropout(cfg["dropout"]),
                nn.Unflatten(1, (256, 16)),
            )
            self.conv1 = nn.Sequential(
                nn.utils.weight_norm(
                    nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2, bias=False),
                    dim=None,
                ),
                nn.Mish(),
                nn.Dropout(cfg["dropout"]),
            )
            self.avg_po = nn.AdaptiveAvgPool1d(8)
            self.conv2 = nn.Sequential(
                nn.utils.weight_norm(
                    nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1), dim=None
                ),
                nn.Mish(),
                nn.Dropout(cfg["dropout"]),
            )
            self.conv2_1 = nn.Sequential(
                nn.utils.weight_norm(
                    nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1), dim=None
                ),
                nn.Mish(),
                nn.Dropout(cfg["dropout"]),
            )
            self.conv2_2 = nn.Sequential(
                nn.utils.weight_norm(
                    nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2), dim=None
                ),
                nn.Mish(),
                nn.Dropout(cfg["dropout"]),
            )
        self.max_po = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.dense2 = nn.utils.weight_norm(nn.Linear(2048, output_size))

    def forward(self, x):
        x = self.dense1(x)
        x = self.conv1(x)
        x = self.avg_po(x)
        x = self.conv2(x)
        x_s = x
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = x * x_s
        x = self.max_po(x)
        x = self.flatten(x)
        x = self.dense2(x)
        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        pred_y = self.forward(X)
        loss = self.criterion(pred_y, y)
        return loss

    def training_epoch_end(self, outputs):
        loss_list = [x["loss"] for x in outputs]
        avg_loss = torch.stack(loss_list).mean()
        self.log("train_avg_loss", avg_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        X, y = batch
        pred_y = self.forward(X)
        loss = self.criterion(pred_y, y)
        return {"valid_loss": loss}

    def validation_epoch_end(self, outputs):
        loss_list = [x["valid_loss"] for x in outputs]
        avg_loss = torch.stack(loss_list).mean()
        self.log("valid_avg_loss", avg_loss, prog_bar=True)
        return avg_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X, _ = batch
        pred_y = self.forward(X)
        return pred_y

    def configure_optimizers(self):
        optimizer = optim.__dict__[self.cfg["optimizer"]["name"]](
            self.parameters(), **self.cfg["optimizer"]["params"]
        )
        if self.cfg["scheduler"] is None:
            return [optimizer]
        else:
            if self.cfg["scheduler"]["name"] == "OneCycleLR":
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    steps_per_epoch=self.cfg["len_train_loader"],
                    **self.cfg["scheduler"]["params"],
                )
                scheduler = {"scheduler": scheduler, "interval": "step"}
            elif self.cfg["scheduler"]["name"] == "ReduceLROnPlateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, **self.cfg["scheduler"]["params"],
                )
                scheduler = {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "valid_avg_loss",
                }
            else:
                scheduler = optim.lr_scheduler.__dict__[self.cfg["scheduler"]["name"]](
                    optimizer, **self.cfg["scheduler"]["params"]
                )
            return [optimizer], [scheduler]


class TableDataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None):
        if y is None:
            self.targets = False
            self.X = X
            self.y = torch.zeros(self.X.shape[0], dtype=torch.float32)
        else:
            self.targets = True
            self.X = X
            self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        if self.targets:
            X = torch.tensor(self.X[index].todense(), dtype=torch.float32)[0]
            y = torch.tensor(self.y[index].todense(), dtype=torch.float32)[0]
        else:
            X = torch.tensor(self.X[index].todense(), dtype=torch.float32)[0]
            y = self.y[index]
        return X, y


class OneDCNNRegressor:
    def __init__(self, cfg, train_X, train_y, valid_X=None, valid_y=None):
        input_size = train_X.shape[1]
        output_size = train_y.shape[1]
        self.train_dataset = TableDataset(train_X, train_y)
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, **cfg["train_loader"],
        )
        cfg["len_train_loader"] = len(self.train_dataloader)
        if valid_X is None:
            self.valid_dataloader = None
        else:
            self.valid_dataset = TableDataset(valid_X, valid_y)
            self.valid_dataloader = torch.utils.data.DataLoader(
                self.valid_dataset, **cfg["valid_loader"],
            )

        self.callbacks = [pl.callbacks.LearningRateMonitor(logging_interval="step")]

        if cfg["early_stopping"] is not None:
            self.callbacks.append(
                pl.callbacks.EarlyStopping(
                    "valid_avg_loss", patience=cfg["early_stopping"]["patience"],
                )
            )

        if cfg["model_save"]:
            self.callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=f"../weights/{cfg['general']['save_name']}",
                    filename=f"fold{cfg['fold_n']}",
                    save_weights_only=True,
                    monitor="valid_avg_loss"
                    if cfg["early_stopping"] is not None
                    else None,
                )
            )

        self.logger = WandbLogger(
            project=cfg["general"]["project_name"],
            name=f"{cfg['general']['save_name']}_{cfg['fold_n']}",
            group=f"{cfg['general']['save_name']}_cv"
            if cfg["general"]["cv"]
            else "all",
            job_type=cfg["job_type"],
            mode="disabled" if cfg["general"]["wandb_desabled"] else "online",
            config=cfg,
        )

        self.model = OneDCNN(input_size, output_size, cfg)
        self.cfg = cfg

        self.trainer = Trainer(
            callbacks=self.callbacks, logger=self.logger, **self.cfg["pl_params"]
        )

    def train(self, weight_path=None):
        if self.valid_dataloader is None:
            self.trainer.fit(
                self.model,
                train_dataloaders=self.train_dataloader,
                ckpt_path=weight_path,
            )
        else:
            self.trainer.fit(
                self.model,
                train_dataloaders=self.train_dataloader,
                val_dataloaders=self.valid_dataloader,
                ckpt_path=weight_path,
            )

    def predict(self, test_X, weight_path=None):
        preds = []
        test_dataset = TableDataset(test_X)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, **self.cfg["test_loader"],
        )
        preds = self.trainer.predict(
            self.model,
            dataloaders=test_dataloader,
            ckpt_path="best" if self.cfg["early_stopping"] is not None else weight_path,
        )
        preds = torch.cat(preds, axis=0)
        preds = preds.cpu().detach().numpy()
        return preds

    def load_weight(self, weight_path):
        self.model.model = self.model.model.load_from_checkpoint(
            checkpoint_path=weight_path, cfg=self.cfg,
        )
        print(f"loaded model ({weight_path})")


class OneDCNNRegressorInference:
    def __init__(self, cfg, input_size, output_size, weight_path=None):
        self.model = OneDCNN(input_size, output_size, cfg)
        self.weight_path = weight_path
        self.cfg = cfg
        self.trainer = Trainer(**self.cfg["pl_params"])

    def predict(self, test_X):
        preds = []
        test_dataset = TableDataset(test_X)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, **self.cfg["test_loader"],
        )
        preds = self.trainer.predict(
            self.model,
            dataloaders=test_dataloader,
            ckpt_path="best"
            if self.cfg["early_stopping"] is not None
            else self.weight_path,
        )
        preds = torch.cat(preds, axis=0)
        preds = preds.cpu().detach().numpy()
        return preds
