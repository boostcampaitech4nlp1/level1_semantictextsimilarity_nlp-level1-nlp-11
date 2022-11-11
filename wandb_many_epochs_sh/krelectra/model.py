import pytorch_lightning as pl
import torch
import torchmetrics
import transformers
import wandb
from torch.optim.lr_scheduler import ExponentialLR

class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1
        )
        self.loss_func = torch.nn.L1Loss()

    def forward(self, x):
        x = self.plm(x)["logits"]

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)
        wandb.log({"train_loss": loss})

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())

        self.log("val_loss", loss)
        wandb.log({"val_loss": loss})

        self.log(
            "val_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )
        wandb.log(
            {
                "val_pearson": torchmetrics.functional.pearson_corrcoef(
                    logits.squeeze(), y.squeeze()
                )
            }
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log(
            "test_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )
        wandb.log(
            {
                "test_pearson": torchmetrics.functional.pearson_corrcoef(
                    logits.squeeze(), y.squeeze()
                )
            }
        )

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.5)
        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
        }
