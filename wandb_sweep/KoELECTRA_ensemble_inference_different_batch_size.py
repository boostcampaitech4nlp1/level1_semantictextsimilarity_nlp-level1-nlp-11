import argparse
import os
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
import transformers
from torch.optim.lr_scheduler import ExponentialLR
from tqdm.auto import tqdm

SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, idx):
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(
        self,
        model_name,
        batch_size,
        shuffle,
        train_path,
        dev_path,
        test_path,
        predict_path,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = 4
        self.pin_memory = True

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, max_length=100
        )
        self.target_columns = ["label"]
        self.delete_columns = ["id"]
        self.text_columns = ["sentence_1", "sentence_2"]

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(
            dataframe.iterrows(), desc="tokenizing", total=len(dataframe)
        ):
            text = "[SEP]".join(
                [item[text_column] for text_column in self.text_columns]
            )
            outputs = self.tokenizer(
                text, add_special_tokens=True, padding="max_length", truncation=True
            )
            data.append(outputs["input_ids"])
        return data

    def preprocessing(self, data):
        data = data.drop(columns=self.delete_columns)

        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []

        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage="fit"):
        if stage == "fit":
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            train_inputs, train_targets = self.preprocessing(train_data)
            val_inputs, val_targets = self.preprocessing(val_data)

            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=args.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


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

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log(
            "val_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log(
            "test_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", default="monologg/koelectra-base-v3-discriminator", type=str
    )
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--batch_size2", default=16, type=int)
    parser.add_argument("--max_epoch", default=1, type=int)
    parser.add_argument("--shuffle", default=True)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--train_path", default="../../data/train.csv")
    parser.add_argument("--dev_path", default="../../data/dev.csv")
    parser.add_argument("--test_path", default="../../data/dev.csv")
    parser.add_argument("--predict_path", default="../../data/test.csv")
    args = parser.parse_args(args=[])

    dataloader = Dataloader(
        args.model_name,
        args.batch_size,
        args.shuffle,
        args.train_path,
        args.dev_path,
        args.test_path,
        args.predict_path,
    )
    dataloader2 = Dataloader(
        args.model_name,
        args.batch_size2,
        args.shuffle,
        args.train_path,
        args.dev_path,
        args.test_path,
        args.predict_path,
    )

    trainer = pl.Trainer(gpus=1, max_epochs=args.max_epoch, log_every_n_steps=1)

    # Inference part
    # model = torch.load("koelectra_wandb_sweep.pt")

    # load checkpoint
    model = Model.load_from_checkpoint(
        "./dongchan-sweep-tuning/my6q4bjk/checkpoints/epoch18-val_pearson0.9201.ckpt"
    )
    model2 = Model.load_from_checkpoint(
        "./dongchan-sweep-tuning/l0ejtnfl/checkpoints/epoch12-val_pearson0.9179.ckpt"
    )

    # test checkpoint for reproducing val_pearson
    # trainer.test(model=model, datamodule=dataloader)
    # trainer.test(model=model2, datamodule=dataloader2)

    # Make output
    predictions = trainer.predict(model=model, datamodule=dataloader)
    predictions = torch.cat(predictions)
    predictions2 = trainer.predict(model=model2, datamodule=dataloader2)
    predictions2 = torch.cat(predictions2)

    final_predictions = [
        (predictions[i] + predictions2[i]) / 2 for i in range(len(predictions))
    ]
    final_predictions = list(round(float(i), 1) for i in final_predictions)

    output = pd.read_csv("../../data/sample_submission.csv")
    output["target"] = final_predictions
    output.to_csv("KoELECTRA_model_ensemble_sweep.csv", index=False)
