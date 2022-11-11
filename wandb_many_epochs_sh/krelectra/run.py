import argparse

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from dataloader import Dataloader
from model import Model

from seed import seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", default="snunlp/KR-ELECTRA-discriminator", type=str
    )
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--shuffle", default=True)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--train_path", default="../../../data/train.csv")
    parser.add_argument("--dev_path", default="../../../data/dev.csv")
    parser.add_argument("--test_path", default="../../../data/dev.csv")
    parser.add_argument("--predict_path", default="../../../data/test.csv")
    parser.add_argument("--output_path", default="output.csv")
    parser.add_argument('--pretrained', type=str, default='', help='pretrained checkpoint file path')
    parser.add_argument('--wandb_project', default='yhkee0404')
    parser.add_argument('--wandb_name', default='10_KrELECTRA_ep80_bs16')
    parser.add_argument("mode", choices=("train", "predict"))
    args = parser.parse_args()
    
    print(args)
    
    seed()
    
    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(
        args.model_name,
        args.batch_size,
        args.shuffle,
        args.train_path,
        args.dev_path,
        args.test_path,
        args.predict_path,
    )
    if args.pretrained:
        # load checkpoint
        model = Model.load_from_checkpoint(
            args.pretrained
        )
    else:
        model = Model(args.model_name, args.learning_rate)
    
    if args.mode == 'train':
        wandb.init(name=args.wandb_name, project=args.wandb_project)
        wandb.config.update(args)

        wandb_logger = WandbLogger(project=args.wandb_project)
        # early_stop_callback = EarlyStopping(
        #     monitor="val_pearson", mode="max", patience=3, min_delta=0.00, verbose=False
        # )
        checkpoint_callback = ModelCheckpoint(
            filename="epoch{epoch}-val_pearson{val_pearson:.4f}",
            monitor="val_pearson",
            save_top_k=5,
            mode="max",
            auto_insert_metric_name=False,
        )
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=args.max_epochs,
            log_every_n_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )

        # Train part
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)
        
        # 학습이 완료된 모델을 저장합니다.
        # torch.save(model, "KrELECTRA_ep80_bs16_lr1e_5.pt")
    else:
        # Inference part
        trainer = pl.Trainer(gpus=1, max_epochs=args.max_epochs, log_every_n_steps=1)
    
        # test checkpoint for reproducing val_pearson
        # trainer.test(model=model, datamodule=dataloader)
        
        # Make output
        predictions = trainer.predict(model=model, datamodule=dataloader)
        predictions = list(round(float(i), 1) + 0 for i in torch.cat(predictions))

        output = pd.read_csv("../../../data/sample_submission.csv")
        output["target"] = predictions
        output.to_csv(args.output_path, index=False)