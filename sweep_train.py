# First configure wandb variables in the command line 
# wandb sweep sweep_config.yaml
# wandb agent YOUR_SWEEP_ID


import os
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from models.cnn_model import FlexibleCNN
from models.lightning_model import LitClassifier
from data.datamodule import INatDataModule

def train_wandb():
    wandb.init()
    config = wandb.config

    # Dataset
    datamodule = INatDataModule(
        data_dir='inaturalist',
        batch_size=config.batch_size,
        augment=config.augment
    )
    datamodule.setup()
    num_classes = len(datamodule.classes)

    # Model
    cnn = FlexibleCNN(
        conv_filters=config.conv_filters,
        kernel_sizes=config.kernel_sizes,
        activation=config.activation,
        dense_neurons=config.dense_neurons,
        num_classes=num_classes,
        use_batchnorm=config.batchnorm,
        dropout=config.dropout
    )
    model = LitClassifier(model=cnn, lr=config.lr)

    # Logger
    wandb_logger = WandbLogger(project=config.project)

    # Callbacks
    callbacks = [
        ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=1, filename='sweep-{epoch}-{val_acc:.2f}'),
        EarlyStopping(monitor='val_acc', patience=3, mode='max')
    ]

    # Trainer
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        max_epochs=config.max_epochs,
        logger=wandb_logger,
        callbacks=callbacks
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    train_wandb()
