import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch

from models.cnn_model import FlexibleCNN
from models.lightning_model import LitClassifier
from data.datamodule import INatDataModule

def parse_args():
    parser = argparse.ArgumentParser(description="Train Flexible CNN on iNaturalist subset")
    
    parser.add_argument('--filters', nargs=5, type=int, default=[32, 64, 128, 256, 256], help="Filters for each conv layer")
    parser.add_argument('--kernel_sizes', nargs=5, type=int, default=[3, 3, 3, 3, 3], help="Kernel size per layer")
    parser.add_argument('--activation', type=str, default="relu", choices=["relu", "tanh", "leakyrelu"])
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--dense_neurons', type=int, default=128)
    parser.add_argument('--batchnorm', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project', type=str, default="inat-cnn")

    return parser.parse_args()

def main():
    args = parse_args()

    # Data
    datamodule = INatDataModule(
        data_dir='inaturalist',
        batch_size=args.batch_size,
        augment=args.augment
    )
    datamodule.setup()
    num_classes = len(datamodule.classes)

    # Model
    cnn = FlexibleCNN(
        conv_filters=args.filters,
        kernel_sizes=args.kernel_sizes,
        activation=args.activation,
        dense_neurons=args.dense_neurons,
        num_classes=num_classes,
        use_batchnorm=args.batchnorm,
        dropout=args.dropout
    )
    model = LitClassifier(model=cnn, lr=args.lr)

    # Callbacks
    callbacks = [
        ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, filename="best-checkpoint"),
        EarlyStopping(monitor="val_acc", patience=3, mode="max")
    ]

    # Logger
    logger = WandbLogger(project=args.project) if args.use_wandb else None

    # Trainer
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger
    )

    # Training
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
