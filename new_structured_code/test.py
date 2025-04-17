# test.py
# Run the test like
# python test.py --checkpoint <path to the model checkpoint>
# python test.py --checkpoint lightning_logs/version_0/checkpoints/best-checkpoint.ckpt

import torch
from pytorch_lightning import Trainer
from models.lightning_model import LitClassifier
from models.cnn_model import FlexibleCNN
from data.datamodule import INatDataModule
from sklearn.metrics import classification_report
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to .ckpt checkpoint")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def main():
    args = parse_args()

    # Setup datamodule (only val loader needed)
    datamodule = INatDataModule(data_dir='inaturalist', batch_size=args.batch_size)
    datamodule.setup()
    val_loader = datamodule.val_dataloader()

    # Load the CNN config you used during training
    model_config = {
        "conv_filters": [32, 64, 128, 256, 256],
        "kernel_sizes": [3, 3, 3, 3, 3],
        "activation": "relu",
        "dense_neurons": 128,
        "num_classes": 10,
        "use_batchnorm": True,
        "dropout": 0.3
    }
    cnn = FlexibleCNN(**model_config)
    model = LitClassifier.load_from_checkpoint(args.checkpoint, model=cnn).to(args.device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=datamodule.classes))

if __name__ == "__main__":
    main()
