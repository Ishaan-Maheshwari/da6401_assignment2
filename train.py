import matplotlib.pyplot as plt
import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from cnn_model import FlexibleCNN
from lit_module import LitClassifier

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--filters", type=str, default="64,128,256,512,512",
                        help="Comma-separated filters per conv layer")
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "leakyrelu", "gelu"])
    parser.add_argument("--filter_pattern", type=str, default="custom",
                        choices=["same", "double", "half", "custom"])
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--batchnorm", action="store_true", help="Use BatchNorm after conv")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--dense_neurons", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    return parser.parse_args()

args = parse_args()

# Parse filters based on pattern
if args.filter_pattern == "same":
    filters = [int(args.filters.split(',')[0])] * 5
elif args.filter_pattern == "double":
    base = int(args.filters.split(',')[0])
    filters = [base * (2 ** i) for i in range(5)]
elif args.filter_pattern == "half":
    base = int(args.filters.split(',')[0])
    filters = [base // (2 ** i) for i in range(5)]
else:
    filters = list(map(int, args.filters.split(',')))


early_stop_cb = EarlyStopping(monitor='val_loss', patience=5, mode='min')
checkpoint_cb = ModelCheckpoint(
    monitor='val_acc',
    mode='max',
    save_top_k=1,
    filename='best-checkpoint'
)

# Transforms
train_tfms = [transforms.Resize((224, 224)), transforms.ToTensor()]
if args.augment:
    train_tfms.insert(0, transforms.RandomHorizontalFlip())

train_dataset = datasets.FakeData(transform=transforms.Compose(train_tfms), size=1000)
val_dataset = datasets.FakeData(transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]), size=200)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

# Build model
cnn = FlexibleCNN(
    conv_filters=filters,
    kernel_sizes=[3] * 5,
    activation=args.activation,
    dense_neurons=args.dense_neurons,
    num_classes=10,
    use_batchnorm=args.batchnorm,
    dropout=args.dropout
)

# Lightning module
model = LitClassifier(cnn, lr=args.lr)

# Logger
# wandb_logger = WandbLogger(project="iNaturalist-CNN", log_model=True)

# Trainer
trainer = pl.Trainer(
    max_epochs=50,
    # logger=wandb_logger,
    precision=16,
    devices=1,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    callbacks=[early_stop_cb, checkpoint_cb]
)

trainer.fit(model, train_loader, val_loader)

# For Visualising filters

def visualize_filters(model, layer_idx=0):
    conv_layer = list(model.model.conv_blocks.children())[layer_idx * 3]  # Conv layers are every 3rd block
    weights = conv_layer.weight.data.cpu()

    # Normalize
    weights = (weights - weights.min()) / (weights.max() - weights.min())

    num_kernels = weights.shape[0]
    num_cols = 8
    num_rows = (num_kernels + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    axs = axs.flatten()

    for i in range(num_kernels):
        axs[i].imshow(weights[i].permute(1, 2, 0))  # CxHxW -> HxWxC
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

# Call it after training
visualize_filters(model)