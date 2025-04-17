# visualize filters
# python visualize.py --checkpoint best.ckpt --filters

# Activation Maps (Image needed)
# python visualize.py --checkpoint best.ckpt --image path/to/sample.jpg

# Confusion Matrix
# python visualize.py --checkpoint best.ckpt --confmat

import torch
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from torchvision.utils import make_grid
from models.cnn_model import FlexibleCNN
from models.lightning_model import LitClassifier
from data.datamodule import INatDataModule
from PIL import Image
import argparse

def load_model(checkpoint_path, device):
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
    model = LitClassifier.load_from_checkpoint(checkpoint_path, model=cnn)
    model.eval().to(device)
    return model

def visualize_filters(model, save_path=None):
    print("[INFO] Visualizing first conv layer filters...")
    first_layer = model.model.conv_layers[0][0]
    weights = first_layer.weight.data.clone().cpu()

    grid = make_grid(weights, nrow=8, normalize=True, padding=1)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("Filters of First Conv Layer")
    plt.axis('off')

    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_activations(model, image_tensor, device):
    print("[INFO] Visualizing activations from first conv layer...")
    x = image_tensor.to(device)
    with torch.no_grad():
        out = model.model.conv_layers[0](x.unsqueeze(0))

    activations = out.squeeze(0).cpu()
    grid = make_grid(activations[:16].unsqueeze(1), nrow=4, normalize=True, padding=1)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("Activations from First Conv Layer")
    plt.axis('off')
    plt.show()

def visualize_confusion_matrix(model, dataloader, class_names, device):
    print("[INFO] Generating confusion matrix...")
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            preds = torch.argmax(model(x), dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def load_image_tensor(path):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])
    img = Image.open(path).convert("RGB")
    return transform(img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, help="Optional image for activation visualization")
    parser.add_argument("--confmat", action="store_true", help="Generate confusion matrix")
    parser.add_argument("--filters", action="store_true", help="Visualize conv filters")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)

    if args.filters:
        visualize_filters(model)

    if args.image:
        image_tensor = load_image_tensor(args.image)
        visualize_activations(model, image_tensor, args.device)

    if args.confmat:
        dm = INatDataModule(data_dir='inaturalist')
        dm.setup()
        val_loader = dm.val_dataloader()
        visualize_confusion_matrix(model, val_loader, dm.classes, args.device)
