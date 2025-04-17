import os
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms
from PIL import Image

from models.cnn_model import FlexibleCNN
from models.lightning_model import LitClassifier
from data.datamodule import InatDataModule

# 🔧 Create output folder
os.makedirs("outputs", exist_ok=True)

def plot_filters(conv_layer, layer_idx):
    filters = conv_layer.weight.data.clone().cpu()
    num_filters = filters.shape[0]
    fig, axs = plt.subplots(1, min(num_filters, 8), figsize=(15, 5))
    for i in range(min(num_filters, 8)):
        axs[i].imshow(filters[i][0], cmap="gray")
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig(f"outputs/filters_layer{layer_idx}.png")
    plt.close()

def plot_confusion_matrix(model, dataloader, class_names):
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            preds = model(images.to(model.device)).argmax(dim=1)
            all_preds.extend(preds.cpu())
            all_labels.extend(labels.cpu())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

def plot_activations(model, image_tensor):
    model.eval()
    activations = []

    x = image_tensor.unsqueeze(0).to(model.device)
    for layer in model.model.conv_layers:
        x = layer(x)
        if isinstance(layer, torch.nn.MaxPool2d):
            activations.append(x.clone().detach().cpu())

    for idx, act in enumerate(activations):
        fig, axs = plt.subplots(1, min(act.shape[1], 6), figsize=(15, 5))
        for i in range(min(act.shape[1], 6)):
            axs[i].imshow(act[0][i], cmap='viridis')
            axs[i].axis('off')
        plt.tight_layout()
        plt.savefig(f"outputs/activation_map_layer{idx}.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--filters", action="store_true", help="Visualize filters")
    parser.add_argument("--confmat", action="store_true", help="Visualize confusion matrix")
    parser.add_argument("--image", type=str, help="Path to an image to visualize activations")
    args = parser.parse_args()

    # Model config must match the checkpoint config
    model_config = {
        "conv_filters": [32, 64, 128, 256, 256],
        "kernel_sizes": [3, 3, 3, 3, 3],
        "activation": "relu",
        "dense_neurons": 128,
        "num_classes": 10,
        "use_batchnorm": False,
        "dropout": 0.3
    }
    cnn = FlexibleCNN(**model_config)
    model = LitClassifier.load_from_checkpoint(args.checkpoint, model=cnn).to("cpu")

    if args.filters:
        print("Saving filter visualizations...")
        for idx, layer in enumerate(cnn.conv_layers):
            if isinstance(layer, torch.nn.Conv2d):
                plot_filters(layer, idx)

    if args.confmat:
        print("Saving confusion matrix...")
        dm = InatDataModule(data_dir="inaturalist", batch_size=32)
        dm.setup("test")
        plot_confusion_matrix(model, dm.val_dataloader(), dm.classes)

    if args.image:
        print("Saving activation maps for image...")
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        img = Image.open(args.image).convert("RGB")
        img_tensor = transform(img)
        plot_activations(model, img_tensor)

    print("✅ Visualizations saved in the 'outputs/' folder.")


#  TO Run
# python visualize.py --checkpoint lightning_logs/version_0/checkpoints/best-checkpoint.ckpt --filters --confmat

# For Image Activations 
# python visualize.py --checkpoint lightning_logs/version_0/checkpoints/best-checkpoint.ckpt --image some_image.jpg