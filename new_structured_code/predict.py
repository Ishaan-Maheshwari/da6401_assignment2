# Run like this
# python predict.py --image sample.jpg --checkpoint lightning_logs/version_0/checkpoints/best-checkpoint.ckpt

import torch
from torchvision import transforms
from PIL import Image
import argparse

from models.cnn_model import FlexibleCNN
from models.lightning_model import LitClassifier
from data.datamodule import INatDataModule

def load_image(path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dim

def predict(image_path, checkpoint_path):
    datamodule = INatDataModule(data_dir='inaturalist')
    datamodule.setup()
    class_names = datamodule.classes

    # Use matching model config
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
    model.eval()

    image = load_image(image_path)
    with torch.no_grad():
        logits = model(image)
        pred = torch.argmax(logits, dim=1).item()
        print(f"Predicted Class: {class_names[pred]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()

    predict(args.image, args.checkpoint)
