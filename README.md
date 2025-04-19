# 🧠 CNN Image Classifier – iNaturalist Subset

This project implements a highly modular and configurable **Convolutional Neural Network (CNN)** image classifier using **PyTorch Lightning**, trained on a 10-class subset of the [iNaturalist](https://www.inaturalist.org/) dataset.

The code is structured for flexibility, experimentation, and scalability — supporting:
- Flexible CNN architecture (configurable filters, activations, etc.)
- Hyperparameter tuning via **W&B sweeps**
- Visualization of learned convolutional filters
- Clean object-oriented design
- GPU support and Colab compatibility

## 📁 Project Structure

```
da6401_assignment2/
│
├── data_module.py         # PyTorch Lightning DataModule for loading & augmenting iNaturalist data
├── cnn_model.py           # Flexible CNN model (nn.Module)
├── lightning_model.py     # LitClassifier: LightningModule wrapping the CNN model, training logic
├── train.py               # Training script, supports CLI + W&B sweep
├── test.py                # Script to test a trained model on validation data
├── predict.py             # Predict the class of a single image
├── visualize.py           # Save learned convolutional filters as image files
├── sweep_config.yaml      # W&B Sweep configuration file for hyperparameter tuning
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 🧪 Dataset Format

Your dataset should be in this format:

```
inaturalist/
├── train/
│   ├── Amphibia/
│   ├── Animalia/
│   └── ...
└── val/
    ├── Amphibia/
    ├── Animalia/
    └── ...
```

Each class folder contains `.jpg` images.

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py \
  --conv_filters="[32, 64, 128, 128, 128]" \
  --kernel_sizes="[5, 3, 3, 3, 3]" \
  --activation relu \
  --dense_neurons 512 \
  --dropout 0.3 \
  --batchnorm True \
  --augment True \
  --batch_size 32 \
  --lr 0.0005 \
  --max_epochs 15 \
  --project assignment_2
```

### 3. Run W&B Sweep

```bash
wandb login
wandb sweep sweep_config.yaml
wandb agent <your-username>/<project-name>/<sweep-id>
```

## 🧠 Module Descriptions

### `cnn_model.py`
- Contains `FlexibleCNN`, a highly configurable CNN model.
- Parameters:
  - Number and size of convolution filters
  - Activation function
  - Dropout
  - Batch normalization
- Includes a method to compute total parameters and operations.

### `lightning_model.py`
- Contains `LitClassifier`, a `pl.LightningModule`
  - Wraps `FlexibleCNN`
  - Defines training/validation/test steps
  - Calculates accuracy
  - Logs metrics and losses using `self.log()`

### `data_module.py`
- Loads dataset from `inaturalist/train` and `inaturalist/val`
- Applies optional augmentations
- Returns train, val, and test dataloaders

### `train.py`
- CLI script to train the model
- Supports arguments for all major hyperparameters
- Can be run standalone or used with W&B sweeps

### `test.py`
- Loads a trained model checkpoint
- Evaluates it on validation data
- Prints accuracy

### `predict.py`
- Predicts the class of a single image using a trained model checkpoint
- Shows top prediction and probabilities

### `visualize.py`
- Saves learned filters of each convolutional layer as PNG images
- Helps visualize how the model “sees” different features

## 🔧 sweep_config.yaml

Defines a W&B sweep using random search. Sweeps over:
- Conv filters
- Kernel sizes
- Activation
- Dense layer size
- Dropout
- Batch norm
- Data augmentation
- Learning rate
- Batch size

## 🖼️ Visualization

```bash
python visualize.py --checkpoint_path path/to/checkpoint.ckpt
```

Outputs images to `filter_viz/`.

## 🔍 Testing

```bash
python test.py --checkpoint_path path/to/checkpoint.ckpt
```

## 🤖 Predicting

```bash
python predict.py --image_path sample.jpg --checkpoint_path path/to/checkpoint.ckpt
```

## 🧼 Tips & Best Practices

- Use W&B sweeps for tuning.
- Visualize filters to interpret learning.
- Run locally for fast iterations, then scale.
- Subset the dataset for rapid prototyping.

## 📜 License

MIT License

## 🙋‍♂️ Questions?

Feel free to reach out!
