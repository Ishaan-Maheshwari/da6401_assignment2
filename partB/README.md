## Fine-Tuning Pre-trained CNN on iNaturalist Subset

This task involves fine-tuning a pre-trained CNN model (ResNet-50) on a subset of the iNaturalist dataset. Instead of training a model from scratch, I used a model pre-trained on ImageNet, which already contains many images of animals and plants. This helps in faster convergence and better generalization, especially when working with a limited dataset.

---

## Dataset Structure

The dataset is organized as follows:

inaturalist/
├── train/
│ ├── class_1/
│ ├── class_2/
│ └── ...
└── val/
├── class_1/
├── class_2/
└── ...

Each class directory contains image files.

---

## Preprocessing

ImageNet models expect images of shape `3 x 224 x 224`, normalized using the ImageNet mean and std. So we apply the following transforms:

- Resize all images to `224x224`
- Normalize using ImageNet statistics
- Use data augmentation (horizontal flips) on training data

---

## Model Adjustment

We use `torchvision.models.resnet50(pretrained=True)`, which has 1000 output classes. Since our dataset has 10 classes, we replace the last fully connected layer with:

```python
model.fc = nn.Linear(2048, 10)
```

---

## Fine-Tuning Strategies

### Strategy 1: Two-Phase Fine-Tuning

1. Freeze all layers except the last classifier layer.

2. Train only the classifier for a few epochs.

3. Unfreeze the whole network and fine-tune with a lower learning rate.

This strategy allows the new classifier head to adapt first, before modifying earlier layers.

### Strategy 2: Layer-wise Fine-Tuning

1. Freeze all layers initially.

2. Unfreeze only the last block (`layer4`) and the classifier layer.

3. Train these for a few epochs.

4. Then unfreeze the full model and fine-tune the rest.

This approach gradually adapts the network and avoids large shifts in pretrained weights.

---

## Training Setup

- Optimizer: Adam

- Loss Function: CrossEntropyLoss

- Device: GPU if available

- Batch Size: 32

- LR: `1e-4` for warm-up, `1e-5` for full fine-tuning

Training and validation accuracy and loss are printed at each epoch to track performance.

---

## Sections

- `Section 1`: Implements Strategy 1

- `Section 2`: Implements Strategy 2

---

## Results

#### Strategy 1 - Entire Model

- Stage 1 while fine tuning classifier layer only
  
  - [Stage 1 - Epoch 1] Train Acc: 0.6490, Val Acc: 0.7415 
  
  - [Stage 1 - Epoch 2] Train Acc: 0.7224, Val Acc: 0.7635 
  
  - [Stage 1 - Epoch 3] Train Acc: 0.7358, Val Acc: 0.7440

- Stage 2 while fine tuning the model
  
  - [Stage 2 - Epoch 1] Train Acc: 0.7444, Val Acc: 0.7595 
  
  - [Stage 2 - Epoch 2] Train Acc: 0.8643, Val Acc: 0.7785 
  
  - [Stage 2 - Epoch 3] Train Acc: 0.8982, Val Acc: 0.7890 [Stage 2 - Epoch 4] Train Acc: 0.9278, Val Acc: 0.7685 [Stage 2 - Epoch 5] Train Acc: 0.9312, Val Acc: 0.7780
