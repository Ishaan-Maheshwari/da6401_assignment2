#### Comparing Training from Scratch vs Fine-Tuning

- Fine-tuning a pre-trained model converges much faster than training from scratch. Pretrained weights already capture useful low- and mid-level features so the model doesn't start from zero.

- We were using only a subset of iNaturalist data. Fine-tuning consistently outperforms training from scratch, especially when the dataset is small. Training from scratch often leads to overfitting or poor generalization.

- Since only a subset of layers (or initially just the final layer) is trained in early epochs, fine-tuning is more efficient in terms of computation and time compared to full training from scratch.

- Fine-tuning allows flexibility — e.g., freezing/unfreezing layers gradually, applying different learning rates per layer group, etc. This control is not possible in a scratch model without rethinking the full architecture.


