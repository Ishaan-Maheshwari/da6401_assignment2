### **Q1: The dimensions of the images in your data may not be the same as that in the ImageNet data. How will you address this?**

- The pre-trained model (ResNet-50) expects input images of size **224x224 pixels**.

- To ensure compatibility, all input images from the iNaturalist subset are **resized to 224x224** during preprocessing.

- This is handled using `torchvision.transforms.Resize()` in the data pipeline.

- We also normalize the images using **ImageNet mean and standard deviation**, so that input distributions match what the pre-trained model expects.

```python
transforms.Resize((224, 224))
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

---

### **Q2: ImageNet has 1000 classes and hence the last layer of the pre-trained model would have 1000 nodes. However, the naturalist dataset has only 10 classes. How will you address this?**

- The original fully-connected (fc) layer in ResNet-50 is:
  
  ```python
  model.fc = nn.Linear(in_features=2048, out_features=1000)
  ```

- I replaced this layer with a new `nn.Linear` layer that outputs **10 logits**, one for each class in the iNaturalist subset:
  
  ```python
  model.fc = nn.Linear(in_features=2048, out_features=10)
  ```

- This allows the model to learn a new classification head suited for the iNaturalist task while keeping the pretrained feature extractor layers intact.



#### Code Implementation

- I loaded a **ResNet-50 model pre-trained on ImageNet** using `torchvision.models.resnet50(pretrained=True)`, and replaced its final classification layer to match the number of classes in the iNaturalist dataset.

- **Two fine-tuning strategies** are implemented:
  
  - Strategy 1: Freeze all layers initially and train only the final classifier, then unfreeze the whole model and fine-tune with a lower learning rate.
  
  - Strategy 2: Unfreeze only the last block (`layer4`) and the final layer first, then unfreeze the entire model and continue training.

- Training and validation loops are written to log **loss and accuracy per epoch**, and use standard PyTorch training patterns with `Adam` optimizer and `CrossEntropyLoss`.


