import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class INatDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='inaturalist', batch_size=32, num_workers=4, augment=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment

    def setup(self, stage=None):
        # Transforms
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip() if self.augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ])
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Load datasets
        self.train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=train_transforms)
        self.val_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform=val_transforms)
        self.classes = self.train_dataset.classes

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
