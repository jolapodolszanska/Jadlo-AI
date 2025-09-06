import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from torchvision.models import efficientnet_b0
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

BASE_PATH = "F:/Badania/appka-targi-v2/food-101/images"
TRAIN_FILE = "F:/Badania/appka-targi-v2/food-101/meta/train.txt"
TEST_FILE  = "F:/Badania/appka-targi-v2/food-101/meta/test.txt"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
UNFREEZE_AT = 10   # od której epoki odmrażamy feature extractor

with open(TRAIN_FILE, "r") as f:
    train_paths = f.read().splitlines()
with open(TEST_FILE, "r") as f:
    test_paths = f.read().splitlines()

classes = sorted({p.split("/")[0] for p in train_paths})
labels_map = {name: i for i, name in enumerate(classes)}
NUM_CLASSES = len(classes)
print(f"Liczba klas: {NUM_CLASSES}")

class FoodDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.paths = [os.path.join(BASE_PATH, f + ".jpg") for f in file_list]
        self.labels = [labels_map[f.split("/")[0]] for f in file_list]
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = self.loader(self.paths[idx])
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

train_ds = FoodDataset(train_paths, transform=transform)
test_ds = FoodDataset(test_paths, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4, pin_memory=True)

class FoodClassifier(pl.LightningModule):
    def __init__(self, num_classes, lr=5e-4, unfreeze_at=UNFREEZE_AT):
        super().__init__()
        self.save_hyperparameters()
        base = efficientnet_b0(weights="IMAGENET1K_V1")
        self.feature_extractor = base.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(base.classifier[1].in_features, num_classes)

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        if self.current_epoch == self.hparams.unfreeze_at:
            print(f"🔓 Odmrażanie feature extractora w epoce {self.current_epoch}")
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

if __name__ == "__main__":
    model = FoodClassifier(num_classes=NUM_CLASSES, lr=5e-4)

    checkpoint = ModelCheckpoint(
        dirpath="checkpoints",
        filename="food101-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        monitor="val_acc",
        mode="max"
    )

    logger = CSVLogger("logs", name="food101")

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint],
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader)

    torch.save(model.state_dict(), "food101_model.pt")
    print("✅ Model zapisany jako food101_model.pt")
