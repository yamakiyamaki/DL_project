# train.py
# Execution command: python3 train.py --e 50 --mn model_name.pth
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.datasets import VOCSegmentation
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import argparse
import time
from torchmetrics import StructuralSimilarityIndexMeasure  # Import SSIM

from FaceSphereDataset import FaceSphereDataset

print(torch.cuda.is_available())     # Should be True
print(torch.cuda.get_device_name(0)) # Should show your GPU model

# --------------- Custom Dataset using VOC as example ---------------
class VOCDataset(Dataset):
    def __init__(self, root, image_set="train", transforms=None):
        self.dataset = VOCSegmentation(root=root, year="2012", image_set=image_set, download=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        image = np.array(image)
        mask = np.array(mask)

        # Convert mask to binary (e.g., segment class 15 only)
        mask = (mask == 15).astype(np.float32)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.unsqueeze(0)  # Add channel dimension

# --------------- Transforms ---------------
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# --------------- Dataloader ---------------
# train_dataset = VOCDataset(root='./data', image_set="train", transforms=transform)
train_dataset = FaceSphereDataset(root_dir='./data/dataset_256px_11f_100im', split='train')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


# --- Command Line Argument Parser ---
parser = argparse.ArgumentParser(description="Train U-Net on VOC dataset")
parser.add_argument('--e', type=int, default=5,
                    help='Number of iterations') # mn: model name
parser.add_argument('--mn', type=str, default='unet_resnet34_voc_50.pth',
                    help='Filename to save trained model') # mn: model name
args = parser.parse_args()


# --------------- U-Net model using ResNet34 encoder ---------------
model = smp.Unet(
    encoder_name="resnet34", # encoder architecture is resnet
    encoder_weights="imagenet", # this resnet pretrained on imagenet
    in_channels=3,
    classes=1,
) # encoder weight is frozen

# Optional: use only encoder
# encoder = model.encoder  # Uncomment if you want to use encoder only

# --------------- Training Setup ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# criterion = nn.BCEWithLogitsLoss()
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)  # Define SSIM metric

def ssim_loss(x, y):
    return 1 - ssim(x, y)  # Return 1 - SSIM to use it as a loss

optimizer = optim.Adam(model.parameters(), lr=1e-5)

# --------------- Training Loop ---------------
def train(model, dataloader, optimizer, criterion, epochs=args.e):
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)

            # BCEWithLogitsLoss
            # loss = criterion(outputs, masks)

            # Using SSIM-based loss
            loss = criterion(outputs, masks)  # Replacing the old loss function


            optimizer.zero_grad()
            loss.backward() # TODO: change loss function
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    end_time = time.time()  # End timing
    print(f"\n Total training time: {(end_time - start_time):.2f} seconds")

train(model, train_loader, optimizer, ssim_loss)

# save model
torch.save(model.state_dict(), args.mn)

