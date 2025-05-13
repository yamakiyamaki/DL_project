# train.py
# Execution command: python3 train.py --e 50 --mn model_name.pth --lr 0.001 --bs 16 --loss ssim --sche 1
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
from torch.optim.lr_scheduler import CyclicLR

from FaceSphereDataset import FaceSphereDataset

print(torch.cuda.is_available())     # Should be True
print(torch.cuda.get_device_name(0)) # Should show your GPU model


# --- Command Line Argument Parser ---
parser = argparse.ArgumentParser(description="Train U-Net on VOC dataset")
parser.add_argument('--e', type=int, default=5,
                    help='Number of iterations') # mn: model name
parser.add_argument('--mn', type=str, default='unet_resnet34_voc_50.pth',
                    help='Filename to save trained model') # mn: model name
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate')
parser.add_argument('--bs', type=int, default=16,
                    help='batch size')
parser.add_argument('--loss', type=str, default="ssim",
                    help='loss function: ssim, mse, l1, comb')
parser.add_argument('--sche', type=int, default=0,
                    help='use cyclic learning rate scheduler: 0 or 1')
args = parser.parse_args()


# --------------- Dataloader ---------------
train_dataset = FaceSphereDataset(root_dir='./data/dataset_256px_11f_100im', split='train')
train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)


# --------------- U-Net model using ResNet34 encoder ---------------
model = smp.Unet(
    encoder_name="resnet34", # encoder architecture is resnet
    encoder_weights="imagenet", # this resnet pretrained on imagenet
    in_channels=3,
    classes=3,
) # encoder weight is frozen
# Optional: use only encoder
# encoder = model.encoder  # Uncomment if you want to use encoder only

# --------------- Training Setup ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ---------------- Loss function ---------------
# ssim
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
def ssim_loss(x, y):
    return 1 - ssim(x, y)  # Return 1 - SSIM to use it as a loss

# ssim + mse
class MSE_SSIM_Loss(nn.Module):
    def __init__(self, ssim_weight=0.5, mse_weight=0.5, device='cpu'):
        super(MSE_SSIM_Loss, self).__init__()
        self.device = device
        self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)  # Send SSIM to device
        self.mse_loss = nn.MSELoss().to(device)  # Send MSE to device
        self.ssim_weight = ssim_weight
        self.mse_weight = mse_weight

    def forward(self, output, target):
        # Calculate MSE loss
        mse = self.mse_loss(output, target)
        
        # Calculate SSIM loss
        ssim = 1 - self.ssim_loss(output, target)  # SSIM is typically between 0 and 1, so we subtract it from 1 to turn it into a loss
        
        # Combine MSE and SSIM losses
        total_loss = self.mse_weight * mse + self.ssim_weight * ssim
        return total_loss

if args.loss == "ssim":
    loss_func = ssim_loss
elif args.loss == "mse":       
    loss_func = nn.MSELoss().to(device)
elif args.loss == "l1":
    loss_func = nn.L1Loss().to(device)
elif args.loss == "comb":
    loss_func = MSE_SSIM_Loss(ssim_weight=0.5, mse_weight=0.5, device=device)  # Make sure the loss function is on the correct device


# ---------------- Optimizer ---------------
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# ---------------- Learning Rate Scheduler --------------
scheduler = CyclicLR(optimizer, base_lr=args.lr, max_lr=1e-3, step_size_up=2000, step_size_down=2000, mode='triangular')

# --------------- Training Loop ---------------
def train(model, dataloader, optimizer, criterion, epochs):
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)

            # Using SSIM-based loss
            loss = criterion(outputs, masks)  # Replacing the old loss function

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            # Step the CLR scheduler after each optimizer step
            if args.sche==True:
                scheduler.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        if args.sche==True:
            print("Current learning rate: ", scheduler.get_lr())   

    end_time = time.time()  # End timing
    print(f"\n Total training time: {(end_time - start_time):.2f} seconds")

train(model, train_loader, optimizer, loss_func, epochs=args.e)

# save model
torch.save(model.state_dict(), args.mn)

# --------------- Visualization ---------------
def normalize(img):
    img = np.array(img).astype(np.float32) / 255.0 
    img = np.transpose(img, (1, 2, 0))
    return img

def unnormalize(img):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.transpose(img, (1, 2, 0))
    return np.clip((img * std + mean), 0, 1)

def visualize_prediction(model, dataset, idx=0):
    model.eval()
    image, mask = dataset[idx]  # image: tensor (3,H,W), mask: (1,H,W) or (3,H,W)
    with torch.no_grad():
        pred = torch.sigmoid(model(image.unsqueeze(0).to(device)))
        pred_mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(normalize(image))
    plt.title("Image")
    plt.subplot(1, 3, 2)
    plt.imshow(normalize(mask))
    plt.title("Ground Truth (RGB)")
    plt.subplot(1, 3, 3)
    plt.imshow(unnormalize(pred_mask))
    plt.title("Prediction (RGB)")
    plt.show()

# Visualize result
for i in range(5):
 visualize_prediction(model, train_dataset, idx=i)


