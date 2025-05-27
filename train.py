# train.py
# Execution command: python3 train.py --e 50 --mn model --lr 0.001 --bs 50 --loss ssim --sche 0
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
from tqdm import tqdm

from FaceSphereDataset import FaceSphereDataset

print(torch.cuda.is_available())     # Should be True
print(torch.cuda.get_device_name(0)) # Should show your GPU model


# --- Command Line Argument Parser ---
parser = argparse.ArgumentParser(description="Train U-Net on VOC dataset")
parser.add_argument('--e', type=int, default=5,
                    help='Number of iterations') # mn: model name
parser.add_argument('--mn', type=str, default='unet.pth',
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

# --------------- Transforms ---------------
transform_face = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

transform_sphere = A.Compose([
    A.Resize(256, 256),
    ToTensorV2()
])

# --------------- Dataloader ---------------
# train_dataset = FaceSphereDataset(root_dir='./data/dataset_256px_11f_100im', split='train', transforms=transform)
# train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
train_dataset = FaceSphereDataset(
    root_dir='./data/dataset_256px_16f_100im', 
    split='train', 
    transforms_face=transform_face, 
    transforms_sphere = transform_sphere
)
train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)

test_dataset = FaceSphereDataset(root_dir='./data/dataset_256px_16f_100im', split='test', transforms_face=transform_face, 
    transforms_sphere = transform_sphere)
test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

val_dataset = FaceSphereDataset(
    root_dir='./data/dataset_256px_16f_100im', 
    split='val', 
    transforms_face=transform_face, 
    transforms_sphere = transform_sphere
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=args.bs, 
    shuffle=False
)


# --------------- U-Net model using ResNet34 encoder ---------------
model = smp.Unet( # TODO: maybe I can retrain the last few layers of the encoder
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
for param in model.encoder.parameters():
    param.requires_grad = False
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
# optimizer = optim.Adam(model.parameters(), lr=args.lr)

# ---------------- Learning Rate Scheduler --------------
scheduler = CyclicLR(optimizer, base_lr=args.lr, max_lr=10.0, step_size_up=2000, step_size_down=2000, mode='triangular')



# --------------- Training Loop ---------------
def train(model, train_dataloader, val_dataloader, optimizer, criterion, epochs):
    global args

    model.train()
    mask = train_dataset.maskTensor.to(device)
    mask = mask.unsqueeze(0).repeat(args.bs, 1, 1, 1)

    train_losses = []
    val_losses = []

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        prgbar= tqdm(train_dataloader)
        for images, gtruth in prgbar:
            images, gtruth = images.to(device), gtruth.to(device)

            outputs = model(images)

            ### USE MASK ON PREDICTION AND GROUND TRUTH
            # print(f"inputs shape: {images.shape}, outputs shape: {outputs.shape}, gtruth shape: {gtruth.shape}, mask shape: {mask.shape}")
            # print("gtruth normal:", gtruth[0, :, 69, 100])
            # print("Output normal:", outputs[0, :, 69, 100])
            # print("gtruth shape:", gtruth.shape)
            # print("mask shape:", mask.shape)
            gtruth = gtruth * mask.int().float()[:gtruth.shape[0], :, :, :]
            outputs = outputs * mask.int().float()[:gtruth.shape[0], :, :, :]

            # print("gtruth masked:", gtruth[0, :, 69, 100])
            # print("Output masked:", outputs[0, :, 69, 100])

            # Using SSIM-based loss
            loss = criterion(outputs, gtruth)  # Replacing the old loss function
            # print(loss) # to check ssim is between 0 to 1.

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            # Step the CLR scheduler after each optimizer step
            if args.sche==True:
                scheduler.step()

            epoch_loss += loss.item()
        
        # Average training loss for this epoch
        avg_train_loss = epoch_loss / len(train_dataloader.dataset)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_epoch_loss = 0
        
        with torch.no_grad():
            prgbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Valid]")
            for images, gtruth in prgbar:
                images, gtruth = images.to(device), gtruth.to(device)
                
                outputs = model(images)
                gtruth = gtruth * mask.int().float()[:gtruth.shape[0], :, :, :]
                outputs = outputs * mask.int().float()[:outputs.shape[0], :, :, :]
                
                loss = criterion(outputs, gtruth)
                
                val_epoch_loss += loss.item()
        
        # Average validation loss for this epoch
        avg_val_loss = val_epoch_loss / len(val_dataloader.dataset)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Avg Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if args.sche==True:
            print("Current learning rate: ", scheduler.get_lr())   

    end_time = time.time()  # End timing
    print(f"\n Total training time: {(end_time - start_time):.2f} seconds")

    return train_losses, val_losses

def plot_losses(train_losses, val_losses, args):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss\n{args.loss} loss, LR={args.lr}, BS={args.bs}')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    loss_plot_filename = f"LP_{args.mn}_{args.loss}_bs{args.bs}_e{args.e}_lr{args.lr}_sche{args.sche}_loss_plot.png"
    plt.savefig(f"train_output/{loss_plot_filename}")
    plt.show()

train_losses, val_losses = train(model, train_loader, val_loader, optimizer, loss_func, epochs=args.e)

plot_losses(train_losses, val_losses, args)

# save model
model_name = args.mn + '_' + str(args.loss) + '_bs' + str(args.bs) + '_e' + str(args.e) + \
             '_lr' + str(args.lr) + '_sche' + str(args.sche) + '.pth'
torch.save(model.state_dict(), model_name)

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

def minmaxscale(img):
    #img = np.array(img).astype(np.float32) / 255.0 
    img = np.transpose(img, (1, 2, 0))
    #img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def visualize_prediction(model, dataset, idx=0): # TODO: check if normalize is correct. bc background color
    model.eval()
    inputs, gtruth = dataset[idx]  # inputs: tensor (3,H,W), gtruth: (1,H,W) or (3,H,W)
    with torch.no_grad():
        # pred = torch.sigmoid(model(inputs.unsqueeze(0).to(device)))
        # Use clamp instead of sigmoid
        # pred = pred.squeeze().cpu().numpy()
        pred = model(inputs.unsqueeze(0).to(device)).squeeze().cpu().numpy()
        pred = np.clip(pred, 0, 1)
    
    # Get the mask as a boolean array
    mask_3d = np.repeat(train_dataset.mask[:, :, np.newaxis], 3, axis=2)

    print(f"inputs shape: {inputs.shape}, outputs shape: {pred.shape}, gtruth shape: {gtruth.shape}, mask shape: {mask_3d.shape}")
    print("gtruth normal:", gtruth[:, 160, 137])
    print("Output normal:", pred[:, 160, 137])
    print("Input normal:", inputs[:, 160, 137])

    # Convert prediction to (H, W, 3) format
    pred = np.transpose(pred, (1, 2, 0))
    #print(pred[126, 69])
    
    # Apply background color directly
    pred_with_bg = np.where(mask_3d, pred, np.array([0.4588, 0.4588, 0.4588]))
    # pred_with_bg = np.array(pred_with_bg).astype(np.float32) / 255.0
    
    # pred_with_bg = minmaxscale(gtruth) * train_dataset.mask_3d

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(unnormalize(inputs))
    plt.title("Image")
    plt.subplot(1, 3, 2)
    #gtruth = np.transpose(gtruth, (1, 2, 0))
    plt.imshow(minmaxscale(gtruth))
    plt.title("Ground Truth (RGB)")
    plt.subplot(1, 3, 3)
    plt.imshow(pred_with_bg) # We do not need unnormalize for output
    plt.title("Prediction (RGB)")

    if idx == 0 or idx == 1:
        # Save the plot as an image file in the /output directory
        outfile = args.mn + '_' + str(args.loss) + '_bs' + str(args.bs) + '_e' + \
                  str(args.e) + '_lr' + str(args.lr) + '_sche' + str(args.sche) + '_' + str(idx) + '.png'
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{outfile}")
        plt.close()
    
    plt.show()

# save
output_dir = './train_output'
os.makedirs(output_dir, exist_ok=True)

# Visualize result
for i in range(5):
    visualize_prediction(model, train_dataset, idx=i)




