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

from FaceSphereDataset import FaceSphereDataset

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
test_dataset = FaceSphereDataset(root_dir='./data/dataset_256px_11f_100im', split='test', transforms_face=transform_face, transforms_sphere=transform_sphere)

# --- Command Line Argument Parser ---
parser = argparse.ArgumentParser(description="Visualize U-Net prediction for VOC dataset.")
parser.add_argument("--idx", type=int, default=0, help="Index of the image to visualize")
parser.add_argument('--mn', type=str, default='unet_resnet34_voc_50.pth',
                    help='Filename to save trained model') # mn: model name
args = parser.parse_args()


# ----- Load model -----
model = smp.Unet( # TODO: maybe I can retrain the last few layers of the encoder
    encoder_name="resnet34", # encoder architecture is resnet
    encoder_weights="imagenet", # this resnet pretrained on imagenet
    in_channels=3,
    classes=3,
) # encoder weight is frozen
model.load_state_dict(torch.load(args.mn))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

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
        # pred = pred.squeeze().cpu().numpy()
        pred = model(inputs.unsqueeze(0).to(device)).squeeze().cpu().numpy()
        pred = np.clip(pred, 0, 1)
        
    # Get the mask as a boolean array
    mask_3d = np.repeat(test_dataset.mask[:, :, np.newaxis], 3, axis=2)

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

    # if idx == 0 or idx == 1:
        # Save the plot as an image file in the /output directory
    outfile = args.mn.split("/")[-1].split(".")[0] +'_' + str(idx) + '.png'
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{outfile}")
    plt.close()
    
    plt.show()

# save
output_dir = './test_output'
os.makedirs(output_dir, exist_ok=True)

# Visualize result
start_idx = args.idx
for i in range(start_idx, start_idx+5):
    visualize_prediction(model, test_dataset, idx=i)