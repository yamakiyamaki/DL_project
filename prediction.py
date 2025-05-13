# prediction.py
# Execution command: python3 prediction.py --idx 5 --mn model_name.pth

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import VOCSegmentation
from torch.utils.data import Dataset
from PIL import Image
import argparse

# ----- Custom Dataset -----
class VOCDataset(Dataset):
    def __init__(self, root, image_set="val", transforms=None):
        self.dataset = VOCSegmentation(root=root, year="2012", image_set=image_set, download=False)
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        image = np.array(image)
        mask = np.array(mask)
        mask = (mask == 15).astype(np.float32)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.unsqueeze(0)

# ----- Transforms -----
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ----- Load dataset -----
dataset = VOCDataset(root='./data', image_set="val", transforms=transform)


# --- Command Line Argument Parser ---
parser = argparse.ArgumentParser(description="Visualize U-Net prediction for VOC dataset.")
parser.add_argument("--idx", type=int, default=0, help="Index of the image to visualize")
parser.add_argument('--mn', type=str, default='unet_resnet34_voc_50.pth',
                    help='Filename to save trained model') # mn: model name
args = parser.parse_args()


# ----- Load model -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,  # Don't load pretrained weights again
    in_channels=3,
    classes=1,
)
model.load_state_dict(torch.load(args.mn, map_location=device))
model = model.to(device)

# ----- Visualization function -----
def visualize_prediction(model, dataset, idx=100):
    model.eval()
    image, mask = dataset[idx]
    with torch.no_grad():
        pred = torch.sigmoid(model(image.unsqueeze(0).to(device)))
        pred_mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.title("Image")
    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze().numpy(), cmap='gray')
    plt.title("Ground Truth")
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap='gray')
    plt.title("Prediction")
    plt.show()

# ----- Run inference -----
visualize_prediction(model, dataset, idx=args.idx) # idx = index of input image