import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import glob
import random
import shutil
from albumentations.pytorch import ToTensorV2
import albumentations as A

class FaceSphereDataset(Dataset):
    def __init__(self, root_dir, split='train', transforms=None):
        """
        Args:
            root_dir: Path to the dataset main directory
            split: 'train' or 'test'
            transforms: Optional transform to be applied on images
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transforms = transforms

        if (not self.transforms):
            self.transforms = A.Compose([
                ToTensorV2()
            ])

        # Get all the face images
        self.face_images = sorted(glob.glob(os.path.join(self.root_dir, "*_Face.png")))
        self.sphere_images = []
        
        for face_path in self.face_images:
            # Extract the base name without extension
            base_name = os.path.basename(face_path)
            # Extract face identifier (e.g., 'Face00')
            face_id = base_name.split('_')[0]
            iter_id = base_name.split('_')[1]
            # Find corresponding sphere image
            sphere_path = os.path.join(self.root_dir, f"{face_id}_{iter_id}_Sphere.png")
            
            if os.path.exists(sphere_path):
                self.sphere_images.append(sphere_path)
            else:
                # If no sphere image found, remove this face from the list
                self.face_images.remove(face_path)
        
        print(f"Found {len(self.face_images)} valid image pairs in {split} set")
    
    def __len__(self):
        return len(self.face_images)
    
    def __getitem__(self, idx):
        # Load face image
        face_img = Image.open(self.face_images[idx]).convert('RGB')
        face_img = np.array(face_img)
        
        # Load corresponding sphere image
        sphere_img = Image.open(self.sphere_images[idx]).convert('RGB')
        sphere_img = np.array(sphere_img)

        if self.transforms:
            augmented_face = self.transforms(image=face_img)
            augmented_sphere = self.transforms(image=sphere_img)
            face_img = augmented_face['image']  # Should now be [C, H, W]
            sphere_img = augmented_sphere['image']

        # Ensure tensors are floating point type
        if isinstance(face_img, torch.Tensor) and face_img.dtype != torch.float32:
            face_img = face_img.float()
        
        if isinstance(sphere_img, torch.Tensor) and sphere_img.dtype != torch.float32:
            sphere_img = sphere_img.float()
        
        return face_img, sphere_img