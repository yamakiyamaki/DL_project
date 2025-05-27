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
    def __init__(self, root_dir, split='train', transforms_face=None, transforms_sphere=None, mask_path="./data/SphereMask.jpg"):
        """
        Args:
            root_dir: Path to the dataset main directory
            split: 'train', 'test' or 'val'
            transforms: Optional transform to be applied on images
        """
        self.root_dir = os.path.join(root_dir, split)
        
        self.transforms_face = transforms_face
        if (not self.transforms_face):
            self.transforms_face = A.Compose([
                ToTensorV2()
            ])

        self.transforms_sphere = transforms_sphere
        if (not self.transforms_sphere):
            self.transforms_sphere = A.Compose([
                ToTensorV2()
            ])

        # Load and process sphere mask
        try:
            # Load mask, convert to grayscale and then binary
            sphere_mask = Image.open(mask_path).convert('L')
            # Convert to numpy array
            sphere_mask_np = np.array(sphere_mask)
            # Convert to binary (0/1)
            threshold = 128  # Adjust threshold as needed
            self.mask = (sphere_mask_np > threshold).astype(np.float32)
            print(f"Loaded sphere mask from {mask_path}, shape: {self.mask.shape}")

            self.mask_3d = np.repeat(self.mask[:, :, np.newaxis], 3, axis=2)
            self.maskTensor = self.transforms_sphere(image=self.mask_3d)['image'].int().float()

        except Exception as e:
            print(f"Warning: Could not load sphere mask from {mask_path}: {e}")
            self.mask = None
            self.maskTensor = None

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

        if (self.transforms_face):
            face_img = face_img.astype(np.float32) / 255.0
            augmented_face = self.transforms_face(image=face_img)
            face_img = augmented_face['image']  # Should now be [C, H, W]

        if (self.transforms_sphere):
            sphere_img = sphere_img.astype(np.float32) / 255.0
            augmented_sphere = self.transforms_sphere(image=sphere_img)
            sphere_img = augmented_sphere['image']

        # Ensure tensors are floating point type
        if isinstance(face_img, torch.Tensor) and face_img.dtype != torch.float32:
            face_img = face_img.float()
        
        if isinstance(sphere_img, torch.Tensor) and sphere_img.dtype != torch.float32:
            sphere_img = sphere_img.float()
        
        return face_img, sphere_img
