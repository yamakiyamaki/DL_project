import os
import random
import shutil
import argparse


def split_dataset(data_root, train_ratio=0.7, val_ratio=0.1, random_seed=351):
    """
    Split the dataset into train, validation, and test sets.
    
    Args:
        data_root: Path to the directory containing all images
        train_ratio: Ratio of training data (default 0.7)
        val_ratio: Ratio of validation data (default 0.1)
        random_seed: Random seed for reproducibility (default 351)
    """
    renders_dir = os.path.join(data_root, "renders")
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")
    
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Check if split already exists
    if len(os.listdir(train_dir)) > 0 and len(os.listdir(val_dir)) > 0 and len(os.listdir(test_dir)) > 0:
        print("Dataset already split")
        return
    
    # Get all face images
    face_files = [f for f in os.listdir(renders_dir) if "_Face.png" in f]
    
    # Group face images by face ID
    face_groups = {}
    for file in face_files:
        # Extract face ID (e.g., 'Face00')
        face_id = file.split('_It')[0] # int(file.split('_It')[0][4:])
        if face_id not in face_groups:
            face_groups[face_id] = []
        face_groups[face_id].append(file)
    
    # Get sphere images
    sphere_files = [f for f in os.listdir(renders_dir) if "_Sphere.png" in f]
    
    # Determine which face IDs go to train, validation, and test sets
    face_ids = list(face_groups.keys())
    random.seed(random_seed)  # For reproducibility
    random.shuffle(face_ids)
    
    num_train = int(len(face_ids) * train_ratio)
    num_val = int(len(face_ids) * val_ratio)
    
    train_face_ids = set(face_ids[:num_train])
    val_face_ids = set(face_ids[num_train:num_train+num_val])
    test_face_ids = set(face_ids[num_train+num_val:])
    
    print(f"Split dataset: {len(train_face_ids)} faces for training, {len(val_face_ids)} faces for validation, {len(test_face_ids)} faces for testing")

    # Print which faces are in validation and test sets
    print("\nValidation set faces:")
    for face_id in sorted(val_face_ids):
        print(f"  {face_id}")
        
    print("\nTest set faces:")
    for face_id in sorted(test_face_ids):
        print(f"  {face_id}")
    
    # Copy files to their respective directories
    for file in face_files:
        face_id = file.split('_It')[0]
        src = os.path.join(renders_dir, file)
        
        if face_id in train_face_ids:
            dst_dir = train_dir
        elif face_id in val_face_ids:
            dst_dir = val_dir
        else:
            dst_dir = test_dir
            
        dst = os.path.join(dst_dir, file)
        shutil.copy2(src, dst)
    
    for file in sphere_files:
        face_id = file.split('_It')[0]
        src = os.path.join(renders_dir, file)
        
        if face_id in train_face_ids:
            dst_dir = train_dir
        elif face_id in val_face_ids:
            dst_dir = val_dir
        else:
            dst_dir = test_dir
            
        dst = os.path.join(dst_dir, file)
        shutil.copy2(src, dst)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Split dataset into train, validation and test sets')
    parser.add_argument('--data_dir', type=str, default='./data/dataset_256px_16f_100im/',
                        help='Directory containing the dataset (default: ./data/dataset_256px_16f_100im/)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of data to use for training (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Ratio of data to use for validation (default: 0.1)')
    parser.add_argument('--random_seed', type=int, default=351,
                        help='Random seed for reproducibility (default: 351)')
    
    args = parser.parse_args()
    
    print(f"Dataset split parameters:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Train ratio: {args.train_ratio}")
    print(f"  Validation ratio: {args.val_ratio}")
    print(f"  Test ratio: {1 - args.train_ratio - args.val_ratio}")
    print(f"  Random seed: {args.random_seed}")
    
    # Split the dataset
    split_dataset(args.data_dir, args.train_ratio, args.val_ratio, args.random_seed)
    
    print("Dataset splitting complete!")