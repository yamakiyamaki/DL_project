import os
import random
import shutil

def split_dataset(data_root, train_ratio=0.8):
    """
    Split the dataset into train and test sets.
    
    Args:
        data_root: Path to the directory containing all images
        train_ratio: Ratio of training data (default 0.8)
    """
    renders_dir = os.path.join(data_root, "renders")
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")
    
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Check if split already exists
    if len(os.listdir(train_dir)) > 0 and len(os.listdir(test_dir)) > 0:
        print("Dataset already split")
        return
    
    # Get all face images
    face_files = [f for f in os.listdir(renders_dir) if "_Face.png" in f]
    
    # Group face images by face ID
    face_groups = {}
    for file in face_files:
        # Extract face ID (e.g., 'Face00')
        face_id = int(file.split('_It')[0][4:])
        if face_id not in face_groups:
            face_groups[face_id] = []
        face_groups[face_id].append(file)
    
    # Get sphere images
    sphere_files = [f for f in os.listdir(renders_dir) if "_Sphere.png" in f]
    
    # Determine which face IDs go to train and test sets
    face_ids = list(face_groups.keys())
    random.seed(42)  # For reproducibility
    random.shuffle(face_ids)
    
    num_train = int(len(face_ids) * train_ratio)
    train_face_ids = set(face_ids[:num_train])
    test_face_ids = set(face_ids[num_train:])
    
    print(f"Split dataset: {len(train_face_ids)} faces for training, {len(test_face_ids)} faces for testing")
    
    # Copy files to their respective directories
    for file in face_files:
        face_id = file.split('_It')[0]
        src = os.path.join(renders_dir, file)
        dst_dir = train_dir if face_id in train_face_ids else test_dir
        dst = os.path.join(dst_dir, file)
        shutil.copy2(src, dst)
    
    for file in sphere_files:
        face_id = file.split('_It')[0]
        src = os.path.join(renders_dir, file)
        dst_dir = train_dir if face_id in train_face_ids else test_dir
        dst = os.path.join(dst_dir, file)
        shutil.copy2(src, dst)

# Split the dataset
split_dataset('./data/dataset_256px_11f_100im/')