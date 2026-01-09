# dataset.py
import cv2
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config

# Define standard ImageNet normalization
transform_pipeline = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_video_frames(video_path):
    """
    Reads a video and returns a Tensor of shape (10, 3, 224, 224).
    """
    # 1. Safety Check: If file is missing, return zeros
    if not os.path.exists(video_path):
        return torch.zeros(config.FRAMES_PER_VIDEO, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)

    # Suppress OpenCV warnings
    os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 2. Safety Check: If video is empty/broken
    if total_frames == 0:
        cap.release()
        return torch.zeros(config.FRAMES_PER_VIDEO, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)

    # 3. Stratified Sampling
    skip = max(int(total_frames / config.FRAMES_PER_VIDEO), 1)

    for i in range(config.FRAMES_PER_VIDEO):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Apply transforms immediately ,it becomes Tensor
            frame_tensor = transform_pipeline(frame)
            frames.append(frame_tensor)
        else:
            break
    cap.release()
    
    # 4. Padding (if video had fewer than 10 frames)
    while len(frames) < config.FRAMES_PER_VIDEO:
        frames.append(torch.zeros(3, config.IMAGE_SIZE, config.IMAGE_SIZE))
        
    # 5. stacking the list of tensor to one big tensor of shape (10, 3, 224, 224)
    return torch.stack(frames)

def find_video_path(filename):
    """
    Locates video in subfolders.
    """
    real_path = os.path.join(config.TRAIN_FOLDER, 'real', filename)
    if os.path.exists(real_path): return real_path
    
    fake_path = os.path.join(config.TRAIN_FOLDER, 'fake', filename)
    if os.path.exists(fake_path): return fake_path
    
    return os.path.join(config.TRAIN_FOLDER, filename)

class LazyDeepfakeDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Use filename column (Adjust 'Video_Name' if needed)
        video_name = row['filename'] 
        full_path = find_video_path(video_name)
        
        video_tensor = load_video_frames(full_path)
        label = torch.tensor(row['Label'], dtype=torch.float32)
        
        return video_tensor, label

def get_data_loaders():
    print("Initializing Lazy Loader...")
    full_dataset = LazyDeepfakeDataset(config.TRAIN_CSV)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader