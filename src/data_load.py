import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import os

def load_data(data_path):
    data = np.load(data_path)
    images = data['images']
    poses = data['poses'].astype(np.float32)
    
    if 'focal' in data.files:
        focal = data['focal']
    else:
        focal = np.array(128.478)
    
    return images, poses, focal

def test_train_split(data_path): 
    images, poses, focal = load_data(data_path)
    H, W = images.shape[1:3]
    #print(images.shape, poses.shape, focal)

    eval, evalpose = images[images.shape[0]-4], poses[poses.shape[0]-2]
    test, testpose = images[images.shape[0]-1], poses[poses.shape[0]-1]
    train, trainpose = images[:images.shape[0]-5,...,:3], poses[:poses.shape[0]-5]
    data_name = os.path.splitext(os.path.basename(data_path))[0]
    return H, W, train, trainpose, eval, evalpose, test, testpose, focal, data_name

class NeRFDataset(Dataset):
    def __init__(self, images, poses):
        self.images = images
        self.poses = poses

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        pose = self.poses[idx]
        return torch.from_numpy(image).float(), pose