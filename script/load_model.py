
import sys
import os
sys.path.append('../src')

from data_load import test_train_split
from model import MyModel
from pos_encoding import posenc
from utils.ray_marching import get_rays_sample_space
from utils.ray_marching import render_rays
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse

model = MyModel(D=8, W=256, L_embed=6)

parser = argparse.ArgumentParser(description='Input samples for the training process.')
parser.add_argument('--npz_file', type=str, required=True,
                        help='compressed numpy where you have: images, poses and focal info')
parser.add_argument('--model_path', type=str, required=True,
                        help='compressed numpy where you have: images, poses and focal info')


args = parser.parse_args()
# data prep 
data_path = args.npz_file
model_path = args.model_path
data_name = os.path.splitext(os.path.basename(data_path))[0]

# preapring model arch
model = MyModel(D=8, W=256, L_embed=6)
model.load_state_dict(torch.load(model_path))
model.eval() 
print(model)

# loading data
H, W, train, trainpose, eval, evalpose, test, testpose, focal = test_train_split(data_path)
test = torch.from_numpy(test)
pts_flat, z_vals, rays_o, rays_d = get_rays_sample_space(H, W, focal, testpose, 2., 6., 64, rand=False)
pts_flat_enc = posenc(pts_flat, 6)
with torch.no_grad(): 
    predictions = model(pts_flat_enc)
    predictions = predictions.view(100, 100, 64, 4)

sigma_a = F.relu(predictions[..., 3]) # extracting density
rgb = torch.sigmoid(predictions[..., :3]) # extracting color
sigma_a.shape, rgb.shape
rgb_render, depth, acc = render_rays(sigma_a, rgb, z_vals)

print(rgb_render.shape, test.shape)
loss = torch.mean((rgb_render - test) ** 2)
psnr = -10. * torch.log10(loss)
print(psnr)
fig, axs = plt.subplots(1, 2, figsize=(10, 5)) 

axs[0].imshow(test)
axs[0].set_title('Original Image')
axs[0].axis('off')  

axs[1].imshow(rgb_render)
axs[1].set_title('Rendered Image')
axs[1].axis('off')

plt.show()