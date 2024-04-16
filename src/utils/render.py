import sys
sys.path.append('../')

from data_load import load_data
from model import MyModel
from pos_encoding import posenc
from utils.ray_marching import get_rays_sample_space
from utils.ray_marching import render_rays
from utils.parser import args_prs_load
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
from base64 import b64encode
import imageio
from ipywidgets import interactive, widgets
import argparse


trans_t = lambda t : tf.convert_to_tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=tf.float32)

rot_phi = lambda phi : tf.convert_to_tensor([
    [1,0,0,0],
    [0,tf.cos(phi),-tf.sin(phi),0],
    [0,tf.sin(phi), tf.cos(phi),0],
    [0,0,0,1],
], dtype=tf.float32)

rot_theta = lambda th : tf.convert_to_tensor([
    [tf.cos(th),0,-tf.sin(th),0],
    [0,1,0,0],
    [tf.sin(th),0, tf.cos(th),0],
    [0,0,0,1],
], dtype=tf.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w


data_path, model_path = args_prs_load()

data_name = os.path.splitext(os.path.basename(data_path))[0]

model = MyModel(D=8, W=256, L_embed=6)
model.load_state_dict(torch.load(model_path))
model.eval() 
frames = []

images, poses, focal = load_data(data_path)
H, W = images.shape[1:3]

for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
    c2w = pose_spherical(th, -30., 4.)
    pts_flat, z_vals = get_rays_sample_space(H, W, focal, c2w[:3,:4], 2., 6., 64, rand=False)
    pts_flat_enc = posenc(pts_flat, 6)
    with torch.no_grad(): 
        predictions = model(pts_flat_enc)
        predictions = predictions.view(100, 100, 64, 4)

    sigma_a = F.relu(predictions[..., 3]) # extracting density
    rgb = torch.sigmoid(predictions[..., :3]) # extracting color
    sigma_a.shape, rgb.shape
    rgb_render, depth, acc = render_rays(sigma_a, rgb, z_vals)
    frames.append((255 * np.clip(rgb_render.cpu().detach().numpy(), 0, 1)).astype(np.uint8))
    
f = 'video.mp4'
imageio.mimwrite(f, frames, fps=30, quality=7)

mp4 = open(f'{data_name}.mp4','rb').read()
data_url = f"data:{data_name}/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls autoplay loop>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)