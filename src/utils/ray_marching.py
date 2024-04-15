import tensorflow as tf
import torch
import numpy as np

def get_rays_sample_space(H, W, focal, c2w, near, far, N_samples, rand=False):
    
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32), tf.range(H, dtype=tf.float32), indexing='xy')
    dirs = tf.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -tf.ones_like(i)], -1)
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = tf.broadcast_to(c2w[:3,-1], tf.shape(rays_d))

    z_vals = tf.linspace(near, far, N_samples)
    if rand:
      z_vals += tf.random.uniform(list(rays_o.shape[:-1]) + [N_samples]) * (far-near)/N_samples
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

    # Run network
    pts_flat = tf.reshape(pts, [-1,3])
    
    pts_flat_torch = torch.from_numpy(pts_flat.numpy())
    rays_o_torch = torch.from_numpy(rays_o.numpy())
    rays_d_torch = torch.from_numpy(rays_d.numpy())
    z_vals_torch = torch.from_numpy(z_vals.numpy())
    return pts_flat_torch, z_vals_torch, rays_o_torch, rays_d_torch

def render_rays(sigma_a, rgb, z_vals):
    # Calculate distances between z values
    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1], torch.full(z_vals[..., :1].shape, 1e10, device=z_vals.device)], -1)
    #print(sigma_a.shape, dists.shape)
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    
    padded_alpha = torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1)
    weights = alpha * torch.cumprod(padded_alpha, dim=-1)[..., 1:]

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    return rgb_map, depth_map, acc_map