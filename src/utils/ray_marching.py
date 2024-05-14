import torch
import numpy as np

def get_rays_sample_space(H, W, focal, c2w, near, far, N_samples, rand=False):
   if isinstance(c2w, np.ndarray):
       c2w = torch.from_numpy(c2w).float()

   i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32), indexing='xy')
    
   dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], dim=-1)
    
   rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    
   rays_o = c2w[:3, -1].expand(rays_d.shape)
    
   z_vals = torch.linspace(near, far, N_samples)
   z_vals = z_vals.expand(rays_o.shape[0], rays_o.shape[1], N_samples)
   z_vals = z_vals.clone() 
   if rand:
       z_vals += torch.rand(rays_o.shape[0], rays_o.shape[1], N_samples) * (far - near) / N_samples
   #Calculate 3D positions of samples along rays
   pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    
   #Flatten the points tensor
   pts_flat = pts.view(-1, 3)
    
   return pts_flat, z_vals

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



def get_rays_sample_space_gpu(H, W, focal, c2w, near, far, N_samples, rand=False, device=torch.device('cpu')):
   if isinstance(c2w, np.ndarray):
       c2w = torch.from_numpy(c2w).float()

   i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32), indexing='xy')
    
   dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], dim=-1)
    
   rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    
   rays_o = c2w[:3, -1].expand(rays_d.shape)
    
   z_vals = torch.linspace(near, far, N_samples)
   z_vals = z_vals.expand(rays_o.shape[0], rays_o.shape[1], N_samples)
   z_vals = z_vals.clone() 
   if rand:
       z_vals += torch.rand(rays_o.shape[0], rays_o.shape[1], N_samples) * (far - near) / N_samples
   #Calculate 3D positions of samples along rays
   pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    
   #Flatten the points tensor
   pts_flat = pts.view(-1, 3)
    
   return pts_flat, z_vals

def render_rays_gpu(sigma_a, rgb, z_vals, device=torch.device('cuda')):
    sigma_a = sigma_a.to(device)
    rgb = rgb.to(device)
    z_vals = z_vals.to(device)

    # Calculate distances between z values
    dists = torch.cat([
        z_vals[..., 1:] - z_vals[..., :-1], 
        torch.full(z_vals[..., :1].shape, 1e10, device=device)  # Ensure new tensor is also on GPU
    ], -1)

    # Compute alpha using the calculated distances
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    
    # Prepare for cumulative product calculation
    padded_alpha = torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1)
    weights = alpha * torch.cumprod(padded_alpha, dim=-1)[..., 1:]

    # Compute RGB map, depth map, and accumulation map
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    return rgb_map, depth_map, acc_map
