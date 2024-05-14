import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from data_load import test_train_split, EDU_NeRFDataset
from model import MyModel
from pos_encoding import posenc
from utils.ray_marching import get_rays_sample_space_gpu, render_rays_gpu
from utils.parser import args_prs_train
from utils.cache import clear_all

# GPU device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

clear_all()

width, pos_enc_l, N_samples, N_iters, save_i, data_path, i_plot, batch_norm, dropout = args_prs_train()
H, W, train, trainpose, eval, evalpose, test, testpose, focal, data_name = test_train_split(data_path)

print(width, pos_enc_l, N_samples, N_iters, save_i, data_path, i_plot, batch_norm, dropout)
writer = SummaryWriter(f'../logs/{data_name}/')

eval = torch.from_numpy(eval).to(device)

model = MyModel(widths=width, L_embed=pos_enc_l, use_dropout=dropout, use_batch_norm=batch_norm).to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-4)
train_dataset = EDU_NeRFDataset(train, trainpose)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
data_iter = iter(train_dataloader)

loss = nn.MSELoss()
t = time.time()

pts_flat_eval, z_vals_eval = get_rays_sample_space_gpu(H, W, focal, evalpose, 2., 6., N_samples, rand=True, device=device) # prepare the one-time eval 3D sampled space
pts_flat_enc_eval = posenc(pts_flat_eval, pos_enc_l).to(device)  # positional encoding
best_loss = 100
eval_loss = 102

for i in range(N_iters + 1):
    try:
        target, pose = next(data_iter)
    except StopIteration:
        data_iter = iter(train_dataloader)
        target, pose = next(data_iter)
    
    target = target.squeeze(0).to(device) 
    pose = pose.squeeze(0).numpy()
    
    
    pts_flat, z_vals = get_rays_sample_space_gpu(H, W, focal, pose, 2., 6., N_samples, rand=True, device=device)
    pts_flat_enc = posenc(pts_flat, pos_enc_l).to(device)  
    
    optimizer.zero_grad()
    model.train()
    raw = model(pts_flat_enc)
    raw = raw.view(H, W, N_samples, 4)
    
    # Calculate loss and backpropagate
    sigma_a = F.relu(raw[..., 3])
    rgb = torch.sigmoid(raw[..., :3])
    rgb_render, depth, acc = render_rays_gpu(sigma_a, rgb, z_vals, device=device)
    train_loss = loss(rgb_render, target)
    train_loss.backward()
    optimizer.step()
    
    # Logging and saving models
    if i%i_plot==0 and i!=0:
        # Evaluation step
        model.eval()
        with torch.no_grad():
            eval_raw = model(pts_flat_enc_eval).view(100, 100, N_samples, 4)
            eval_sigma_a = F.relu(eval_raw[..., 3])
            eval_rgb = torch.sigmoid(eval_raw[..., :3])
            rgb_render_eval, depth_map, acc_map = render_rays_gpu(eval_sigma_a, eval_rgb, z_vals_eval, device=device)
            eval_loss = loss(rgb_render_eval, eval)
            eval_psnr = -10. * torch.log10(eval_loss)
        
        # TensorBoard logging
        print(f"Epoch {i}: eval loss = {eval_loss.item()} and eval psnr = {eval_psnr.item()}")
        train_psnr = -10. * torch.log10(train_loss)
        rgb_render_rep = rgb_render_eval.permute(2, 0, 1)
        writer.add_scalar('Loss/eval', eval_loss.item(), i)
        writer.add_scalar('Loss/train', train_loss.item(), i)
        writer.add_scalar('Metrics/Train PSNR', train_psnr.item(), i)
        writer.add_scalar('Metrics/Eval PSNR', eval_psnr.item(), i)
        writer.add_image('Rendered image', rgb_render_rep, global_step=i)
        
        # Save model checkpoints
        if i%save_i==0 and i>1500 :
            model_path = f'../logs/{data_name}/model_state_dict_step_latest.pth'
            torch.save(model.state_dict(), model_path)
            print(f"latest model saved at iteration {i} with loss {best_loss}")
        if i > 50 and eval_loss<best_loss :
            best_loss = eval_loss  # Update the best known loss
            model_path = f'../logs/{data_name}/model_state_dict_step_best.pth'
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved at iteration {i} with loss {best_loss}")

# Close resources and report
writer.close()
total_time = time.time() - t
print('Done, total time:', total_time)

