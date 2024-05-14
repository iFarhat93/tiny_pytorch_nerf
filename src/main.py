from data_load import test_train_split
from data_load import EDU_NeRFDataset
from model import MyModel
from pos_encoding import posenc
from utils.ray_marching import get_rays_sample_space
from utils.ray_marching import render_rays
from utils.parser import args_prs_train
from utils.cache import clear_all
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm_notebook as tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim


clear_all()

width, pos_enc_l, N_samples, N_iters, save_i, data_path, i_plot, batch_norm, dropout = args_prs_train() # parse arguments 

H, W, train, trainpose, eval, evalpose, test, testpose, focal, data_name = test_train_split(data_path)
print(width, pos_enc_l, N_samples, N_iters, save_i, data_path, i_plot, batch_norm, dropout)

writer = SummaryWriter(f'../logs/{data_name}/') # tensorboard writer


eval = torch.from_numpy(eval)


model = MyModel(widths=width, L_embed=pos_enc_l, use_dropout=dropout, use_batch_norm=batch_norm)


optimizer = torch.optim.Adam(model.parameters(), lr=5e-4) 
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

train_dataset = EDU_NeRFDataset(train, trainpose)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
data_iter = iter(train_dataloader)

loss = nn.MSELoss()
t = time.time()

pts_flat_eval, z_vals_eval = get_rays_sample_space(H, W, focal, evalpose, 2., 6., N_samples, rand=True) # prepare the one-time eval 3D sampled space
pts_flat_enc_eval = posenc(pts_flat_eval, pos_enc_l)  # positional encoding
best_loss = 100
eval_loss = 102

for i in range(N_iters+1):
    try:
        target, pose = next(data_iter)
    except StopIteration:
        data_iter = iter(train_dataloader)
        target, pose = next(data_iter)
    
    target = target.squeeze(0) # Now shape: [100, 100, 3]
    pose = pose.squeeze(0).numpy()  
    #plt.imshow(target)
    #plt.show()
    pts_flat, z_vals = get_rays_sample_space(H, W, focal, pose, 2., 6., N_samples, rand=True) # sampling 3D space  
    
    pts_flat_enc = posenc(pts_flat, pos_enc_l)  # positional encoding

    model.train()
       
    optimizer.zero_grad()

    raw = model(pts_flat_enc)


    raw = raw.view(H, W, N_samples, 4)
    
    sigma_a = F.relu(raw[..., 3]) 
    rgb = torch.sigmoid(raw[..., :3]) 
    
    rgb_render, depth, acc = render_rays(sigma_a, rgb, z_vals)
    
    train_loss = loss(rgb_render, target)
    
    train_loss.backward()
    
    optimizer.step()
    #scheduler.step() 

    if i%i_plot==0 and i!=0:

        with torch.no_grad():
            model.eval() 
            eval_raw = model(pts_flat_enc_eval).view(100, 100, N_samples, 4)
            #print(eval_raw)
            eval_sigma_a = F.relu(eval_raw[..., 3]) 
            eval_rgb = torch.sigmoid(eval_raw[..., :3])
            rgb_render_eval, depth_map, acc_map = render_rays(eval_sigma_a, eval_rgb, z_vals_eval)
            
            eval_loss = loss(rgb_render_eval, eval)
            eval_psnr = -10. * torch.log10(eval_loss)
        #print(rgb_render.shape)
        train_psnr = -10. * torch.log10(train_loss)

        print(f"Epoch {i}: eval loss = {eval_loss.item()} and eval psnr = {eval_psnr.item()}")
        rgb_render_rep = rgb_render_eval.permute(2, 0, 1) # prepare for tensor board

        writer.add_scalar('Loss/eval', eval_loss.item(), i)
        writer.add_scalar('Loss/train', train_loss.item(), i)
        writer.add_scalar('Metrics/Train PSNR', train_psnr.item(), i)
        writer.add_scalar('Metrics/Eval PSNR', eval_psnr.item(), i)
        writer.add_image('rendered image', rgb_render_rep, global_step=i)


    if i%save_i==0 and i>1500 :
        model_path = f'../logs/{data_name}/model_state_dict_step_latest.pth'
        torch.save(model.state_dict(), model_path)
        print(f"latest model saved at iteration {i} with loss {best_loss}")
    if i > 50 and eval_loss<best_loss :
        best_loss = eval_loss  # Update the best known loss
        model_path = f'../logs/{data_name}/model_state_dict_step_best.pth'
        torch.save(model.state_dict(), model_path)
        print(f"New best model saved at iteration {i} with loss {best_loss}")


total_t = time.time() - t
print('Done, total time: '+str(total_t))
writer.close() 
