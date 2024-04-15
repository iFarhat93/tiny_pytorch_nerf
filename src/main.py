from data_load import test_train_split
from data_load import NeRFDataset
from model import MyModel
from pos_encoding import posenc
from utils.ray_marching import get_rays_sample_space
from utils.ray_marching import render_rays
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

# Initialize parser
parser = argparse.ArgumentParser(description='Input samples for the training process.')

parser.add_argument('--npz_file', type=str, required=True,
                    help='Compressed input numpy file containing: images, poses, and focal info')
parser.add_argument('--N_samples', type=int, required=False, default=64,
                    help='Number of samples in the 3D space (default: 64)')
parser.add_argument('--N_iter', type=int, required=False, default=1000,
                    help='Number of training iterations (default: 1000)')
parser.add_argument('--save_pts', type=int, required=False, default=100,
                    help='Save model every N iterations (default: 100)')
parser.add_argument('--depth', type=int, required=False, default=8,
                    help='Model depth (default: 8)')
parser.add_argument('--width', type=int, required=False, default=256,
                    help='Model width (default: 256)')
parser.add_argument('--pos_enc', type=int, required=False, default=6,
                    help='Positional encodings dimension (default: 6)')

# Parse arguments
args = parser.parse_args()

# Accessing argument values
depth = args.depth
width = args.width
pos_enc_l = args.pos_enc
N_samples = args.N_samples
N_iters = args.N_iter
save_i = args.save_pts
data_path = args.npz_file
i_plot = 10

H, W, train, trainpose, eval, evalpose, test, testpose, focal, data_name = test_train_split(data_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel(D=depth, W=width, L_embed=pos_enc_l)

psnrs = []
iternums = []
rgbs = []

eval = torch.from_numpy(eval)
writer = SummaryWriter(f'../logs/{data_name}/')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

train_dataset = NeRFDataset(train, trainpose)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
data_iter = iter(train_dataloader)


pts_flat_eval, z_vals_eval = get_rays_sample_space(H, W, focal, evalpose, 2., 6., N_samples, rand=False)
pts_flat_enc_eval = posenc(pts_flat_eval, pos_enc_l)  # positional encoding
loss = nn.MSELoss()

t = time.time()
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
    pts_flat, z_vals = get_rays_sample_space(H, W, focal, pose, 2., 6., N_samples, rand=False) # sampling space accordingly 
    
    pts_flat_enc = posenc(pts_flat, pos_enc_l)  # positional encoding

    model.train()
       
    optimizer.zero_grad()

    raw = model(pts_flat_enc)
    
    raw = raw.view(H, W, N_samples, 4)
    
    sigma_a = F.relu(raw[..., 3]) # extracting density
    rgb = torch.sigmoid(raw[..., :3]) # extracting rgb color
    
    rgb_render, depth, acc = render_rays(sigma_a, rgb, z_vals)
    
    train_loss = loss(rgb_render, target)
    #print('lossssssssssssssssssss', train_loss)
    
    train_loss.backward()
    
    optimizer.step()
    #scheduler.step() 
    
    if i%i_plot==0 and i!=0:

        with torch.no_grad():

            eval_raw = model(pts_flat_enc_eval).view(100, 100, 64, 4)
            
            eval_sigma_a = F.relu(eval_raw[..., 3]) # extracting density
            eval_rgb = torch.sigmoid(eval_raw[..., :3]) # extracting rgb color
            rgb_render, depth_map, acc_map = render_rays(eval_sigma_a, eval_rgb, z_vals_eval)
            rgb_render_rep = rgb_render.permute(2, 0, 1) # prepare for tensor board
            #rgb_render_rep = torch.clamp(rgb_render_rep, 0, 1) # normalize
            eval_loss = loss(rgb_render, eval)
            eval_psnr = -10. * torch.log10(eval_loss)
        #print(rgb_render.shape)
        train_psnr = -10. * torch.log10(train_loss)

        print(f"Epoch {i}: train_Loss = {train_loss.item()} and eval_loss = {eval_loss.item()}")
        print("Train PSNR: ", train_psnr, "Eval PSNR: ", eval_psnr)
        
        writer.add_scalar('Loss/eval', eval_loss.item(), i)
        writer.add_scalar('Loss/train', train_loss.item(), i)
        writer.add_scalar('Metrics/Train PSNR', train_psnr.item(), i)
        writer.add_scalar('Metrics/Eval PSNR', eval_psnr.item(), i)
        writer.add_image('rendered image', rgb_render_rep, global_step=i)
        #psnrs.append(eval_psnr)  
        #iternums.append(i)
        #rgbs.append(rgb_render)

    if i%save_i==0 and i!=0:
        model_path = f'../logs/{data_name}/model_state_dict_step_{i}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"model saved at iteration {i}")
        

total_t = time.time() - t
print('Done, total time: '+total_t)
writer.close() 