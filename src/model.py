import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, D=8, W=256, L_embed=6):
        super(MyModel, self).__init__()
        self.D = D
        if L_embed is None:
            raise ValueError("L_embed must be provided")
        input_dim = 3 + 3*2*L_embed
        layers = [nn.Linear(input_dim, W)] 
        for i in range(1, D):
            layers.append(nn.Linear(W, W))
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(W, 4)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        x = self.output_layer(x)
        return x
