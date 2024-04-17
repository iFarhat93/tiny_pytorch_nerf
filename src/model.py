import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# class MyModel(nn.Module):
#     def __init__(self, D=8, W=256, L_embed=6):
#         super(MyModel, self).__init__()
#         self.D = D
#         if L_embed is None:
#             raise ValueError("L_embed must be provided")
#         input_dim = 3 + 3 * 2 * L_embed  # Calculate input dimension based on embedding length
        
#         first_layer = nn.Linear(input_dim, W)
#         init.xavier_uniform_(first_layer.weight)  
#         init.zeros_(first_layer.bias)  
        
        
#         layers = [first_layer]
#         for i in range(1, D):
#             layer = nn.Linear(W, W)
#             init.xavier_uniform_(layer.weight)  
#             init.zeros_(layer.bias) 
#             layers.append(layer)
        
#         self.layers = nn.ModuleList(layers)
        
#         self.output_layer = nn.Linear(W, 4)
#         init.xavier_uniform_(self.output_layer.weight) 
#         init.zeros_(self.output_layer.bias)  

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#             x = torch.relu(x)
#         x = self.output_layer(x)
#         return x


class MyModel(nn.Module):
    def __init__(self, D=8, W=256, L_embed=6, use_dropout=False, use_batch_norm=False):
        super(MyModel, self).__init__()
        self.D = D
        if L_embed is None:
            raise ValueError("L_embed must be provided")
        input_dim = 3 + 3 * 2 * L_embed  # Calculate input dimension based on embedding length
        
        self.layers = nn.ModuleList()
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm
        
        if self.use_batch_norm:
            self.norms = nn.ModuleList()
        if self.use_dropout:
            self.dropouts = nn.ModuleList()
        
        for i in range(D):
            layer = nn.Linear(input_dim if i == 0 else W, W)
            init.xavier_uniform_(layer.weight)
            init.zeros_(layer.bias)
            self.layers.append(layer)
            
            if self.use_batch_norm:
                self.norms.append(nn.BatchNorm1d(W))
            if self.use_dropout:
                self.dropouts.append(nn.Dropout(0.1))
        
        self.output_layer = nn.Linear(W, 4)
        init.xavier_uniform_(self.output_layer.weight)
        init.zeros_(self.output_layer.bias)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_batch_norm:
                x = self.norms[i](x)
            x = F.relu(x)
            if self.use_dropout:
                x = self.dropouts[i](x)
        x = self.output_layer(x)
        return x




class EnhancedModel(nn.Module):
    def __init__(self, D=8, W=256, L_embed=6, use_attention=True):
        super(EnhancedModel, self).__init__()
        self.D = D
        self.use_attention = use_attention
        input_dim = 3 + 3 * 2 * L_embed

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        first_layer = nn.Linear(input_dim, W)
        init.xavier_uniform_(first_layer.weight)
        self.layers.append(first_layer)
        self.norms.append(nn.GroupNorm(32, W))  # Using GroupNorm

        for i in range(1, D):
            layer = nn.Linear(W, W)
            init.xavier_uniform_(layer.weight)
            self.layers.append(layer)
            self.norms.append(nn.GroupNorm(32, W))  # Consistent use of GroupNorm

        if self.use_attention:
            self.attention = nn.MultiheadAttention(W, num_heads=8, dropout=0.1)
        
        self.output_layer = nn.Linear(W, 4)
        init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        x = F.leaky_relu(self.layers[0](x))  # Changed to LeakyReLU
        x = self.norms[0](x)
        
        for layer, norm in zip(self.layers[1:], self.norms[1:]):
            identity = x
            x = F.leaky_relu(layer(x))
            x = norm(x)
            if self.use_attention:
                x, _ = self.attention(x, x, x)
            x += identity  # Residual Connection
        
        x = self.output_layer(x)
        return x