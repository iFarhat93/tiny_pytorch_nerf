import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class MyModel(nn.Module):
    def __init__(self, widths, L_embed=6, use_dropout=False, use_batch_norm=False):
        super(MyModel, self).__init__()
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
        
        previous_width = input_dim
        for width in widths:
            layer = nn.Linear(previous_width, width)
            init.xavier_uniform_(layer.weight)
            init.zeros_(layer.bias)
            self.layers.append(layer)
            
            if self.use_batch_norm:
                self.norms.append(nn.BatchNorm1d(width))
            if self.use_dropout:
                self.dropouts.append(nn.Dropout(0.1))
            
            previous_width = width  # Update the input dimension for the next layer
        
        # The output layer now takes the last width in the list as its input size
        self.output_layer = nn.Linear(widths[-1], 4)
        init.xavier_uniform_(self.output_layer.weight)
        init.zeros_(self.output_layer.bias)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_batch_norm and i < len(self.norms):  # Check added to avoid out of index errors
                x = self.norms[i](x)
            x = F.relu(x)
            if self.use_dropout and i < len(self.dropouts):  # Check added to avoid out of index errors
                x = self.dropouts[i](x)
        x = self.output_layer(x)
        return x

