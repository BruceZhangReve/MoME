import torch
import torch.nn as nn
import torch.nn.functional as F
#from models.layers import *
#from models.patch_layer import *

class TSLinear(nn.Module):
    def __init__(self, L, T):
        super(TSLinear, self).__init__()
        self.fc = nn.Linear(L, T)

    def forward(self, x):
        return self.fc(x)

class TSMixer(nn.Module):
    def __init__(self, args):
        super(TSMixer, self).__init__()
        self.n_vars = args.n_vars
        self.in_len = args.in_len
        self.out_len = args.out_len
        self.d_model = args.hidden_dim
        self.d_ff = 2 * args.hidden_dim
        self.device = args.device #torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        self.mixer_layers = []
        self.n_mixer = args.n_layers
        for i in range(self.n_mixer):
            self.mixer_layers.append(MixerLayer(self.n_vars, self.in_len, self.d_ff, args.dropout)) 
        self.mixer_layers = nn.ModuleList(self.mixer_layers)
        self.temp_proj = TemporalProj(self.in_len, self.out_len)

        
    def forward(self, x):
        for i in range(self.n_mixer):
            x = self.mixer_layers[i](x) #[bs, in_seq, n_vars]
        x = self.temp_proj(x) # x: [bs, out_seq, n_vars]
        
        return x

class MLPTime(nn.Module):
    def __init__(self, n_vars, seq_len, dropout_rate):
        super(MLPTime, self).__init__()
        self.fc = nn.Linear(seq_len, seq_len)
        self.n_vars = n_vars
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class MLPFeat(nn.Module):
    def __init__(self, C, ff_dim, dropout_rate=0.1):
        super(MLPFeat, self).__init__()
        self.fc1 = nn.Linear(C, ff_dim)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ff_dim, C)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class MixerLayer(nn.Module):
    def __init__(self, n_vars, seq_len, ff_dim, dropout):
        super(MixerLayer, self).__init__()
        self.mlp_time = MLPTime(n_vars, seq_len, dropout)
        self.mlp_feat = MLPFeat(n_vars, ff_dim, dropout)

    def batch_norm_2d(self, x):
        # x has shape (B, L, C) 
        #[bs, in_seq, n_vars]
        return (x - x.mean()) / x.std()
    
    def forward(self, x):
        # x has shape (B, L, C) 
        res_x = x

        x = x.transpose(1, 2) #[bs, in_seq, n_vars] -> [bs, n_vars, in_seq]

        x = self.mlp_time(x) #[bs, n_vars, in_seq], shared MLP across channels if non-ccm

        x = x.transpose(1, 2) + res_x #[bs, n_vars, in_seq] -> [bs, in_seq, n_vars]
        res_x = x

        x = self.mlp_feat(x) + res_x #[bs, in_seq, n_vars] -> [bs, in_seq, n_vars]
        return x

class TemporalProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TemporalProj, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x: [bs, seq_len, n_vars]
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(1, 2)
        return x  #[bs, seq_len, n_vars]

        


class Flatten_Head(nn.Module):
    def __init__(self, n_vars, d_model, target_window, head_dropout=0.2):
        super().__init__()
        self.n_vars = n_vars
        self.linears = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.flattens = nn.ModuleList()
        for i in range(self.n_vars):
            self.flattens.append(nn.Flatten(start_dim=-2))
            self.linears.append(nn.Linear(d_model, target_window))
            self.dropouts.append(nn.Dropout(head_dropout))
            
    def forward(self, x):        # x: [bs, nvars, d_model];
        x_out = []
        for i in range(self.n_vars):
            z = x[:,i,:]          # z: [bs, d_model]
            z = self.linears[i](z)                    # z: [bs, target_window]
            z = self.dropouts[i](z)
            x_out.append(z)
        x = torch.stack(x_out, dim=1)                 # x: [bs, nvars, target_window]
        return x            #x: [bs, nvars, target_window]