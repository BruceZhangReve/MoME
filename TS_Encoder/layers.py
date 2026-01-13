import torch
from torch import nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from .attention import *


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: BLC
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean  #output: BLC
    
    
    
def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        print(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)



class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x




class _Cluster_assigner(nn.Module):
    def __init__(self, n_vars, n_cluster, seq_len, d_model):
        super(_Cluster_assigner, self).__init__()
        self.n_vars = n_vars
        self.n_cluster = n_cluster
        self.linear = nn.Linear(seq_len, d_model)
        self.cluster = nn.Linear(d_model*2, 1)
        
        
    def forward(self, x, cluster_emb):     
        # x: [bs, seq_len, n_vars]
        # cluster_emb: [n_cluster, d_model]
        x = x.permute(0,2,1)
        x_emb = self.linear(x).reshape(-1, cluster_emb.shape[-1])      #[bs*n_vars, d_model]
        bn = x_emb.shape[0]
        bs = int(bn/self.n_vars)
        x_emb_batch = x_emb.repeat(self.n_cluster, 1)   
        cluster_emb_batch = torch.repeat_interleave(cluster_emb, bn, dim=0)
        out = torch.cat([x_emb_batch, cluster_emb_batch], dim=-1)
        prob = F.sigmoid(self.cluster(out)).squeeze(-1).reshape(self.n_cluster, bs, self.n_vars).permute(1,2,0)
        # prob: [bs, n_vars, n_cluster]
        prob_avg = torch.mean(prob, dim=0)      #[n_vars, n_cluster]
        prob_avg = F.softmax(prob_avg, dim=-1)
        return prob_avg


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        # self.fc3 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

class _Cluster_assigner(nn.Module):
    def __init__(self, n_vars, n_cluster, seq_len, d_model, device, epsilon=0.05):
        super(_Cluster_assigner, self).__init__()
        self.n_vars = n_vars
        self.n_cluster = n_cluster
        self.d_model = d_model
        self.epsilon = epsilon
        # linear_layer = [nn.Linear(seq_len, d_model), nn.ReLU(), nn.Linear(d_model, d_model)]
        # self.linear = MLP(seq_len, d_model)
        self.linear = nn.Linear(seq_len, d_model)
        self.cluster_emb = torch.empty(self.n_cluster, self.d_model).to(device) #nn.Parameter(torch.rand(n_cluster, in_dim * out_dim), requires_grad=True)
        nn.init.kaiming_uniform_(self.cluster_emb, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        
        
    def forward(self, x, cluster_emb):     
        # x: [bs, seq_len, n_vars]
        # cluster_emb: [n_cluster, d_model]
        n_vars = x.shape[-1]
        x = x.permute(0,2,1)
        x_emb = self.linear(x).reshape(-1, self.d_model)      #[bs*n_vars, d_model]
        bn = x_emb.shape[0]
        bs = max(int(bn/n_vars), 1) 
        prob = torch.mm(self.l2norm(x_emb), self.l2norm(cluster_emb).t()).reshape(bs, n_vars, self.n_cluster)
        # prob: [bs, n_vars, n_cluster]
        prob_temp = prob.reshape(-1, self.n_cluster)
        prob_temp = sinkhorn(prob_temp, epsilon=self.epsilon)
        mask = self.concrete_bern(prob_temp)   #[bs*n_vars, n_cluster]
        num_var_pc = torch.sum(mask, dim=0)
        adpat_cluster = torch.matmul(x_emb.transpose(0,1), mask)/(num_var_pc + 1e-6)  #[d_model, n_cluster]
        cluster_emb = cluster_emb + adpat_cluster.transpose(0,1)
        prob_avg = torch.mean(prob, dim=0)      #[n_vars, n_cluster]
        prob_avg = sinkhorn(prob_avg, epsilon=self.epsilon)
        return prob_avg, cluster_emb
    
    def concrete_bern(self, prob, temp = 0.07):
        random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(prob.device)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
        prob_bern = ((prob + random_noise) / temp).sigmoid()
        return prob_bern

    
class Cluster_assigner(nn.Module):
    def __init__(self, n_vars, n_cluster, seq_len, d_model, device='cuda', epsilon=0.05):
        super(Cluster_assigner, self).__init__()
        self.n_vars = n_vars
        self.n_cluster = n_cluster
        self.d_model = d_model
        self.epsilon = epsilon
        self.device = device
        self.linear = nn.Linear(seq_len, d_model)
        
        # Cluster embeddings
        self.cluster_emb = torch.empty(self.n_cluster, self.d_model).to(device)
        nn.init.kaiming_uniform_(self.cluster_emb, a=math.sqrt(5))
        
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.p2c = CrossAttention(d_model, n_heads=1)
        self.i = 0

    def forward(self, x, cluster_emb):     
        # x: [bs, seq_len, n_vars]
        # cluster_emb: [n_cluster, d_model]
        n_vars = x.shape[-1]
        x = x.permute(0,2,1)
        x_emb = self.linear(x).reshape(-1, self.d_model)      #[bs*n_vars, d_model]
        bn = x_emb.shape[0]
        bs = max(int(bn/n_vars), 1) 
        prob = torch.mm(self.l2norm(x_emb), self.l2norm(cluster_emb).t()).reshape(bs, n_vars, self.n_cluster)
        # prob: [bs, n_vars, n_cluster]
        prob_temp = prob.reshape(-1, self.n_cluster)
        prob_temp = sinkhorn(prob_temp, epsilon=self.epsilon)
        prob_avg = torch.mean(prob, dim=0)    #[n_vars, n_cluster]
        prob_avg = sinkhorn(prob_avg, epsilon=self.epsilon)
        mask = self.concrete_bern(prob_avg)   #[bs, n_vars, n_cluster]

        x_emb_ = x_emb.reshape(bs, n_vars,-1)
        cluster_emb_ = cluster_emb.repeat(bs,1,1)
        cluster_emb = self.p2c(cluster_emb_, x_emb_, x_emb_, mask=mask.transpose(0,1))
        cluster_emb_avg = torch.mean(cluster_emb, dim=0)
        #print(cluster_emb.shape, cluster_emb_.shape, x_emb_.shape, mask.shape)
    
        return prob_avg, cluster_emb_avg
     
    def concrete_bern(self, prob, temp = 0.07):
        random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(prob.device)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
        prob_bern = ((prob + random_noise) / temp).sigmoid()
        return prob_bern

# This is basically a softmax with teprature control
def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):   #[n_vars, n_cluster]
    Q = torch.exp(out / epsilon)
    sum_Q = torch.sum(Q, dim=1, keepdim=True) 
    Q = Q / (sum_Q)
    return Q


def cluster_aggregator(var_emb, mask):
    '''
        var_emb: (bs*patch_num, nvars, d_model)
        mask: (nvars, n_cluster)
        return: (bs*patch_num, n_cluster, d_model)
    '''
    num_var_pc = torch.sum(mask, dim=0)
    var_emb = var_emb.transpose(1,2)
    cluster_emb = torch.matmul(var_emb, mask)/(num_var_pc + 1e-6)
    cluster_emb = cluster_emb.transpose(1,2)
    return cluster_emb


    
class CrossAttention(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    input:
        queries: (bs, L, d_model)
        keys: (_, S, d_model)
        values: (bs, S, d_model)
        mask: (L, S)
    return: (bs, L, d_model)

    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout = 0.1):
        super(CrossAttention, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = MaskAttention(scale=None, attention_dropout = dropout)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, mask=None):
        # input dim: d_model
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = queries.view(B, L, H, -1)
        keys = keys.view(B, S, H, -1)
        values = values.view(B, S, H, -1)
       
        out = self.inner_attention(
            queries,
            keys,
            values,
            mask,
        )

        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)
       

        return out # B, L, d_model



#####################################################################################################################
#I think previous code has some issues, let me try another one

class Cluster_assigner_fixed(nn.Module):
    def __init__(self, n_vars, n_cluster, seq_len, d_model, device='cuda', epsilon=0.05):
        super(Cluster_assigner_fixed, self).__init__()
        self.n_vars = n_vars
        self.n_cluster = n_cluster
        self.d_model = d_model
        self.epsilon = epsilon
        self.device = device
        self.linear = nn.Linear(seq_len, d_model)
        
        # Cluster embeddings
        self.cluster_emb = torch.empty(self.n_cluster, self.d_model).to(device)
        nn.init.kaiming_uniform_(self.cluster_emb, a=math.sqrt(5))
        
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        #self.p2c = CrossAttention(d_model, n_heads=1)
        self.p2c = CrossAttention_fixed(d_model, n_heads=1)
        self.i = 0

    def forward(self, x, cluster_emb):     
        # x: [bs, seq_len, n_vars]
        # cluster_emb: [n_cluster, d_model]
        n_vars = x.shape[-1]
        x = x.permute(0,2,1)
        x_emb = self.linear(x).reshape(-1, self.d_model)      #[bs*n_vars, d_model]
        bn = x_emb.shape[0]
        bs = max(int(bn/n_vars), 1) 
        prob = torch.mm(self.l2norm(x_emb), self.l2norm(cluster_emb).t()).reshape(bs, n_vars, self.n_cluster)
        # prob: [bs, n_vars, n_cluster]
        prob_temp = prob.reshape(-1, self.n_cluster)
        prob_temp = sinkhorn(prob_temp, epsilon=self.epsilon)
        prob_avg = torch.mean(prob, dim=0)    #[n_vars, n_cluster]
        prob_avg = sinkhorn(prob_avg, epsilon=self.epsilon)
        mask = self.concrete_bern(prob_avg)   #[bs, n_vars, n_cluster]

        x_emb_ = x_emb.reshape(bs, n_vars,-1)
        cluster_emb_ = cluster_emb.repeat(bs,1,1)
        #detach() because I don't want this to influence x_embedding, is this code right?
        cluster_emb = self.p2c(cluster_emb_, x_emb_.detach(), x_emb_.detach(), mask=mask.transpose(0,1))
        cluster_emb_avg = torch.mean(cluster_emb, dim=0)
        #print(cluster_emb.shape, cluster_emb_.shape, x_emb_.shape, mask.shape)
    
        return prob_avg, cluster_emb_avg
     
    def concrete_bern(self, prob, temp = 0.07):
        random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(prob.device)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
        prob_bern = ((prob + random_noise) / temp).sigmoid()
        return prob_bern


class CrossAttention_fixed(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout=0.1):
        super(CrossAttention_fixed, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values
        self.mix = mix

        self.W_q = nn.Linear(d_model, n_heads * d_keys)
        self.W_k = nn.Linear(d_model, n_heads * d_keys)
        self.W_v = nn.Linear(d_model, n_heads * d_values)

        self.inner_attention = MaskAttention(scale=1 / math.sqrt(d_keys), attention_dropout=dropout)
        self.out_proj = nn.Linear(n_heads * d_values, d_model)

    def forward(self, queries, keys, values, mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        Q = self.W_q(queries).view(B, L, H, self.d_keys)
        K = self.W_k(keys).view(B, S, H, self.d_keys)
        V = self.W_v(values).view(B, S, H, self.d_values)

        # Call existing attention logic
        out = self.inner_attention(Q, K, V, mask)

        # Output projection
        if self.mix:
            out = out.transpose(1, 2).contiguous()
        return self.out_proj(out.view(B, L, -1))



##We want to accomodate Telecom data, where the challenge is to input and output both numerical and categorical items.

class Cat_Embed(nn.Module):
    def __init__(self, cat_indices, cat_vocab_size):
        """
        cat_indices: list[int], which channels are categorical
        cat_vocab_size: list[int], vocab size per categorical channel
        """
        super(Cat_Embed, self).__init__()
        self.cat_indices = cat_indices
        self.cat_vocab_size = cat_vocab_size
        self.embed_dim = 1 # a scalar value to represent the category, learnt through embedding layer

        self.embedders = nn.ModuleList([
            nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embed_dim)
            for vocab_size in self.cat_vocab_size
        ])

    def forward(self, x):
        """
        x: [bs, n_vars, in_len]
        return: [bs, n_vars, in_len], categorical channels embedded
        """
        x_out = x.clone()
        for i, var_idx in enumerate(self.cat_indices):
            x_cat = x[:, var_idx, :].long()  # [bs, in_len]
            x_emb = self.embedders[i](x_cat).squeeze(-1)  # [bs, in_len]

            x_out[:, var_idx, :] = x_emb
        return x_out

    def get_cat_indices(self):
        return self.cat_indices

    def get_vocab_sizes(self):
        return self.cat_vocab_size
    

class MixedProjector(nn.Module):
    def __init__(self, cat_indices, vocab_sizes):
        """
        cat_indices: list[int], indices of categorical variables
        vocab_sizes: list[int], number of categories per categorical variable
        """
        super(MixedProjector, self).__init__()
        self.cat_indices = cat_indices
        self.vocab_sizes = vocab_sizes
        self.cat_embed_dim = 1

        # Each categorical variable gets its own projection to vocab_size
        self.projectors = nn.ModuleList([
            nn.Linear(self.cat_embed_dim, vocab_size)
            for vocab_size in vocab_sizes
        ])

    def forward(self, y_pred):
        """
        y_pred: [bs, n_vars, out_len]  -- model output before projection
        Returns:
            A tuple of (y_num, y_cat)
            y_num: [bs, n_numerical, out_len]    (continuous output)
            y_cat: [bs, n_categorical, out_len, vocab_size] "logits vector" per categorical channel
        """
        bs, n_vars, out_len = y_pred.shape
        device = y_pred.device

        # Separate categorical and numerical indices
        all_indices = list(range(n_vars))
        num_indices = [i for i in all_indices if i not in self.cat_indices]

        # Select numerical outputs
        y_num = y_pred[:, num_indices, :]  # [bs, n_numerical, out_len]

        # Project categorical outputs
        y_cat_logits = []
        for i, var_idx in enumerate(self.cat_indices):
            y_c = y_pred[:, var_idx, :].unsqueeze(-1)             # [bs, out_len, 1]
            logit = self.projectors[i](y_c)         # [bs, out_len, vocab_size]
            y_cat_logits.append(logit)
        y_cat = torch.stack(y_cat_logits, dim=1) #[bs, n_categorical, out_len, vocab_size]

        return y_num, y_cat  # tuple of (continuous, categorical)



###I take these from One-Fits All codebase###
class DataEmbedding_wo_time(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_time, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # print("x.shape = {}".format(x.shape))
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
###I take these from One-Fits All codebase###


###I take these from TimeLLM codebase###
class ReplicationPad1d(nn.Module):
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input):
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        return output
    
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Positional embedding
        # self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x)
        return self.dropout(x), n_vars


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
###I take these from TimeLLM codebase###