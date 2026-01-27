import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import FullAttention, AttentionLayer
from models.emb_layers import DataEmbedding_inverted
from models.layers import RevIN, Cat_Embed, MixedProjector
import numpy as np
import math
#Adapted from paper https://arxiv.org/abs/2310.06625


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.n_cluster = args.n_cluster
        self.n_vars = args.batch_size if args.data in ["M4", "stock"] else args.data_dim
        self.individual = args.individual
        self.seq_len = args.in_len
        self.d_model = args.d_model
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
                
        self.encoder = iTransformer_backbone(args)

        #This is for Telecom datasets, where categorical channels are involved
        self.categorical = args.categorical
        if args.categorical:
            self.cat_embed = Cat_Embed(cat_indices=[16,17], cat_vocab_size=[3,3])
            self.mixed_proj = MixedProjector(cat_indices=[16,17], vocab_sizes=[3,3])
            # Add two learnable log-variance parameters for dynamic loss scaling
            self.log_sigma_num = nn.Parameter(torch.zeros(()))
            self.log_sigma_cat = nn.Parameter(torch.zeros(()))
            #self.rev_in = RevIN(args.data_dim-2, affine=True, subtract_last=False)
        else:
            pass
            # Note that we already have a RevIn in the backbone!!!!!
            #self.rev_in = RevIN(args.data_dim, affine=True, subtract_last=False)


    def forward(self, x_seq, if_update=False):       # [bs, seq_len, n_vars]

        x_seq = x_seq.permute(0,2,1) # [bs, n_vars, seq_len]

        if self.categorical:
            x = self.cat_embed(x_seq)  # x: [bs, n_vars, in_len], with categorical channels embedded
            x = x.transpose(1,2) # now x is [bs, in_seq, n_vars]
            x_num = x[:, :, :16]      # numerical features
            x_cat = x[:, :, 16:]      # categorical features
            #print("SECONDE",x_num.shape, x_num.device)
            #x_num = self.rev_in(x_num, mode="norm")
            x = torch.cat([x_num, x_cat], dim=2)  #[bs, in_seq, n_vars]
            x = x.transpose(1,2) #[bs, in_seq, n_vars] -> #[bs, n_vars, in_seq]
        else:
            x = x_seq

        x, gate_score_lis = self.encoder(x) #out shape:[bs, target_window, nvars]

        if self.categorical:
            x_num_ = x[:, :, :16]      # numerical features
            x_cat_ = x[:, :, 16:]      # categorical features
            #x_num_ = self.rev_in(x_num_, mode="denorm")
            x = torch.cat([x_num_, x_cat_], dim=2)  #[bs, in_seq, n_vars]
            out = self.mixed_proj(x.transpose(1,2)) 
            # out = (out_num, out_cat) a tupple
            # out_num: [bs, n_numerical, out_len], out_cat: [bs, n_categorical, out_len, vocab_size]
        else:
            out = x

        return out, gate_score_lis, None #[:, :self.out_len, :]   # [bs, out_len, n_vars]
    

class iTransformer_backbone(nn.Module):
    def __init__(self, args):
        super(iTransformer_backbone, self).__init__()
        self.seq_len = args.in_len
        self.pred_len = args.out_len
        self.output_attention = args.output_attention
        self.use_norm = args.ravin_norm
        self.enc_embedding = DataEmbedding_inverted(args.in_len, args.d_model, args.dropout)
        self.class_strategy = args.class_strategy
        self.revin_layer = RevIN(args.data_dim, affine=True, subtract_last=False)
        self.top_k = args.topk
        self.moe_mode = args.moe_mode
        self.n_experts = args.n_experts
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, args.factor, attention_dropout=args.dropout,
                                      output_attention=args.output_attention), args.d_model, args.n_heads),
                    args.d_model,
                    args.n_heads,
                    dropout=args.dropout,
                    activation=args.activation,
                    num_experts=self.n_experts,
                    topk=args.topk,
                    moe_mode=self.moe_mode
                ) for l in range(args.n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model), moe_mode =args.moe_mode
        )

        if self.moe_mode == 'multi_expert':
            self.Gating = nn.Linear(args.d_model, self.n_experts, bias=False)
            self.experts = nn.ModuleList([nn.Linear(args.d_model, args.out_len, bias=True) for _ in range(self.n_experts)])
        else:
            self.projector = nn.Linear(args.d_model, args.out_len, bias=True)


    def forecast(self, x_enc):
        """
        #x_enc: [bs, seq_len, n_vars]
        """
        gate_score_lis = None

        if self.use_norm:
            x_enc = self.revin_layer(x_enc, 'norm')

        B, _, N = x_enc.shape

        # Embedding: [bs, seq_len, n_vars] -> [bs, n_vars, d_model]
        enc_out = self.enc_embedding(x_enc, None)

        # Multivariate Attention: [bs, n_vars, d_model] -> [bs, n_vars, d_model]
        enc_out, attns, gate_score_lis = self.encoder(enc_out, attn_mask=None)
        
        # Projection head: [bs, n_vars, d_model] -> [bs, out_len, n_vars]
        if self.moe_mode == "multi_expert":
            gate_logits = self.Gating(enc_out) #[B, N, E]
            gate_logits = F.softmax(gate_logits, dim=-1)
            weights, selected_experts = torch.topk(gate_logits, self.top_k)  # [bs, n_vars, topk]
            output = torch.zeros(B, N, self.pred_len, device=enc_out.device, dtype=enc_out.dtype)
            for i, expert in enumerate(self.experts):
                batch_idx, token_idx, kth = torch.where(selected_experts == i)
                #print(expert(x[batch_idx, token_idx]).shape) #[N_i, d_model]
                output[batch_idx, token_idx] += (
                    weights[batch_idx, token_idx, kth][:, None] * expert(enc_out[batch_idx, token_idx])
                    )
            enc_out = output.permute(0, 2, 1)
        else:
            enc_out = self.projector(enc_out).permute(0, 2, 1)

        if self.use_norm:
            enc_out = self.revin_layer(enc_out, 'denorm')

        return enc_out, attns, gate_score_lis

    def forward(self, x_enc):
        '''
        input:
            x_enc: [bs, nvars, seq_len]
        return: 
            x_enc: [bs, nvars, target_window]
        '''       
        x_enc = x_enc.permute(0,2,1) # [bs, seq_len, nvars]
        out, attns, gate_score_lis = self.forecast(x_enc) # out shape: [bs, out_len, n_vars]

        return out, gate_score_lis # [B,S,N] , [B, n_cluster, d_model]





class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, moe_mode='single_expert'):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.moe_mode = moe_mode

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []; gate_score_lis = []
        if self.conv_layers is not None:
            # Since it's always None we didn't modify this part
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn, gate_scores = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn, gate_scores = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)
                if gate_scores is not None:
                    #print("HERE")
                    gate_score_lis.append(gate_scores.mean(0))

        if self.norm is not None:
            x = self.norm(x)

        return x, attns, gate_score_lis
    

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", num_experts=5, topk=2, moe_mode='single_expert'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        activation_layer = nn.ReLU() if activation == "relu" else nn.GELU()

        self.moe_mode = moe_mode
        self.top_k = topk
        self.n_experts = num_experts
        #for moe:
        if self.moe_mode == "multi_expert":
            self.Gating = nn.Linear(d_model, self.n_experts, bias=False)
            self.experts = nn.ModuleList([
                    nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    activation_layer,
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout)
                ) for _ in range(num_experts)
            ])
        else:
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        #remark: Conv1d(kernel_size=1) is equivalent to Linear1(x) → ReLU → Linear2(x)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        x: [bs, n_vars, d_model]
        cls_emd: [n_clusters, d_model]
        prob_matrix: [n_vars, n_clusters]
        """
        bs = x.shape[0]
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        if self.moe_mode == "multi_expert":
            gate_logits = self.Gating(y) #[B, N, E]
            gate_logits = F.softmax(gate_logits, dim=-1)
            weights, selected_experts = torch.topk(gate_logits, self.top_k)  # [bs, n_vars, topk]
            output = torch.zeros_like(y) #[bs, n_vars, d_model]
            for i, expert in enumerate(self.experts):
                batch_idx, token_idx, kth = torch.where(selected_experts == i)
                #print(expert(x[batch_idx, token_idx]).shape) #[N_i, d_model]
                output[batch_idx, token_idx] += (
                    weights[batch_idx, token_idx, kth][:, None] * expert(y[batch_idx, token_idx])
                )
            y = output

            return self.norm2(x + y), attn, gate_logits
        else:
            gate_logits = None
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))

            return self.norm2(x + y), attn, gate_logits
    


"""
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.n_cluster = args.n_cluster
        self.n_vars = args.batch_size if args.data in ["M4", "stock"] else args.data_dim
        self.individual = args.individual
        self.seq_len = args.in_len
        self.d_model = args.d_model
        self.if_moe = args.if_moe
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        if self.individual == "c":
            self.Cluster_assigner = Cluster_assigner(self.n_vars, self.n_cluster, self.seq_len, self.d_model, device=self.device)
            #self.Cluster_assigner = Cluster_assigner_fixed(self.n_vars, self.n_cluster, self.seq_len, self.d_model, device=self.device)
            self.cluster_emb = self.Cluster_assigner.cluster_emb
        else:
            if not self.if_moe:
                self.cluster_emb = torch.empty(self.n_cluster, self.d_model).to(self.device)
            else:
                self.cluster_emb = nn.Parameter(torch.empty(self.n_cluster, self.d_model).to(self.device), requires_grad=True)
                nn.init.kaiming_uniform_(self.cluster_emb, a=math.sqrt(5))
                
        self.encoder = iTransformer_backbone(args)
        self.cluster_prob = None

        #This is for Telecom datasets, where categorical channels are involved
        self.categorical = args.categorical
        if args.categorical:
            self.cat_embed = Cat_Embed(cat_indices=[16,17], cat_vocab_size=[3,3])
            self.mixed_proj = MixedProjector(cat_indices=[16,17], vocab_sizes=[3,3])
            # Add two learnable log-variance parameters for dynamic loss scaling
            self.log_sigma_num = nn.Parameter(torch.zeros(()))
            self.log_sigma_cat = nn.Parameter(torch.zeros(()))
            self.rev_in = RevIN(args.data_dim-2, affine=True, subtract_last=False)
        else:
            pass
            #self.rev_in = RevIN(args.data_dim, affine=True, subtract_last=False)


    def forward(self, x_seq, if_update=False):       # [bs, seq_len, n_vars]

        if self.individual == "c":
            self.cluster_prob, cluster_emb_1 = self.Cluster_assigner(x_seq, self.cluster_emb)      #[n_vars, n_cluster] for both tensors

        x_seq = x_seq.permute(0,2,1) # [bs, n_vars, seq_len]

        if self.categorical:
            x = self.cat_embed(x_seq)  # x: [bs, n_vars, in_len], with categorical channels embedded
            x = x.transpose(1,2) # now x is [bs, in_seq, n_vars]
            x_num = x[:, :, :16]      # numerical features
            x_cat = x[:, :, 16:]      # categorical features
            #print("SECONDE",x_num.shape, x_num.device)
            x_num = self.rev_in(x_num, mode="norm")
            x = torch.cat([x_num, x_cat], dim=2)  #[bs, in_seq, n_vars]
            x = x.transpose(1,2) #[bs, in_seq, n_vars] -> #[bs, n_vars, in_seq]
        else:
            x = x_seq

        #out, cls_emb, gate_score_lis = self.encoder(x_seq, self.cluster_emb, self.cluster_prob) #out shape:[bs, target_window, nvars]
        x, cls_emb, gate_score_lis = self.encoder(x, self.cluster_emb, self.cluster_prob) #out shape:[bs, target_window, nvars]

        if self.categorical:
            x_num_ = x[:, :, :16]      # numerical features
            x_cat_ = x[:, :, 16:]      # categorical features
            x_num_ = self.rev_in(x_num_, mode="denorm")
            x = torch.cat([x_num_, x_cat_], dim=2)  #[bs, in_seq, n_vars]
            out = self.mixed_proj(x.transpose(1,2)) 
            # out = (out_num, out_cat) a tupple
            # out_num: [bs, n_numerical, out_len], out_cat: [bs, n_categorical, out_len, vocab_size]
        else:
            out = x

        if if_update and self.individual == "c":
            self.cluster_emb = nn.Parameter(cluster_emb_1, requires_grad=True) 

        return out, gate_score_lis, self.cluster_prob#[:, :self.out_len, :]   # [bs, out_len, n_vars]
    

class iTransformer_backbone(nn.Module):
    def __init__(self, args):
        super(iTransformer_backbone, self).__init__()
        self.seq_len = args.in_len
        self.pred_len = args.out_len
        self.output_attention = args.output_attention
        self.use_norm = args.ravin_norm
        self.enc_embedding = DataEmbedding_inverted(args.in_len, args.d_model, args.dropout)
        self.class_strategy = args.class_strategy
        self.revin_layer = RevIN(args.data_dim, affine=True, subtract_last=False)
        self.individual = args.individual
        self.if_moe = args.if_moe
        self.topk = args.topk
        self.moe_head = args.moe_head
        self.num_experts = args.n_cluster
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, args.factor, attention_dropout=args.dropout,
                                      output_attention=args.output_attention), args.d_model, args.n_heads),
                    args.d_model,
                    args.n_heads,
                    dropout=args.dropout,
                    activation=args.activation,
                    num_experts=self.num_experts,
                    topk=args.topk,
                    if_moe=args.if_moe
                ) for l in range(args.n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model), if_moe=args.if_moe
        )
        if args.individual == "i":
            #self.projector = nn.Linear(args.d_model, args.out_len, bias=True)
            if not self.if_moe:
                self.projector = nn.Linear(args.d_model, args.out_len, bias=True)
            else:
                if self.moe_head == 'ind':
                    #TESTING, comment out if no good
                    self.projector = Flatten_Head(args.data_dim, args.d_model, args.out_len)
                else:
                    self.projector = nn.Linear(args.d_model, args.out_len, bias=True)
        else:
            self.cluster_projector = Cluster_wise_linear(args.n_cluster,args.data_dim,args.d_model,args.out_len,self.device)

    def forecast(self, x_enc, cluster_prob, cls_emd):

        #x_enc: [bs, seq_len, n_vars]
        #cluster_prob: [nvars, n_cluster]

        if self.use_norm:
            x_enc = self.revin_layer(x_enc, 'norm')

        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, None)   # Embedding: [bs, seq_len, n_vars] -> [bs, n_vars, d_model]
        enc_out, attns, gate_score_lis = self.encoder(enc_out, cls_emd=cls_emd, prob_matrix=cluster_prob, attn_mask=None, topk=self.topk) # Multivariate Attention: [bs, n_vars, d_model] -> [bs, n_vars, d_model]
        #enc_out, attns = self.encoder(enc_out, self.,attn_mask=None)
        if self.individual == "i":
            enc_out = self.projector(enc_out).permute(0, 2, 1) # Projection head: [bs, n_vars, d_model] -> [bs, out_len, n_vars]
        else:
            enc_out = self.cluster_projector(enc_out, cluster_prob).permute(0, 2, 1)

        if self.use_norm:
            enc_out = self.revin_layer(enc_out, 'denorm')

        return enc_out, attns, gate_score_lis

    def forward(self, x_enc, cls_emb, prob):
        '''
        input:
            x_enc: [bs, nvars, seq_len]
            cls_emb: [n_cluster, d_model]
        return: 
            x_enc: [bs, nvars, target_window]
            cls_emb: [n_cluster, d_model]
        '''       
        x_enc = x_enc.permute(0,2,1) # [bs, seq_len, nvars]
        out, attns, gate_score_lis = self.forecast(x_enc, prob, cls_emb) # out shape: [bs, out_len, n_vars]
        #out, attns = self.forecast(x_enc, prob) # out shape: [bs, out_len, n_vars]
        # The following two sentence is not necessary, no patch in iTransformer
        ##cls_emb = cls_emb.expand(out.shape[0], cls_emb.shape[0], cls_emb.shape[1])       # (bs*patch_num, n_cluster, d_model)
        ##cls_emb = cls_emb.mean(0)
        return out, cls_emb, gate_score_lis # [B,S,N] , [n_cluster, d_model]



class Cluster_wise_linear(nn.Module):
    def __init__(self, n_cluster, n_vars, in_dim, out_dim, device):
        super().__init__()
        self.n_cluster = n_cluster
        self.n_vars = n_vars
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linears = nn.ModuleList()
        for i in range(n_cluster):
            self.linears.append(nn.Linear(in_dim, out_dim))
        
    def forward(self, x, prob):
        # x: [bs, n_vars, in_dim]
        # prob: [n_vars, n_cluster]
        # return: [bs, n_vars, out_dim]
        bsz = x.shape[0]
        output = []
        for layer in self.linears:
            output.append(layer(x))
        output = torch.stack(output, dim=-1).to(x.device)  #[bsz, n_vars,  out_dim, n_cluster]
        prob = prob.unsqueeze(-1)  #[n_vars, n_cluster, 1]
        output = torch.matmul(output, prob).reshape(bsz, -1, self.out_dim)   #[bsz, n_vars, out_dim]
        return output
    



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

"""



"""
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.seq_len = args.in_len
        self.pred_len = args.out_len
        self.output_attention = args.output_attention
        self.use_norm = args.pre_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(args.in_len, args.d_model, args.dropout)
        self.class_strategy = args.class_strategy
        #self.revin_layer = RevIN(args.data_dim, affine=True, subtract_last=False)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, args.factor, attention_dropout=args.dropout,
                                      output_attention=args.output_attention), args.d_model, args.n_heads),
                    args.d_model,
                    args.n_heads,
                    dropout=args.dropout,
                    activation=args.activation
                ) for l in range(args.n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )
        self.projector = nn.Linear(args.d_model, args.out_len, bias=True)

    #def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    def forecast(self, x_enc):

        if self.use_norm:
            #x_enc = self.revin_layer(x_enc, 'norm')
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, None) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        enc_out = self.projector(enc_out).permute(0, 2, 1)
        #enc_out = enc_out.permute(0, 2, 1)
        
        if self.use_norm:
            #enc_out = self.revin_layer(enc_out, 'denorm')
            # De-Normalization from Non-stationary Transformer
            enc_out = enc_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            enc_out = enc_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return enc_out, attns


    #x_enc:(B,L,N)
    def forward(self, x_enc, if_update):
        #dec_out, attns = self.forecast(x_enc, x_mark_enc)
        dec_out, attns = self.forecast(x_enc)

        #print("here ",type(dec_out),dec_out.shape)
        
        return dec_out  # [B,S,N]
        #if self.output_attention:
            #return dec_out, attns
        #else:
            #return dec_out  # [B,S,N]
        
    #x_enc:(B,L,N)
    #def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        #dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        #if self.output_attention:
            #return dec_out[:, -self.pred_len:, :], attns
        #else:
            #return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class moe_projector(nn.Module):
    def __init__(self, n_cluster, in_dim, out_dim, topk):
        super().__init__()
        self.Gating = GatingNetwork_v2(in_dim, n_cluster, topk)
        self.n_cluster = n_cluster
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(n_cluster)])
        
    def forward(self, x, prob):
        # x: [bs, n_vars, in_dim]
        # prob: [n_vars, n_cluster]
        # return: [bs, n_vars, out_dim]
        bs = x.shape[0]
        gate_scores = self.Gating(prob) # [n_vars, n_clusters]
        gate_scores = gate_scores.unsqueeze(0).repeat(bs, 1, 1)  # [n_vars, n_clusters] -> [bs, n_vars, n_clusters]
        output = torch.stack([linear(x) for linear in self.linears], dim=2)  # [bs, n_vars, n_clusters, d_model]
        output = torch.einsum('bnk,bnkd->bnd', gate_scores, output)  # [bs, n_vars, d_model]

        return output

"""
