import argparse
import os
import torch
import numpy as np
from exp.exp_ccm import Exp_CCM
from exp.exp_neoe import Exp_neoe
from exp.exp_moe import Exp_moe
from utils.tools import string_split
import math
import wandb
from torchinfo import summary
parser = argparse.ArgumentParser(description='Neoe')

#########################################################
os.environ["WANDB_MODE"] = "offline" 
#########################################################

# model: PatchTST / DLinear / TSMixer / TimesNet

seed=25
np.random.seed(seed)
torch.manual_seed(seed)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument('--data', type=str, default='ECL', help='data')
parser.add_argument('--test_data', type=str, default='ETTm2', help='data')
parser.add_argument('--zero_shot_test', type=str2bool, default=False)
parser.add_argument('--model', type=str, default='PatchTST', help='model')
parser.add_argument('--root_path', type=str, default='./datasets', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')  
parser.add_argument('--data_split', type=str, default='0.7,0.1,0.2',help='train/val/test split, can be ratio or number')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location to store model checkpoints')
parser.add_argument('--data_dim', type=int, default=7, help='Number of dimensions of the MTS data (D)')

parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')

parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer initial learning rate')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--individual', type=str, default="i", help="i: individual; c: cluster, else: all-in dimension")
parser.add_argument('--beta', type=float, default=0.3, help="loss weight for similarity loss")
parser.add_argument('--in_len', type=int, default=96, help='input MTS length (T)')
parser.add_argument('--out_len', type=int, default=48, help='output MTS length (\tau)')


parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
parser.add_argument('--n_layers', type=int, default=4, help='num of encoder layers (N)')
parser.add_argument('--d_ff', type=int, default=64, help='dimension of MLP in transformer')
parser.add_argument('--n_cluster', type=int, default=3, help="number of clusters")


## model parameters
parser.add_argument('--d_model', type=int, default=512, help='dimension of hidden states (d_model)')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--attn_dropout', type=float, default=0.3, help='attention dropout')
parser.add_argument('--pre_norm', type=str2bool, default=False, help='pre normalization')
parser.add_argument('--stride', type=int, default=8, help="stride")
parser.add_argument('--pretrain_head', type=str2bool, default=False, help='pretrain head')
parser.add_argument('--patch_len', type=int, default=16, help='patch length (L_seg)')
parser.add_argument('--max_seq_len', type=int, default=1024, help="maximum number of sequence_length")
parser.add_argument('--padding_patch', type=str, default='end', help='None: None; end: padding on the end')
#newly added model related
parser.add_argument('--output_attention', type=str, default='False', help='whether or not return attention matrix')
parser.add_argument('--activation', type=str, default='gelu', help='activation functions')
parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--if_moe', action='store_true', help='Enable MoE architecture')
#parser.add_argument('--cv_alpha', type=float, default=0.5, help='a part of cluster distance loss')
#parser.add_argument('--ravin_norm', action='store_true', help='Enable normalize and denormalize of original series')
parser.add_argument('--ravin_norm', type=str2bool, default=True)
parser.add_argument('--moe_head', type=str, default='comb', help='independent or combined head')# else 'ind'
parser.add_argument('--categorical', action='store_true', help='whether or not has cetogorical input channels')

parser.add_argument('--sparse_expert', type=str2bool, default=True, help='Whether use bernullie sampling when gating')
parser.add_argument('--strategy', type=str, default='l2', help='l2 distance similarity or cosine similarity when gating')
#parser.add_argument('--shared_routing', type=str2bool, default=False, help='one gating for all layer or the otherway round')
#parser.add_argument('--moe_mode', type=str, choices=["shared", "per_layer", "single_expert"],
                    #default="shared", help="MoE routing mode: shared, per_layer, or single_expert")

parser.add_argument('--moe_mode', type=str, choices=["multi_expert", "single_expert"],
                    default="single_expert", help="MoE routing mode: multi_expert or single_expert")
parser.add_argument('--expert_type', type=str, default='mlp', help='mlp/itransformer')
parser.add_argument('--use_lb_loss', type=str2bool, default=False)
parser.add_argument('--n_experts', type=int, default=2, help='number of experts')
parser.add_argument('--topk', type=int, default=1, help='num of top experts')
parser.add_argument('--cv', type=float, default=0.3, help='loss weight for moe auxiliary loss')
parser.add_argument('--capacity_factor', type=float, default=0.8)



parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--save_pred', action='store_true', help='whether to save the predicted future MTS', default=True)
parser.add_argument('--cuda', type=int, default=0, help='GPU device.')

args = parser.parse_args()


if args.zero_shot_test is False:
    args.test_data = args.data


data_parser = {
    'ETTh1':{'data':'ETTh1.csv', 'data_dim':7, 'split':[12*30*24, 4*30*24, 4*30*24]},
    'ETTm1':{'data':'ETTm1.csv', 'data_dim':7, 'split':[4*12*30*24, 4*4*30*24, 4*4*30*24]},
    'ETTh2':{'data':'ETTh2.csv', 'data_dim':7, 'split':[12*30*24, 4*30*24, 4*30*24]},
    # 'ETTh2':{'data':'ETTh2.csv', 'data_dim':7, 'split':[10021, 3389, 3389]},
    'ETTm2':{'data':'ETTm2.csv', 'data_dim':7, 'split':[4*12*30*24, 4*4*30*24, 4*4*30*24]},
    # 'WTH':{'data':'WTH.csv', 'data_dim':12, 'split':[28*30*24, 10*30*24, 10*30*24]},
    # 'ECL':{'data':'ECL.csv', 'data_dim':321, 'split':[15*30*24, 3*30*24, 4*30*24]},
    'WTH':{'data':'WTH.csv', 'data_dim':12, 'split':[0.7, 0.1, 0.2]},
    'ECL':{'data':'ECL.csv', 'data_dim':321, 'split':[0.7, 0.1, 0.2]},
    'ILI':{'data':'ILI.csv', 'data_dim':7, 'split':[0.7, 0.1, 0.2]},
    'TRF':{'data':'TRF.csv', 'data_dim':862, 'split':[0.7, 0.1, 0.2]},
    'EXR':{'data':'EXR.csv', 'data_dim':8, 'split':[0.7, 0.1, 0.2]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.data_dim = data_info['data_dim']
    args.data_split = data_info['split']
else:
    args.data_split = string_split(args.data_split)
    
args.freq = "t" if args.data in ["ETTm1", "ETTm2"] else "h"
args.test_data_path = data_parser[args.test_data]['data']
args.test_data_split = data_parser[args.data]['split']


# I think we can look at data and choose num of clusters, comment out for now
#args.n_cluster = math.ceil(args.data_dim * args.cluster_ratio)

wandb.login(key="")
wandb.init(project="time_series_forecasting", )  #  mode="disabled"
wandb.config.update(args)
#print('Args in experiment:')
#print(args)

#print("Train data:", args.data_path)
#print("Test data:", args.test_data_path)


#Exp = Exp_neoe
Exp = Exp_moe

for ii in range(args.itr):
    # setting record of experiments
    #setting = '{}_{}_il{}_ol{}_pl{}_ratio{}_dm{}_nh{}_el{}_itr{}'.format(args.model, args.data, 
                #args.in_len, args.out_len, args.patch_len, args.cluster_ratio,
                #args.d_model, args.n_heads, args.n_layers, ii)
    if args.zero_shot_test == False:
        setting = '{}_{}_{}_il{}_ol{}_pl{}_cluster{}_dm{}_nh{}_el{}_itr{}'.format(args.model, args.expert_type, args.data, 
                    args.in_len, args.out_len, args.patch_len, args.n_cluster,
                    args.d_model, args.n_heads, args.n_layers, ii)
    else:
        setting = 'ZeroShot_{}_{}_{}_il{}_ol{}_pl{}_cluster{}_dm{}_nh{}_el{}_itr{}'.format(args.model, args.expert_type, args.data, 
                    args.in_len, args.out_len, args.patch_len, args.n_cluster,
                    args.d_model, args.n_heads, args.n_layers, ii)

    exp = Exp(args) # set experiments
    print('>>>>>>>start training on: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    print('>>>>>>>testing on {}: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.test_data, setting))
    exp.test(setting, args.save_pred)
    
    
