import os
os.environ.setdefault("WANDB_MODE", "offline")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import random
import argparse
import numpy as np
from tqdm import tqdm 
from typing import Tuple
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from data.dataloader import build_loader_from_saved
from data.dataset import WeatherDataset, FinanceDataset
from utils import (
    normalize_timeseries, 
    denormalize_timeseries, 
    calculate_mape,
    compute_temperature_trend,
    get_transformer_backbone
)
from TS_Encoder.mome import MoMe_vis1, MoMe_vis2
from TS_Encoder.mmlinear import MMLinear
from layers import QueryPool

def get_ts_embed(
    args,
    llm,
    ts_encoder: nn.Module,
    batch,
    language_instructor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return inputs_embeds.
    Note: 
        ts: tensor [B, in_len]; selected_expert [B*C*P, num_experts]
    """
    # Initial processing and move to embedding device
    ts = batch['input_window'].to(args.device)
    normalized_ts, denorm_params = normalize_timeseries(batch['input_window'])
    batch['denorm_params'] = denorm_params
    ts = normalized_ts.to(args.device)

    if args.ts_encoder in ["MoMe", "MoMeP", "mmlinear", "mmlinearP"]:
        if ts_encoder.modulation:
            txt_embeds = llm.get_input_embeddings()(batch['text_input_ids'].to(args.device))  # [B, t, d_llm]
            txt_attn_mask = batch['text_attention_mask'].to(args.device)
            with torch.no_grad():
                outputs = llm(inputs_embeds=txt_embeds, 
                              attention_mask=txt_attn_mask, 
                              output_hidden_states=True,
                              use_cache=False
                              )
            last_hidden = outputs.hidden_states[-1]  # [B, T, d_llm]

            Ins_tk = language_instructor(last_hidden.to(torch.float32), txt_attn_mask) # ... => [B, N_i, d]
        else:
            pass
    else:
        pass
    
    # TS and Text encode
    if args.ts_encoder == 'mmlinear':
        ts = ts.unsqueeze(1) #[B, in_len] -> [B, 1, in_len]
        if ts_encoder.modulation == False:
            ts_embed = ts_encoder(ts) # [B, 1, in_len] -> [B, 1, out_len]
        else:
            ts_embed = ts_encoder(ts, Ins_tk) 
    elif args.ts_encoder in ['MoMe', 'MoMeP']:
        ts = ts.unsqueeze(1) #[B, in_len] -> [B, 1, in_len]
        if ts_encoder.modulation == False:
            selected_experts = None
            ts_embed = ts_encoder(ts) # [B, 1, in_len] -> [B, 1*P, d]
        else:
            ts_embed, selected_experts = ts_encoder(ts, Ins_tk) #[B, 1*P, d]; [B*C*P, num_experts]
    else:
        raise ValueError("Please specify a valid TS-Encoder!")

    return ts_embed, selected_experts


def load_model_components(args):
    dtype = torch.bfloat16 if args.use_bfloat16 else torch.float32

    if args.modulation:
        print(f"Finetuned Base-LLM Loaded")
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_model,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
            )
        llm.eval() 
        for param in llm.parameters():
            param.requires_grad = False
        print(f"LLM Loaded")
    else:
        llm = None

    print(f"Loading TS-Encoder: {args.ts_encoder}")
    if args.ts_encoder == 'mmlinear':
        ts_encoder = MMLinear(in_len=args.in_len, out_len=args.out_len,
                              top_k=args.topk, n_experts=args.n_experts,
                              modulation=args.modulation).to(args.device)
        
        Instructor = None
        if ts_encoder.modulation:
            Instructor = QueryPool(d_model=llm.config.hidden_size,
                                   n_queries=args.instructor_query,
                                   n_heads=1,
                                   d_proj=args.out_len,
                                   dropout=args.dropout).to(args.device)
        for param in ts_encoder.parameters():
            param.requires_grad = False

    elif args.ts_encoder == 'MoMe':
        ts_encoder = MoMe_vis1(in_len=args.in_len, patch_len=args.patch_len, hidden_dim=args.hidden_dim, 
                          top_k=args.topk, num_experts=args.n_experts, modulation=args.modulation).to(args.device)
        ts_token_num = ts_encoder.patch_num
        for param in ts_encoder.parameters():
            param.requires_grad = False
        
        Instructor = None
        if ts_encoder.modulation:
            Instructor = QueryPool(d_model=llm.config.hidden_size,
                                   n_queries=args.instructor_query,
                                   n_heads=1,
                                   d_proj=args.hidden_dim,
                                   dropout=args.dropout).to(args.device)
    elif args.ts_encoder == 'MoMeP':
        ts_encoder = MoMe_vis2(in_len=args.in_len, patch_len=args.patch_len, hidden_dim=args.hidden_dim, 
                           top_k=args.topk, num_experts=args.n_experts, modulation=args.modulation).to(args.device)
        ts_token_num = ts_encoder.patch_num
        for param in ts_encoder.parameters():
            param.requires_grad = False
        
        Instructor = None
        if ts_encoder.modulation:
            Instructor = QueryPool(d_model=llm.config.hidden_size,
                                   n_queries=args.instructor_query,
                                   n_heads=1,
                                   d_proj=args.hidden_dim,
                                   dropout=args.dropout).to(args.device)
    else:
        raise ValueError("Invalid Time Series Encoder!")


    if args.task in ['finance_trend_prediction', 'weather_trend_prediction']:
        if args.task == 'finance_trend_prediction':
            if args.finance_trend_choice == '3way':
                num_class = 3
            elif args.finance_trend_choice == '5way':
                num_class = 5
            else:
                raise ValueError("args.finance_trend_choice is either 3way or 5 way")

            if args.ts_encoder in ['mmlinear', 'mmlinearP']:
                downstream_head = None
            else:
                downstream_head = nn.Sequential(nn.Flatten(start_dim=-2),
                                                nn.Linear(ts_token_num * args.hidden_dim, num_class)).to(args.device)

        elif args.task == 'weather_trend_prediction':
            num_class = 3
            if args.ts_encoder in ['mmlinear', 'mmlinearP']:
                downstream_head = None
            else:
                downstream_head = nn.Sequential(nn.Flatten(start_dim=-2),
                                                nn.Linear(ts_token_num * args.hidden_dim, num_class)).to(args.device)

        else:
            raise KeyError(f"Please specify a valid args.task!")
        
    elif args.task in ['finance_forecast', 'weather_forecast']:
        # [B, T, d] => [B, T*d] => [B, out_len]
        if args.ts_encoder in ['mmlinear', 'mmlinearP']:
            downstream_head = None
        else:
            downstream_head = nn.Sequential(nn.Flatten(start_dim=-2),
                                            nn.Linear(ts_token_num * args.hidden_dim, args.out_len)).to(args.device)

    else:
         raise KeyError(f"Please specify a valid args.task!")


    print(f"Load Pre-trained Parameters: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)

    #print(ts_encoder)
    ts_encoder.load_state_dict(checkpoint["ts_encoder"])

    if downstream_head is not None:
        downstream_head.load_state_dict(checkpoint["downstream_head"])

    if Instructor is not None:
        Instructor.load_state_dict(checkpoint["Instructor"])
    
    if args.use_checkpoint_args:
        print("Using the configurations in the checkpoint file.")
        checkpoint_args = checkpoint["args"]
        for key, value in checkpoint_args.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    return ts_encoder, downstream_head, llm, Instructor
    


@torch.no_grad()
def evaluate_full_test_set(
    args,
    llm,
    ts_encoder,
    downstream_head,
    language_instructor,
    test_loader,
) -> tuple[float, float, dict]:
    ts_encoder.eval()
    if downstream_head is not None:
        downstream_head.eval()

    ##### End label mapping #####
 
    results = {
        "input_seq": [],
        "selected_experts": []
    }
    

    pbar = tqdm(test_loader, desc="Full Test Set Evaluation")

    if args.task in ['weather_forecast', 'finance_forecast']:

        for batch_idx, batch in enumerate(pbar):

            input_seq = batch['input_window']
            _, selected_experts = get_ts_embed(args, llm, ts_encoder, batch, language_instructor)

            results["input_seq"].append(input_seq.to(torch.float32).cpu().numpy())
            results["selected_experts"].append(selected_experts.to(torch.float32).cpu().numpy())

        return results
    
    elif args.task in ['finance_trend_prediction', "weather_trend_prediction"]:

        for batch_idx, batch in enumerate(pbar):

            input_seq = batch['input_window']
            _, selected_experts = get_ts_embed(args, llm, ts_encoder, batch, language_instructor)
            results["input_seq"].append(input_seq.to(torch.float32).cpu().numpy())
            results["selected_experts"].append(selected_experts.to(torch.float32).cpu().numpy())

        return results
    
    else:
        raise KeyError("agrs.task not supproted")


###########################################################################


# -------------------- Argparse --------------------
def parse_args():
    parser = argparse.ArgumentParser()

    # -------- Model & Data --------
    parser.add_argument("--llm_model", default="./llm/Qwen1.5-MoE-A2.7B") # we still need this for tokenizer to build/load dataset
    parser.add_argument("--dataset_path", default="./data/processed/weather/aligned_in14days_out3days")
    parser.add_argument("--output_dir", default="output/debug")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.0)

    parser.add_argument("--data_pkl_dir", type=str, default="./data/saved_datasets/weather_forecasting")
    parser.add_argument("--data_suffix", type=str, default="in14_out3")


    # -------- Train Hyperparameter --------
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)


    # -------- TS-Encoder & MoE --------
    parser.add_argument("--instructor_query", type=int, default=2)
    parser.add_argument("--modulation", action="store_true", help="MoMe")
    parser.add_argument("--patch_len", type=int, default=4)
    parser.add_argument("--ts_encoder", type=str, default="time-moe")
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_vars", type=int, default=1)
    parser.add_argument("--seq_len_channel", type=int, default=128) #This might not be used anymore
    parser.add_argument("--hidden_dim", type=int, default=32) #note 384 is the embedding dimensionf for time-moe
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_experts", type=int, default=3)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--moe_mode", default="No")  # multi_expert / No
    parser.add_argument("--n_heads", type=int, default=4)

    parser.add_argument("--in_len", type=int, default=336)
    parser.add_argument("--out_len", type=int, default=72)
    parser.add_argument("--max_text_length", type=int, default=2048)

    # -------- Numerical Safety & Dtypes --------
    parser.add_argument("--pre_align_clip", type=float, default=1e5,
                        help="If > 0, clamp TS embedding to [-clip, clip] BEFORE AlignLayer (both train & test).")
    parser.add_argument("--use_bfloat16", action="store_true",
                        help="Load LLM in bfloat16 (if supported) instead of float16.")

    # -------- Others --------
    parser.add_argument("--project_name", default="Fusion")
    parser.add_argument("--scale", type=bool, default=False)
    parser.add_argument("--upsampling_pad_direction", type=str, default="forward")
    parser.add_argument("--upsampling_type", type=str, default="pad")
    parser.add_argument("--downsampling_type", type=str, default="average")
    parser.add_argument("--pad_mode", type=str, default="constant")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task", type=str, default="weather_forecast")
    parser.add_argument("--finance_trend_choice", type=str, default="3way") #'3way' or '5way'
    parser.add_argument("--weather_trend_choice", type=str, default="future") #'past' or 'future'

    # -------- Eval --------
    parser.add_argument("--eval_mode", type=str, default="full_test", help="full_test or random_sample")
    parser.add_argument("--save_full_predictions", action="store_true", help="whether or not save predicted series")
    parser.add_argument("--use_checkpoint_args", action="store_true", help="whether or not use identical configurations as checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="./output/8-28-base/ts_encoder_epoch3.pt")
    parser.add_argument("--sample_seed", type=int, default=7)

    return parser.parse_args()



def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.sample_seed)

    llm_path = args.llm_model

    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id  


    if args.task in ["weather_forecast", "weather_trend_prediction"]:
        dataset = WeatherDataset(
            data_dir=args.dataset_path,
            tokenizer=tokenizer,
            input_seq_len=args.in_len,
            output_seq_len=args.out_len,
            max_text_length=args.max_text_length
        )
    elif args.task == "finance_forecast":
        dataset = FinanceDataset(
            data_dir=args.dataset_path,
            tokenizer=tokenizer,
            input_len=args.in_len,
            output_len=args.out_len, 
            max_text_length=2048
            )
    elif args.task == "finance_trend_prediction":
        dataset = FinanceDataset(
            data_dir=args.dataset_path,
            tokenizer=tokenizer,
            input_len=args.in_len,
            output_len=args.out_len, 
            max_text_length=2048
            )
    else:
        raise NotImplementedError("Unknow Evaluation Task")

    # build the models
    ts_encoder, downstream_head, llm, Instructor = load_model_components(args)
    
    _, _, test_loader = build_loader_from_saved(dataset=dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                train_ratio=args.train_ratio,val_ratio=args.val_ratio, seed=args.seed, 
                                                save_dir=args.data_pkl_dir, suffix=args.data_suffix, train_shuffle=False)

    results = evaluate_full_test_set(args, llm, ts_encoder, downstream_head, Instructor, test_loader)

    pred_path = os.path.join(args.output_dir, "test_results.npz")
    np.savez(
        pred_path,
        input_seq=results["input_seq"],
        selected_experts=results["selected_experts"],
    )
    print(f"Complete results are saved at: {pred_path}")
            

if __name__ == "__main__":
    #CUDA_VISIBLE_DEVICES=0 python evaluate_vis.py --task finance_trend_prediction --in_len 134 --finance_trend_choice 3way --data_pkl_dir ./data/saved_datasets/finance_trend_prediction --dataset_path ./data/processed/finance/pair_in_30days_1hours_out_7days_1hours --data_suffix in30_out7 --batch_size 1 --checkpoint_path output/FT-L-MoMEP/ts_encoder_epoch9.pt  --output_dir ./output/Expert_Selection/FT-L-MoMEP --hidden_dim 32 --patch_len 8 --n_experts 4 --topk 2 --ts_encoder MoMeP --use_bfloat16 --modulation --instructor_query 3 --eval_mode random_sample --sample_seed 77
    #CUDA_VISIBLE_DEVICES=0 python evaluate_vis.py --task finance_trend_prediction --in_len 134 --finance_trend_choice 3way --data_pkl_dir ./data/saved_datasets/finance_trend_prediction --dataset_path ./data/processed/finance/pair_in_30days_1hours_out_7days_1hours --data_suffix in30_out7 --batch_size 1 --checkpoint_path output/FT-L-MoME/ts_encoder_epoch9.pt  --output_dir ./output/Expert_Selection/FT-L-MoME --hidden_dim 32 --patch_len 8 --n_experts 4 --topk 2 --ts_encoder MoMe --use_bfloat16 --modulation --instructor_query 3
    torch.multiprocessing.set_start_method('spawn')
    main()

