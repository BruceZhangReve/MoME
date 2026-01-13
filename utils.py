import os
import torch
import numpy as np
from torch.nn import Linear
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def normalize_timeseries(x: torch.Tensor, dim=-1, eps: float = 1e-3):
    """
    x: [B, C] [B, T] or [B, ... , T]
    Return: z, (mean, std) for denormalize
    """
    x32 = x.to(torch.float32)
    mean = x32.mean(dim=dim, keepdim=True)
    std = x32.std(dim=dim, keepdim=True)
    std = std.clamp_min(eps)
    z = (x32 - mean) / std

    z = z.clamp_(-10.0, 10.0)
    return z.to(x.dtype), (mean, std)

def denormalize_timeseries(z: torch.Tensor, params):
    mean, std = params
    z32 = z.to(torch.float32)
    y = z32 * std + mean
    return y.to(z.dtype)




# MAPE: (|y - \hat{y}| / |y|)
def calculate_mape(pred, target):
    # avoid devision by 0
    epsilon = 1e-8
    return torch.mean(torch.abs((pred - target) / (target.abs() + epsilon)))*100


def plot_series(filename, input_ts, output_ts, predicted_ts):#, save_folder):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(input_ts)), input_ts, label="Input Time Series", marker='o')
    plt.plot(range(len(input_ts), len(input_ts) + len(output_ts)), output_ts, label="Ground Truth", marker='o')
    plt.plot(range(len(input_ts), len(input_ts) + len(predicted_ts)), predicted_ts, label="Predicted", linestyle='dashed')
    plt.legend()
    plt.title(f"Prediction for {filename}")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.grid()
    #plt.savefig(os.path.join(save_folder, filename.replace('.json', '.png')))
    plt.close()

def calculate_cross_epoch_similarity(param_epoch0, param_epoch1):
    """calculate the weight matrices similarities"""
    p0_flat = param_epoch0.flatten()
    p1_flat = param_epoch1.flatten()
    
    pearson_corr, pearson_p = pearsonr(p0_flat, p1_flat)

    cos_sim = cosine_similarity(p0_flat.reshape(1, -1), p1_flat.reshape(1, -1))[0][0]

    mse = np.mean((p0_flat - p1_flat) ** 2)

    l2_dist = np.linalg.norm(p0_flat - p1_flat)
    l2_norm = l2_dist / (np.linalg.norm(p0_flat) + np.linalg.norm(p1_flat) + 1e-8)

    return {
        "Pearson Correlation": (round(pearson_corr, 4), f"p-Value: {pearson_p:.2e}"),
        "Cosine Similarity": round(cos_sim, 4),
        "MSE": round(mse, 6),
        "Normalized L2 Distance": round(l2_norm, 6)
    }



def compare_tensor_difference(tensor1, tensor2):
    max_val1 = tensor1.max().item()
    min_val1 = tensor1.min().item()
    mean_val1 = tensor1.mean().item()
    std_dev1 = tensor1.std().item()
 
    max_val2 = tensor2.max().item()
    min_val2 = tensor2.min().item()
    mean_val2 = tensor2.mean().item()
    std_dev2 = tensor2.std().item()

    print("Tensor 1 - Max:", max_val1, "Min:", min_val1, "Mean:", mean_val1, "Std Dev:", std_dev1)
    print("Tensor 2 - Max:", max_val2, "Min:", min_val2, "Mean:", mean_val2, "Std Dev:", std_dev2)

    return None

def get_base_model(m):
    return getattr(m, "model", None) or getattr(m, "transformer", None) or m

def get_transformer_backbone(m):
    """
    Robustly unwrap PEFT/DS/FSDP wrappers to get the transformer backbone
    that returns BaseModelOutputWithPast(last_hidden_state=...).
    Works for PeftModelForCausalLM(LoraModel(Qwen2MoeForCausalLM(...))).
    """
    if hasattr(m, "get_base_model"):
        try:
            m = m.get_base_model()  # e.g., Qwen2MoeForCausalLM
        except Exception:
            pass

    # 2) some models has LoRA PeftModelForCausalLM(base_model=LoraModel(model=...))
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model  # -> Qwen2MoeForCausalLM

    # 3) from *ForCausalLM take transformer backbone
    backbone = getattr(m, "model", None) or getattr(m, "transformer", None)
    if backbone is None:
        backbone = m

    return backbone


### Below are some functions for weather trend related stuff ###
def compute_temperature_trend(past_temperatures, next_temperatures=None):
    """
    params:
        past_temperatures: tensor of shape [B, L], where B is batch size and L is input window's length (has to be a multiple of 24)
        next_temperatures: [B, M] tensor, where M is the future temprature (also has to be a multiple of 24)
    return:
       A string, which can either be "increasing", "decreasing" or "stable"
    """
    if not isinstance(past_temperatures, torch.Tensor):
        past_temperatures = torch.tensor(past_temperatures, dtype=torch.float32)
    
    B, L = past_temperatures.shape
    past_days = L // 24 
    past_temperatures = past_temperatures.reshape(B, past_days, 24)
    
    if next_temperatures is None:
        # for past trend
        daily_mean = torch.mean(past_temperatures, dim=2)  # [B, past_days]
        
        x = torch.arange(past_days, device=past_temperatures.device).unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1).float()
        
        x = torch.cat([x, torch.ones_like(x)], dim=-1)  # [B, past_days, 2]
        
        # (X^T X)^-1 X^T y
        x_transpose = x.transpose(1, 2)  # [B, 2, past_days]
        x_transpose_x = torch.bmm(x_transpose, x)  # [B, 2, 2]
        x_transpose_x_inv = torch.inverse(x_transpose_x)  # [B, 2, 2]
        x_transpose_y = torch.bmm(x_transpose, daily_mean.unsqueeze(-1))  # [B, 2, 1]
        coefficients = torch.bmm(x_transpose_x_inv, x_transpose_y)  # [B, 2, 1]
        
        slope = coefficients[:, 0, 0]  #[B]
        
        trends = []
        for s in slope:
            if s >= 0.25:
                trends.append("increasing")
            elif s <= -0.25:
                trends.append("decreasing")
            else:
                trends.append("stable")
        return trends
    else:
        # for future trend
        if not isinstance(next_temperatures, torch.Tensor):
            next_temperatures = torch.tensor(next_temperatures, dtype=torch.float32, device=past_temperatures.device)
        
        B_next, M = next_temperatures.shape
        assert B_next == B, f"Batch size does not match: past_temperatures is{B}, next_temperaturesis{B_next}"
        assert M % 24 == 0, f"the future step{M}must be a multiple of 24"
        
        future_days = M // 24 
        
        next_temperatures = next_temperatures.reshape(B, future_days, 24) # [B, future_days, 24]
        
        future_daily_means = torch.mean(next_temperatures, dim=2)  # [B, future_days]
        future_overall_mean = torch.mean(future_daily_means, dim=1)  # [B],
        
        last_day_temp = past_temperatures[:, -1, :]  # [B, 24]
        last_day_mean = torch.mean(last_day_temp, dim=1)  # [B]
        
        diff = future_overall_mean - last_day_mean  # [B]
        
        trends = []
        for d in diff:
            if d >= 0.5:
                trends.append("increasing")
            elif d <= -0.5:
                trends.append("decreasing")
            else:
                trends.append("stable")
        return trends
    


def get_temperature_diff_max_min(temperatures):
    """
    param:
        temperatures: [B, L] (note, agsin, L has to be a multiple of 24)
    
    return:
        a dict, containing keys of ["max", "min", "diff"]
    """
    if not isinstance(temperatures, torch.Tensor):
        temperatures = torch.tensor(temperatures, dtype=torch.float32)
    
    B, L = temperatures.shape
    days = L // 24  
    temperatures = temperatures.reshape(B, days, 24) # [B, days, 24]
    
    daily_max = torch.max(temperatures, dim=2).values  # [B, days]
    daily_min = torch.min(temperatures, dim=2).values  # [B, days]
    daily_diff = daily_max - daily_min  # [B, days]
    
    results = []
    for b in range(B):
        batch_results = []
        for d in range(days):
            batch_results.append({
                "max": round(float(daily_max[b, d]), 1),
                "min": round(float(daily_min[b, d]), 1),
                "diff": round(float(daily_diff[b, d]), 1)
            })
        results.append(batch_results)
    
    return results