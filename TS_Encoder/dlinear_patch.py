from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
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
        return res, moving_mean


class TimeSeriesPatchEncoder(nn.Module):
    def __init__(self, patch_len, hidden_dim):
        super().__init__()
        self.patch_len = patch_len
        self.hidden_dim = hidden_dim
        self.decomposition = series_decomp(25)  # should return (seasonal, trend) in [B, C, L]

        self.linear_seasonal = nn.Linear(patch_len, hidden_dim)
        self.linear_trend = nn.Linear(patch_len, hidden_dim)

    def forward(self, x):
        """
        Input: x - [B, C, L]
        Output: encoded - [B, C, P, D]
        """
        B, C, L = x.shape
        
        seasonal_init, trend_init = self.decomposition(x.permute(0,2,1))
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)  # [B, C, L]

        # Ensure L is divisible by patch_len
        if L % self.patch_len != 0:
            pad_len = self.patch_len - (L % self.patch_len)
            seasonal_init = nn.functional.pad(seasonal_init, (0, pad_len), mode='constant', value=0)
            trend_init = nn.functional.pad(trend_init, (0, pad_len), mode='constant', value=0)
            L += pad_len

        P = L // self.patch_len  #x: [B, C, L] => [B, C, P, patch_len]

        # Reshape to patches: [B, C, P, patch_len]
        seasonal = seasonal_init.view(B, C, P, self.patch_len)
        trend = trend_init.view(B, C, P, self.patch_len)

        seasonal_out = self.linear_seasonal(seasonal)  # [B, C, P, D]
        trend_out = self.linear_trend(trend)           # [B, C, P, D]
        out = seasonal_out + trend_out
        out = out.reshape(B, -1, self.hidden_dim) #[B, CP, d]

        return out