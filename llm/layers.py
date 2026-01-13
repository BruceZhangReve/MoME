import torch
import torch.nn as nn


# Let's put align layers here
class AlignLayer(nn.Module):
    def __init__(self, input_dim, output_dim, init_scale=0.1):
        super(AlignLayer, self).__init__()
        self.pre_ln  = nn.LayerNorm(input_dim, eps=1e-5)
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.xavier_normal_(self.linear.weight, gain=0.5)
        self.post_ln = nn.LayerNorm(output_dim, eps=1e-5, elementwise_affine=False)

    def forward(self, x):
        x = self.pre_ln(x)
        x = self.linear(x)
        x = self.post_ln(x)
        return x


# We use these head for llm-based method
class RegressionHead(nn.Module):
    """
    In: [B, T, d]
    Out: [B, out_seq]
    """
    def __init__(self, hidden_dim, ts_token_num, target_window):
        super().__init__()
        self.ts_token_num = ts_token_num
        self.proj = nn.Linear(hidden_dim, hidden_dim//64) # [B, T, d] -> [B, T, d'], perhaps 2048 -> 128
        self.flatten = nn.Flatten(start_dim=-2)  # [B, T, d] -> [B, T*d]
        self.forecaster = nn.Linear(ts_token_num * (hidden_dim//64), target_window)
        #It's simply too much, perhaps we try ITFormer's way (just pick a few indicator token)

    def forward(self, x):
        # x: [B, T, d]
        x = x[:, :self.ts_token_num, :]  # [B, t, d] denote t to be ts_token_num
        x = self.proj(x) # [B, T, d] -> [B, T, d'], perhaps 2048 -> 128
        x = self.flatten(x)  # [B, t, d] -> [B, t*d]
        res = self.forecaster(x) # [B, out_seq]
        return res


class ClassificationHead(nn.Module):
    """
    In: [B, T, d]
    Out: [B, C]
    """
    def __init__(self, hidden_dim, ts_token_num, num_classes):
        super().__init__()
        self.ts_token_num = ts_token_num
        self.proj = nn.Linear(hidden_dim, hidden_dim//64) # [B, T, d] -> [B, T, d'], perhaps 2048 -> 128
        self.flatten = nn.Flatten(start_dim=-2)  # [B, T, d] -> [B, T*d]
        self.forecaster = nn.Linear(ts_token_num * (hidden_dim//64), num_classes)

    def forward(self, x):
        # x: [B, T, d]
        x = x[:, :self.ts_token_num, :]  # [B, t, d] denote t to be ts_token_num
        x = self.proj(x) # [B, T, d] -> [B, T, d'], perhaps 2048 -> 128
        x = self.flatten(x)  # [B, t, d] -> [B, t*d]
        res = self.forecaster(x) # [B, out_seq]
        return res


#################################################################################
# We use the following head for late-fusion based approach

class RegressionHead_latefusion(nn.Module):
    """
    In: [B, d]
    Out: [B, out_seq]
    """
    def __init__(self, hidden_dim, target_window):
        super().__init__()
        self.forecaster = nn.Linear(hidden_dim, target_window)

    def forward(self, x):
        # x: [B, d] or [B, 2d] depending on how to conduct the late fusion
        return self.forecaster(x) # [B, out_seq]


class ClassificationHead_latefusion(nn.Module):
    """
    In: [B, d]
    Out: [B, out_seq]
    """
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.forecaster = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, d] or [B, 2d] depending on how to conduct the late fusion
        return self.forecaster(x) # [B, out_seq]



#################################################################################
# We use these head for tsencoder only-based method
# in order to make a fair comparison, we mannuly set the dimension to be same as above
class RegressionHeadforTMoE(nn.Module):
    """
    In: [B, T, d]
    Out: [B, out_seq]
    """
    def __init__(self, hidden_dim, ts_token_num, target_window):
        super().__init__()
        self.ts_token_num = ts_token_num
        self.proj = nn.Linear(hidden_dim, 2048//64) # [B, T, d] -> [B, T, d'], perhaps 2048 -> 128
        self.flatten = nn.Flatten(start_dim=-2)  # [B, T, d] -> [B, T*d]
        self.forecaster = nn.Linear(ts_token_num * (2048//64), target_window)

    def forward(self, x):
        # x: [B, T, d]
        x = x[:, :self.ts_token_num, :]  # [B, t, d] denote t to be ts_token_num
        x = self.proj(x)
        x = self.flatten(x)  # [B, t, d] -> [B, t*d]
        res = self.forecaster(x) # [B, out_seq]
        return res
    

class ClassificationHeadforTMoE(nn.Module):
    """
    In: [B, T, d]
    Out: [B, C]
    """
    def __init__(self, hidden_dim, ts_token_num, num_classes):
        super().__init__()
        self.ts_token_num = ts_token_num
        self.proj = nn.Linear(hidden_dim, 2048//64) # [B, T, d] -> [B, T, d'], perhaps 2048 -> 128
        self.flatten = nn.Flatten(start_dim=-2)  # [B, T, d] -> [B, T*d]
        self.forecaster = nn.Linear(ts_token_num * (2048//64), num_classes)

    def forward(self, x):
        # x: [B, T, d]
        x = x[:, :self.ts_token_num, :]  # [B, t, d] denote t to be ts_token_num
        x = self.proj(x) # [B, T, d] -> [B, T, d'], perhaps 2048 -> 128
        x = self.flatten(x)  # [B, t, d] -> [B, t*d]
        res = self.forecaster(x) # [B, out_seq]
        return res



class RegressionHeadforMoMe(nn.Module):
    """
    In: [B, CP, d]
    Out: [B, out_seq]
    """
    def __init__(self, hidden_dim, ts_token_num, target_window):
        super().__init__()
        self.ts_token_num = ts_token_num
        self.flatten = nn.Flatten(start_dim=-2)  # [B, T, d] -> [B, T*d]
        self.forecaster = nn.Linear(ts_token_num * hidden_dim, target_window)

    def forward(self, x):
        # x: [B, T, d]
        #print(x.shape, self.ts_token_num)
        x = self.flatten(x)  # [B, t, d] -> [B, t*d]
        res = self.forecaster(x) # [B, out_seq]
        return res