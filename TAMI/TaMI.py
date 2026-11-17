import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from iTransformer import iTransformer
import warnings

# 忽略所有 FutureWarning 警告
warnings.simplefilter(action='ignore', category=FutureWarning)

def masked_mae_cal(inputs, target, mask):
    """ calculate Mean Absolute Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)

class SDR(nn.Module):
    """Structural Dependency Recovery (SDR).

    将多变量序列按变量维重组为 system tokens 做跨变量依赖建模，再由 Mamba
    状态演化算子做时序依赖建模。stage="full" 时额外拼接时间间隔 delta，
    stage="base" 时只使用观测值与掩码。
    """
    def __init__(self, n_groups, n_group_inner_layers, seq_len, feature_num, d_model, d_inner, n_head, d_k, d_v, dropout, diagonal_attention_mask, device, stage):
        super().__init__()
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.stage = stage
        if self.stage == "full":
            actual_feature_num = feature_num * 3
        else :
            actual_feature_num = feature_num * 2

        self.backbone = iTransformer(
            num_variates=feature_num,  # 特征数
            lookback_len=seq_len,  # 序列长度
            depth=6,  # 深度，可根据需求调整
            dim=d_model,  # 模型维度
            num_tokens_per_variate=1,  # 可根据需求调整
            pred_length=seq_len,  # 预测长度
            dim_head=d_k,  # 注意力头的维度
            heads=n_head,  # 多头注意力头数
            attn_dropout=dropout,
            ff_mult=4,  # 前馈网络放大倍数
            ff_dropout=dropout,
            num_mem_tokens=2,  # memory tokens 数量，可调整
            use_reversible_instance_norm=False,
            reversible_instance_norm_affine=False,
            flash_attn=True,  # 使用 flash attention
            use_mamba=True
        )
        self.embedding_1 = nn.Linear(actual_feature_num, d_model)
        self.reduce_dim = nn.Linear(d_model, feature_num)


    def forward(self, X,masks,delta=None):
        if self.stage == "full":
            x = torch.cat([X, delta], dim=2)
            input_X1 = torch.cat([x, masks], dim=2)
        else :
            input_X1 = torch.cat([X, masks], dim=2)
        enc_output = self.backbone(X)
        result_1 = list(enc_output.values())[0]
        result_temp = masks * X + (1 - masks) * result_1
        return result_temp,result_1


class TaMI(nn.Module):
    """TaMI 的完整插补模型，内部由两个 SDR 分支组成。"""
    def __init__(self, n_groups, n_group_inner_layers, seq_len, feature_num, d_model, d_inner, n_head, d_k, d_v, dropout, diagonal_attention_mask, device):
        super(TaMI, self).__init__()
        self.encoder=SDR(n_groups, n_group_inner_layers, seq_len, feature_num, d_model, d_inner, n_head, d_k, d_v, dropout, diagonal_attention_mask, device, stage="full")
        self.encoder2=SDR(n_groups, n_group_inner_layers, seq_len, feature_num, d_model, d_inner, n_head, d_k, d_v, dropout, diagonal_attention_mask, device, stage="base")

    def forward(self, x, m, delta = None):
        if delta is None:
            output = self.encoder2(x, m)
        else:
            output = self.encoder(x, m, delta)
        return output



class SDR_loss(nn.Module):
    """Structural Dependency Recovery 目标: L_SDR = L_miss + alpha * L_obs。

    L_obs  在观测位置(掩码为 1)上约束重建保真度，
    L_miss 在被 mask 掉的位置上约束缺失值恢复。
    """
    def __init__(self):
        super().__init__()

    def forward(self, X, M, X_recon, X_holdout, indicating_mask, alpha):
        L_obs = torch.sum((M * X - M * X_recon) ** 2) / torch.sum(M)
        L_miss = torch.sum((indicating_mask * X_holdout - indicating_mask * X_recon) ** 2) / torch.sum(indicating_mask)
        return alpha[0] * L_obs + alpha[1] * L_miss

