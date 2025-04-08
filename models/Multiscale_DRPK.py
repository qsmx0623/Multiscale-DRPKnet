import torch
import torch.nn as nn
from argparse import Namespace
import torch.nn.functional as F
from layers import DRPK


class RecurrentPattern(torch.nn.Module):
    """
    From the author:
    # Thanks for the contribution of wayhoww.
    # The new implementation uses index arithmetic with modulo to directly gather cyclic data in a single operation,
    # while the original implementation manually rolls and repeats the data through looping.
    # It achieves a significant speed improvement (2x ~ 3x acceleration).
    # See https://github.com/ACAT-SCUT/CycleNet/pull/4 for more details.
    """
    def __init__(self, cycle_len, channel_size):
        super(RecurrentPattern, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len    
        return self.data[gather_index.long()]


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle #basic
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin
        self.cycle_pattern = configs.cycle_pattern
        self.pattern_num = configs.pattern_nums
        self.hidden_dim = configs.batch_size
        self.drpkn = DRPK.DRPKNet(hidden_dim = self.hidden_dim, K = self.pattern_num)
        # self.drn = DRNet(hidden_dim = 64, K=self.pattern_nums) 消融先验知识的实验

        self.cycleQueue = RecurrentPattern(cycle_len=self.cycle_len, channel_size=self.enc_in)

        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear':
            self.model = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            )

    def forward(self, x, x_mark):
       # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)

        if self.cycle_pattern == 'daily':
            cycle_index_daily = x_mark[..., 0] * self.cycle_len  # 每日周期
            cycle_index_daily = cycle_index_daily[:, -1]
            # instance norm
            if self.use_revin:
                seq_mean = torch.mean(x, dim=1, keepdim=True)
                seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
                x = (x - seq_mean) / torch.sqrt(seq_var)
        
            pred = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)
            # remove the cycle of the input data
            x1 = x - self.cycleQueue(cycle_index_daily, self.seq_len)
            # forecasting with channel independence (parameters-sharing)
            y1 = self.model(x1.permute(0, 2, 1)).permute(0, 2, 1)
        
            cycle = self.cycleQueue((cycle_index_daily + self.seq_len) % self.cycle_len, self.pred_len)

            # 初始化一个空的列表来保存每个通道的重构信号
            reconstructed_signal_list = []

            # 遍历每个通道
            for i in range(x.shape[2]):
            # 获取当前通道的pred y1, y2, y3 和 error
                current_pred = pred[:, :, i] # 形状 [B, L]
                current_error = y1[:, :, i] # 形状 [B, L]
                current_cycle = cycle[:, :, i]  # 形状 [B, L]
        
                # 通过 DRNet 计算当前通道的重构信号和权重
                weights = self.drpkn(current_pred.unsqueeze(-1), current_error.unsqueeze(-1), current_cycle.unsqueeze(-1), K = self.pattern_num)
                pred_error_cycle = torch.cat([current_pred.unsqueeze(-1), current_error.unsqueeze(-1), current_cycle.unsqueeze(-1),], dim=2)  
                 
                weights_avg = torch.mean(weights, dim=1)  # [B, 3]
                weights_avg = weights_avg.unsqueeze(1).expand(-1, self.pred_len, -1)  # [B, L, 3]
                # 计算重构信号
                channel_reconstructed = (weights_avg * pred_error_cycle).sum(dim=2)  # 沿着通道维度求和
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1))
                # 将每个通道的重构信号拼接起来，得到 [B, L, enc_in]
                y = torch.cat(reconstructed_signal_list, dim=2)
            
            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean
               
        elif self.cycle_pattern == 'daily+weekly':
            cycle_index_daily = x_mark[..., 0] * self.cycle_len * 7
            cycle_index_daily = cycle_index_daily[:, -1]
            cycle_index_weekly = x_mark[..., 1] * 7  # 每周周期
            cycle_index_weekly = cycle_index_weekly[:, -1]
            cycle_index = cycle_index_daily + cycle_index_weekly  # 合并每日和每周周期

            # instance norm
            if self.use_revin:
                seq_mean = torch.mean(x, dim=1, keepdim=True)
                seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
                x = (x - seq_mean) / torch.sqrt(seq_var)
        
            pred = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)

            x1 = x - self.cycleQueue(cycle_index_daily, self.seq_len) # remove the cycle of the input data
            y1 = self.model(x1.permute(0, 2, 1)).permute(0, 2, 1)
            x2 = x - self.cycleQueue(cycle_index_weekly, self.seq_len)
            y2 = self.model(x2.permute(0, 2, 1)).permute(0, 2, 1)

            cycle = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
            # 初始化一个空的列表来保存每个通道的重构信号
            reconstructed_signal_list = []

            # 遍历每个通道
            for i in range(x.shape[2]):
            # 获取当前通道的pred y1, y2, y3 和 error
                current_pred = pred[:, :, i]
                current_error = torch.stack([y1[:, :, i], y2[:, :, i]], dim=2)# 形状 [B, L, K]
                current_cycle = cycle[:, :, i] # 形状 [B, L]
        
                # 通过 DRNet 计算当前通道的重构信号和权重
                weights = self.drpkn(current_pred.unsqueeze(-1), current_error, current_cycle.unsqueeze(-1), K = self.pattern_num)
                pred_error_cycle = torch.cat([current_pred.unsqueeze(-1), current_error, current_cycle.unsqueeze(-1)], dim=2)
                 
                weights_avg = torch.mean(weights, dim=1)  # [B, 1+K+1]
                weights_avg = weights_avg.unsqueeze(1).expand(-1, self.pred_len, -1)  # [B, L, 1+K+1]
                # 计算重构信号
                channel_reconstructed = (weights_avg * pred_error_cycle).sum(dim=2)  # 沿着通道维度求和
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1)) 
                # 将每个通道的重构信号拼接起来，得到 [B, L, enc_in]
                y = torch.cat(reconstructed_signal_list, dim=2)
            
            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean

        elif self.cycle_pattern == 'daily+monthly':
            cycle_index_daily = x_mark[..., 0] * self.cycle_len * 30
            cycle_index_daily = cycle_index_daily[:, -1]
            cycle_index_monthly = x_mark[..., 2] * 30  # 每月周期
            cycle_index_monthly = cycle_index_monthly[:, -1]
            cycle_index = cycle_index_daily + cycle_index_monthly # 合并每天和每月周期

            # instance norm
            if self.use_revin:
                seq_mean = torch.mean(x, dim=1, keepdim=True)
                seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
                x = (x - seq_mean) / torch.sqrt(seq_var)
        
            pred = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)

            x1 = x - self.cycleQueue(cycle_index_daily, self.seq_len) # remove the cycle of the input data
            y1 = self.model(x1.permute(0, 2, 1)).permute(0, 2, 1)
            x3 = x - self.cycleQueue(cycle_index_monthly, self.seq_len) # remove the cycle of the input data
            y3 = self.model(x3.permute(0, 2, 1)).permute(0, 2, 1)

            cycle = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
            # 初始化一个空的列表来保存每个通道的重构信号
            reconstructed_signal_list = []

            # 遍历每个通道
            for i in range(x.shape[2]):
            # 获取当前通道的pred y1, y3 和 error
                current_pred = pred[:, :, i]
                current_error = torch.stack([y1[:, :, i], y3[:, :, i]], dim=2)# 形状 [B, L, K]
                current_cycle = cycle[:, :, i] # 形状 [B, L]
        
                # 通过 DRNet 计算当前通道的重构信号和权重
                weights = self.drpkn(current_pred.unsqueeze(-1), current_error, current_cycle.unsqueeze(-1), K = self.pattern_num)
                pred_error_cycle = torch.cat([current_pred.unsqueeze(-1), current_error, current_cycle.unsqueeze(-1)], dim=2)
                 
                weights_avg = torch.mean(weights, dim=1)  # [B, 1+K+1]
                weights_avg = weights_avg.unsqueeze(1).expand(-1, self.pred_len, -1)  # [B, L, 1+K+1]
                # 计算重构信号
                 # 计算重构信号
                channel_reconstructed = (weights_avg * pred_error_cycle).sum(dim=2)  # 沿着通道维度求和
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1)) 
                # 将每个通道的重构信号拼接起来，得到 [B, L, enc_in]
                y = torch.cat(reconstructed_signal_list, dim=2)
            
            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean

        elif self.cycle_pattern == 'daily+yearly':
            cycle_index_daily = x_mark[..., 0] * self.cycle_len * 366
            cycle_index_daily = cycle_index_daily[:, -1]
            cycle_index_yearly = x_mark[..., 3] * 366  # 每年周期
            cycle_index_yearly = cycle_index_yearly[:, -1]
            cycle_index = cycle_index_daily + cycle_index_yearly # 合并每日和每年周期

            # instance norm
            if self.use_revin:
                seq_mean = torch.mean(x, dim=1, keepdim=True)
                seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
                x = (x - seq_mean) / torch.sqrt(seq_var)
        
            pred = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)
            x1 = x - self.cycleQueue(cycle_index_daily, self.seq_len) # remove the cycle of the input data
            y1 = self.model(x1.permute(0, 2, 1)).permute(0, 2, 1)
            x4 = x - self.cycleQueue(cycle_index_yearly, self.seq_len) # remove the cycle of the input data
            y4 = self.model(x4.permute(0, 2, 1)).permute(0, 2, 1)

            cycle = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
            # 初始化一个空的列表来保存每个通道的重构信号
            reconstructed_signal_list = []
            # print(y1.shape)
            # 遍历每个通道
            for i in range(x.shape[2]):
            # 获取当前通道的pred y1, y4 和 error
                current_pred = pred[:, :, i]
                current_error = torch.stack([y1[:, :, i], y4[:, :, i]], dim=2)# 形状 [B, L, K]
                current_cycle = cycle[:, :, i] # 形状 [B, L]
        
                # 通过 DRNet 计算当前通道的重构信号和权重
                weights = self.drpkn(current_pred.unsqueeze(-1), current_error, current_cycle.unsqueeze(-1), K = self.pattern_num)
                pred_error_cycle = torch.cat([current_pred.unsqueeze(-1), current_error, current_cycle.unsqueeze(-1)], dim=2)
                 
                weights_avg = torch.mean(weights, dim=1)  # [B, 1+K+1]
                weights_avg = weights_avg.unsqueeze(1).expand(-1, self.pred_len, -1)  # [B, L, 1+K+1]
                # 计算重构信号
                channel_reconstructed = (weights_avg * pred_error_cycle).sum(dim=2)  # 沿着通道维度求和
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1)) 
                # 将每个通道的重构信号拼接起来，得到 [B, L, enc_in]
                y = torch.cat(reconstructed_signal_list, dim=2)
            
            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean

        elif self.cycle_pattern == 'daily+weekly+monthly':
            cycle_index_daily = x_mark[..., 0] * self.cycle_len * 30
            cycle_index_daily = cycle_index_daily[:, -1]
            cycle_index_weekly = x_mark[..., 1] * 7  # 每周周期
            cycle_index_weekly = cycle_index_weekly[:, -1]
            cycle_index_monthly = x_mark[..., 2] * 30  # 每月周期
            cycle_index_monthly = cycle_index_monthly[:, -1]
            cycle_index = cycle_index_daily + cycle_index_weekly +cycle_index_monthly # 合并每日，每周和每月周期

            # instance norm
            if self.use_revin:
                seq_mean = torch.mean(x, dim=1, keepdim=True)
                seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
                x = (x - seq_mean) / torch.sqrt(seq_var)
        
            pred = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)
            x1 = x - self.cycleQueue(cycle_index_daily, self.seq_len) # remove the cycle of the input data
            y1 = self.model(x1.permute(0, 2, 1)).permute(0, 2, 1)
            x2 = x - self.cycleQueue(cycle_index_weekly, self.seq_len)
            y2 = self.model(x2.permute(0, 2, 1)).permute(0, 2, 1)
            x3 = x - self.cycleQueue(cycle_index_monthly, self.seq_len) # remove the cycle of the input data
            y3 = self.model(x3.permute(0, 2, 1)).permute(0, 2, 1)

            cycle = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
            # 初始化一个空的列表来保存每个通道的重构信号
            reconstructed_signal_list = []

            # 遍历每个通道
            for i in range(x.shape[2]):
            # 获取当前通道的pred y1, y2, y3 和 error
                current_pred = pred[:, :, i]
                current_error = torch.stack([y1[:, :, i], y2[:, :, i], y3[:, :, i]], dim=2)# 形状 [B, L, K]
                current_cycle = cycle[:, :, i] # 形状 [B, L]
        
                # 通过 DRNet 计算当前通道的重构信号和权重
                weights = self.drpkn(current_pred.unsqueeze(-1), current_error, current_cycle.unsqueeze(-1), K = self.pattern_num)
                pred_error_cycle = torch.cat([current_pred.unsqueeze(-1), current_error, current_cycle.unsqueeze(-1)], dim=2)
                 
                weights_avg = torch.mean(weights, dim=1)  # [B, 1+K+1]
                weights_avg = weights_avg.unsqueeze(1).expand(-1, self.pred_len, -1)  # [B, L, 1+K+1]
                # 计算重构信号
                 # 计算重构信号
                channel_reconstructed = (weights_avg * pred_error_cycle).sum(dim=2)  # 沿着通道维度求和
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1)) 
                # 将每个通道的重构信号拼接起来，得到 [B, L, enc_in]
                y = torch.cat(reconstructed_signal_list, dim=2)
            
            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean

        elif self.cycle_pattern == 'daily+weekly+yearly':
            cycle_index_daily = x_mark[..., 0] * self.cycle_len * 366
            cycle_index_daily = cycle_index_daily[:, -1]
            cycle_index_weekly = x_mark[..., 1] * 7  # 每周周期
            cycle_index_weekly = cycle_index_weekly[:, -1]
            cycle_index_yearly = x_mark[..., 3] * 366  # 每年周期
            cycle_index_yearly = cycle_index_yearly[:, -1]
            cycle_index = cycle_index_daily + cycle_index_weekly + cycle_index_yearly # 合并每日，每周和每年周期

            # instance norm
            if self.use_revin:
                seq_mean = torch.mean(x, dim=1, keepdim=True)
                seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
                x = (x - seq_mean) / torch.sqrt(seq_var)
        
            pred = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)
            x1 = x - self.cycleQueue(cycle_index_daily, self.seq_len) # remove the cycle of the input data
            y1 = self.model(x1.permute(0, 2, 1)).permute(0, 2, 1)
            x2 = x - self.cycleQueue(cycle_index_weekly, self.seq_len)
            y2 = self.model(x2.permute(0, 2, 1)).permute(0, 2, 1)
            x4 = x - self.cycleQueue(cycle_index_yearly, self.seq_len) # remove the cycle of the input data
            y4 = self.model(x4.permute(0, 2, 1)).permute(0, 2, 1)

            cycle = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
            # 初始化一个空的列表来保存每个通道的重构信号
            reconstructed_signal_list = []
            # print(y1.shape)
            # 遍历每个通道
            for i in range(x.shape[2]):
            # 获取当前通道的pred y1, y2, y4 和 error
                current_pred = pred[:, :, i]
                current_error = torch.stack([y1[:, :, i], y2[:, :, i], y4[:, :, i]], dim=2)# 形状 [B, L, K]
                current_cycle = cycle[:, :, i] # 形状 [B, L]
        
                # 通过 DRNet 计算当前通道的重构信号和权重
                weights = self.drpkn(current_pred.unsqueeze(-1), current_error, current_cycle.unsqueeze(-1), K = self.pattern_num)
                pred_error_cycle = torch.cat([current_pred.unsqueeze(-1), current_error, current_cycle.unsqueeze(-1)], dim=2)
                 
                weights_avg = torch.mean(weights, dim=1)  # [B, 1+K+1]
                weights_avg = weights_avg.unsqueeze(1).expand(-1, self.pred_len, -1)  # [B, L, 1+K+1]
                # 计算重构信号
                channel_reconstructed = (weights_avg * pred_error_cycle).sum(dim=2)  # 沿着通道维度求和
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1)) 
                # 将每个通道的重构信号拼接起来，得到 [B, L, enc_in]
                y = torch.cat(reconstructed_signal_list, dim=2)
            
            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean

        
        elif self.cycle_pattern == 'daily+monthly+yearly':
            cycle_index_daily = x_mark[..., 0] * self.cycle_len * 366
            cycle_index_daily = cycle_index_daily[:, -1]
            cycle_index_monthly = x_mark[..., 2] * 31  # 每月周期
            cycle_index_monthly = cycle_index_monthly[:, -1]
            cycle_index_yearly = x_mark[..., 3] * 366  # 每年周期
            cycle_index_yearly = cycle_index_yearly[:, -1]
            cycle_index = cycle_index_daily  + cycle_index_monthly + cycle_index_yearly # 合并每日，每月和每年周期

            # instance norm
            if self.use_revin:
                seq_mean = torch.mean(x, dim=1, keepdim=True)
                seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
                x = (x - seq_mean) / torch.sqrt(seq_var)
        
            pred = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)
            x1 = x - self.cycleQueue(cycle_index_daily, self.seq_len) # remove the cycle of the input data
            y1 = self.model(x1.permute(0, 2, 1)).permute(0, 2, 1)
            x3 = x - self.cycleQueue(cycle_index_monthly, self.seq_len) # remove the cycle of the input data
            y3 = self.model(x3.permute(0, 2, 1)).permute(0, 2, 1)
            x4 = x - self.cycleQueue(cycle_index_yearly, self.seq_len) # remove the cycle of the input data
            y4 = self.model(x4.permute(0, 2, 1)).permute(0, 2, 1)

            cycle = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
            # 初始化一个空的列表来保存每个通道的重构信号
            reconstructed_signal_list = []
            # print(y1.shape)
            # 遍历每个通道
            for i in range(x.shape[2]):
            # 获取当前通道的pred y1, y3，y4和 error
                current_pred = pred[:, :, i]
                current_error = torch.stack([y1[:, :, i], y3[:, :, i], y4[:, :, i]], dim=2)# 形状 [B, L, K]
                current_cycle = cycle[:, :, i] # 形状 [B, L]
        
                # 通过 DRNet 计算当前通道的重构信号和权重
                weights = self.drpkn(current_pred.unsqueeze(-1), current_error, current_cycle.unsqueeze(-1), K = self.pattern_num)
                pred_error_cycle = torch.cat([current_pred.unsqueeze(-1), current_error, current_cycle.unsqueeze(-1)], dim=2)
                 
                weights_avg = torch.mean(weights, dim=1)  # [B, 1+K+1]
                weights_avg = weights_avg.unsqueeze(1).expand(-1, self.pred_len, -1)  # [B, L, 1+K+1]
                # 计算重构信号
                channel_reconstructed = (weights_avg * pred_error_cycle).sum(dim=2)  # 沿着通道维度求和
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1)) 
                # 将每个通道的重构信号拼接起来，得到 [B, L, enc_in]
                y = torch.cat(reconstructed_signal_list, dim=2)
            
            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean

        elif self.cycle_pattern == 'daily+weekly+monthly+yearly':
            cycle_index_daily = x_mark[..., 0] * self.cycle_len * 366
            cycle_index_daily = cycle_index_daily[:, -1]
            cycle_index_weekly = x_mark[..., 1] * 7  # 每周周期
            cycle_index_weekly = cycle_index_weekly[:, -1]
            cycle_index_monthly = x_mark[..., 2] * 31  # 每月周期
            cycle_index_monthly = cycle_index_monthly[:, -1]
            cycle_index_yearly = x_mark[..., 3] * 366  # 每年周期
            cycle_index_yearly = cycle_index_yearly[:, -1]
            cycle_index = cycle_index_daily + cycle_index_weekly + cycle_index_monthly + cycle_index_yearly # 合并每日，每周，每月和每年周期

            # instance norm
            if self.use_revin:
                seq_mean = torch.mean(x, dim=1, keepdim=True)
                seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
                x = (x - seq_mean) / torch.sqrt(seq_var)
        
            pred = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)
            x1 = x - self.cycleQueue(cycle_index_daily, self.seq_len) # remove the cycle of the input data
            y1 = self.model(x1.permute(0, 2, 1)).permute(0, 2, 1)
            x2 = x - self.cycleQueue(cycle_index_weekly, self.seq_len)
            y2 = self.model(x2.permute(0, 2, 1)).permute(0, 2, 1)
            x3 = x - self.cycleQueue(cycle_index_monthly, self.seq_len) # remove the cycle of the input data
            y3 = self.model(x3.permute(0, 2, 1)).permute(0, 2, 1)
            x4 = x - self.cycleQueue(cycle_index_yearly, self.seq_len) # remove the cycle of the input data
            y4 = self.model(x4.permute(0, 2, 1)).permute(0, 2, 1)

            cycle = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
            # 初始化一个空的列表来保存每个通道的重构信号
            reconstructed_signal_list = []
            # print(y1.shape)
            # 遍历每个通道
            for i in range(x.shape[2]):
            # 获取当前通道的pred y1, y2, y3，y4和 error
                current_pred = pred[:, :, i]
                current_error = torch.stack([y1[:, :, i], y2[:, :, i], y3[:, :, i], y4[:, :, i]], dim=2)# 形状 [B, L, K]
                current_cycle = cycle[:, :, i] # 形状 [B, L]
        
                # 通过 DRNet 计算当前通道的重构信号和权重
                weights = self.drpkn(current_pred.unsqueeze(-1), current_error, current_cycle.unsqueeze(-1), K = self.pattern_num)
                pred_error_cycle = torch.cat([current_pred.unsqueeze(-1), current_error, current_cycle.unsqueeze(-1)], dim=2)
                 
                weights_avg = torch.mean(weights, dim=1)  # [B, 1+K+1]
                weights_avg = weights_avg.unsqueeze(1).expand(-1, self.pred_len, -1)  # [B, L, 1+K+1]
                # 计算重构信号
                channel_reconstructed = (weights_avg * pred_error_cycle).sum(dim=2)  # 沿着通道维度求和
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1)) 
                # 将每个通道的重构信号拼接起来，得到 [B, L, enc_in]
                y = torch.cat(reconstructed_signal_list, dim=2)
            
            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean


        else:
            raise Exception("please specify cycle pattern")

        return y, weights_avg
