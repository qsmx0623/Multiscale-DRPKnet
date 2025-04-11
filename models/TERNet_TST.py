from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from layers import CWS_Combiner
from layers.PatchTST_backbone import PatchTST_backbone

class RecurrentPattern(torch.nn.Module):
    def __init__(self, pattern_len, channel_size):
        super(RecurrentPattern, self).__init__()
        self.pattern_len = pattern_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(pattern_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.pattern_len    
        return self.data[gather_index.long()]


class Model(nn.Module):
    def __init__(self, configs,max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.pattern_len = configs.pattern #basic
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin
        self.cycle_pattern = configs.cycle_pattern
        self.pattern_num = configs.pattern_nums
        self.hidden_dim = configs.batch_size
        self.cswnet = CWS_Combiner.CWSNet(hidden_dim = self.hidden_dim, K = self.pattern_num)

        self.patternQueue = RecurrentPattern(pattern_len=self.pattern_len, channel_size=self.enc_in)

        #传入PatchTST的配置
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        # 使用 PatchTST 的 Backbone 替换现有模型
        self.patch_tst = PatchTST_backbone(
            c_in=self.enc_in, context_window=self.seq_len, target_window=self.pred_len, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs
        )

    def forward(self, x, x_mark):
       # x: (batch_size, seq_len, enc_in), pattern_index: (batch_size,)

        if self.cycle_pattern == 'daily':
            pattern_index_daily = x_mark[..., 0] * self.pattern_len  # 每日周期
            pattern_index_daily = pattern_index_daily[:, -1]
            # instance norm
            if self.use_revin:
                seq_mean = torch.mean(x, dim=1, keepdim=True)
                seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
                x = (x - seq_mean) / torch.sqrt(seq_var)
        
            pred = self.patch_tst(x.permute(0, 2, 1)).permute(0, 2, 1)
            # remove the pattern of the input data
            x1 = x - self.patternQueue(pattern_index_daily, self.seq_len)
            # forecasting with channel independence (parameters-sharing)
            y1 = self.patch_tst(x1.permute(0, 2, 1)).permute(0, 2, 1)

            pattern_daily = self.patternQueue(pattern_index_daily, self.seq_len)
            pattern = self.patternQueue((pattern_index_daily + self.seq_len) % self.pattern_len, self.pred_len)

            # 初始化三个空的列表来保存每个通道的中间变量
            reconstructed_signal_list = []
            weights_list = []
            current_error_list =[]
            pred_error_pattern_list = []
            pattern_list = []

            # 遍历每个通道
            for i in range(x.shape[2]):
            # 获取当前通道的pred y1 和 error
                current_pred = pred[:, :, i] # 形状 [B, L]
                error = x1[:, :, i]
                current_error = y1[:, :, i] # 形状 [B, L]
                current_pattern = pattern[:, :, i]  # 形状 [B, L]
        
                current_error_list.append(error)
                pattern_list.append(pattern_daily[:, :, i])
                                                 
                # 通过 CWS 计算当前通道的重构信号和权重
                weights = self.cswnet(current_pred.unsqueeze(-1), current_error.unsqueeze(-1), current_pattern.unsqueeze(-1), K = self.pattern_num)
                pred_error_pattern = torch.cat([current_pred.unsqueeze(-1), current_error.unsqueeze(-1), current_pattern.unsqueeze(-1),], dim=2)  
                 
                weights_avg = torch.mean(weights, dim=1)  # [B, 3]
                weights_avg = weights_avg.unsqueeze(1).expand(-1, self.pred_len, -1)  # 沿着时间维度求均值——因为权重的优化是一个和时间无关的过程
                
                # 计算重构信号
                channel_reconstructed = (weights_avg * pred_error_pattern).sum(dim=2)  # 沿着通道维度求和
                weights = torch.unique(weights_avg, dim=1) # 保留时间维度上的不重复样本
                weights_list.append(weights.squeeze(1)) #去掉第二个维度，得到形状[B, 1+K+1]
                pred_error_pattern_list.append(pred_error_pattern.squeeze(1))
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1)) 
                
                # 将每个通道的中间拼接起来，得到 [B, L, enc_in]
                y = torch.cat(reconstructed_signal_list, dim=2)
                weights=torch.cat(weights_list, dim=1)
                patterns = pattern_list
                error=current_error_list
                pred_error_perio=torch.cat(pred_error_pattern_list, dim=2)
            
            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean
               
        elif self.cycle_pattern == 'daily+weekly':
            pattern_index_daily = x_mark[..., 0] * self.pattern_len * 7
            pattern_index_daily = pattern_index_daily[:, -1]
            pattern_index_weekly = x_mark[..., 1] * 7  # 每周周期
            pattern_index_weekly = pattern_index_weekly[:, -1]
            pattern_index = pattern_index_daily + pattern_index_weekly  # 合并每日和每周周期

            # instance norm
            if self.use_revin:
                seq_mean = torch.mean(x, dim=1, keepdim=True)
                seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
                x = (x - seq_mean) / torch.sqrt(seq_var)
        
            pred = self.patch_tst(x.permute(0, 2, 1)).permute(0, 2, 1)

            x1 = x - self.patternQueue(pattern_index_daily, self.seq_len) # remove the pattern of the input data
            y1 = self.patch_tst(x1.permute(0, 2, 1)).permute(0, 2, 1)
            x2 = x - self.patternQueue(pattern_index_weekly, self.seq_len)
            y2 = self.patch_tst(x2.permute(0, 2, 1)).permute(0, 2, 1)

            # Save patternQueue outputs as intermediate variables
            pattern_daily = self.patternQueue(pattern_index_daily, self.seq_len)
            pattern_weekly = self.patternQueue(pattern_index_weekly, self.seq_len)

            pattern = self.patternQueue((pattern_index + self.seq_len) % self.pattern_len, self.pred_len)
            
            # 初始化空的列表来保存每个通道的中间变量
            reconstructed_signal_list = []
            weights_list = []
            current_error_list =[]
            pred_error_pattern_list = []
            pattern_list = []  # List to store the intermediate pattern values

            # 遍历每个通道
            for i in range(x.shape[2]):
            # 获取当前通道的pred y1, y2 和 error
                current_pred = pred[:, :, i]
                current_error = torch.stack([y1[:, :, i], y2[:, :, i]], dim=2)# 形状 [B, L, K]
                current_pattern = pattern[:, :, i] # 形状 [B, L]
        
                current_error_list.append(current_error)
                # Append patterns to the pattern_list
                pattern_list.append(torch.stack([pattern_daily[:, :, i], pattern_weekly[:, :, i]], dim=2))

                # 通过 CWS 计算当前通道的重构信号和权重
                weights = self.cswnet(current_pred.unsqueeze(-1), current_error, current_pattern.unsqueeze(-1), K = self.pattern_num)
                pred_error_pattern = torch.cat([current_pred.unsqueeze(-1), current_error, current_pattern.unsqueeze(-1)], dim=2)
                 
                weights_avg = torch.mean(weights, dim=1)  # [B, 1+K+1]
                weights_avg = weights_avg.unsqueeze(1).expand(-1, self.pred_len, -1)  # 沿着时间维度求均值——因为权重的优化是一个和时间无关的过程
                
                # 计算重构信号
                channel_reconstructed = (weights_avg * pred_error_pattern).sum(dim=2)  # 沿着通道维度求和
                weights = torch.unique(weights_avg, dim=1) # 保留时间维度上的不重复样本
                weights_list.append(weights.squeeze(1)) #去掉第二个维度，得到形状[B, 1+K+1]
                pred_error_pattern_list.append(pred_error_pattern.squeeze(1))
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1)) 

                # 将每个通道的中间拼接起来，得到 [B, L, enc_in]
                y = torch.cat(reconstructed_signal_list, dim=2)
                weights=torch.cat(weights_list, dim=1)
                patterns = torch.cat(pattern_list, dim=2)
                error=torch.cat(current_error_list, dim=2)
                pred_error_perio=torch.cat(pred_error_pattern_list, dim=2)
            
            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean

        elif self.cycle_pattern == 'daily+monthly':
            pattern_index_daily = x_mark[..., 0] * self.pattern_len * 30
            pattern_index_daily = pattern_index_daily[:, -1]
            pattern_index_monthly = x_mark[..., 2] * 30  # 每月周期
            pattern_index_monthly = pattern_index_monthly[:, -1]
            pattern_index = pattern_index_daily + pattern_index_monthly # 合并每天和每月周期

            # instance norm
            if self.use_revin:
                seq_mean = torch.mean(x, dim=1, keepdim=True)
                seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
                x = (x - seq_mean) / torch.sqrt(seq_var)
        
            pred = self.patch_tst(x.permute(0, 2, 1)).permute(0, 2, 1)

            x1 = x - self.patternQueue(pattern_index_daily, self.seq_len) # remove the pattern of the input data
            y1 = self.patch_tst(x1.permute(0, 2, 1)).permute(0, 2, 1)
            x3 = x - self.patternQueue(pattern_index_monthly, self.seq_len) # remove the pattern of the input data
            y3 = self.patch_tst(x3.permute(0, 2, 1)).permute(0, 2, 1)

            pattern_daily = self.patternQueue(pattern_index_daily, self.seq_len)
            pattern_monthly = self.patternQueue(pattern_index_monthly, self.seq_len)

            pattern = self.patternQueue((pattern_index + self.seq_len) % self.pattern_len, self.pred_len)
            
            # 初始化三个空的列表来保存每个通道的中间变量
            reconstructed_signal_list = []
            weights_list = []
            current_error_list =[]
            pred_error_pattern_list = []
            pattern_list = [] 

            # 遍历每个通道
            for i in range(x.shape[2]):
            # 获取当前通道的pred y1, y3 和 error
                current_pred = pred[:, :, i]
                error = torch.stack([x1[:, :, i], x3[:, :, i]], dim=2)
                current_error = torch.stack([y1[:, :, i], y3[:, :, i]], dim=2)# 形状 [B, L, K]
                current_pattern = pattern[:, :, i] # 形状 [B, L]
        
                current_error_list.append(error)
                pattern_list.append(torch.stack([pattern_daily[:, :, i], pattern_monthly[:, :, i]], dim=2))
                
                # 通过 CWS 计算当前通道的重构信号和权重
                weights = self.cswnet(current_pred.unsqueeze(-1), current_error, current_pattern.unsqueeze(-1), K = self.pattern_num)
                pred_error_pattern = torch.cat([current_pred.unsqueeze(-1), current_error, current_pattern.unsqueeze(-1)], dim=2)
                 
                weights_avg = torch.mean(weights, dim=1)  # [B, 1+K+1]
                weights_avg = weights_avg.unsqueeze(1).expand(-1, self.pred_len, -1)  # 沿着时间维度求均值——因为权重的优化是一个和时间无关的过程
                
                # 计算重构信号
                channel_reconstructed = (weights_avg * pred_error_pattern).sum(dim=2)  # 沿着通道维度求和
                weights = torch.unique(weights_avg, dim=1) # 保留时间维度上的不重复样本
                weights_list.append(weights.squeeze(1)) #去掉第二个维度，得到形状[B, 1+K+1]
                pred_error_pattern_list.append(pred_error_pattern.squeeze(1))
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1)) 
                
                # 将每个通道的中间拼接起来，得到 [B, L, enc_in]
                y = torch.cat(reconstructed_signal_list, dim=2)
                weights=torch.cat(weights_list, dim=1)
                patterns = torch.cat(pattern_list, dim=2)
                error=torch.cat(current_error_list, dim=2)
                pred_error_perio=torch.cat(pred_error_pattern_list, dim=2)
            
            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean

        elif self.cycle_pattern == 'daily+yearly':
            pattern_index_daily = x_mark[..., 0] * self.pattern_len * 366
            pattern_index_daily = pattern_index_daily[:, -1]
            pattern_index_yearly = x_mark[..., 3] * 366  # 每年周期
            pattern_index_yearly = pattern_index_yearly[:, -1]
            pattern_index = pattern_index_daily + pattern_index_yearly # 合并每日和每年周期

            # instance norm
            if self.use_revin:
                seq_mean = torch.mean(x, dim=1, keepdim=True)
                seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
                x = (x - seq_mean) / torch.sqrt(seq_var)
        
            pred = self.patch_tst(x.permute(0, 2, 1)).permute(0, 2, 1)
            x1 = x - self.patternQueue(pattern_index_daily, self.seq_len) # remove the pattern of the input data
            y1 = self.patch_tst(x1.permute(0, 2, 1)).permute(0, 2, 1)
            x4 = x - self.patternQueue(pattern_index_yearly, self.seq_len) # remove the pattern of the input data
            y4 = self.patch_tst(x4.permute(0, 2, 1)).permute(0, 2, 1)

            pattern_daily = self.patternQueue(pattern_index_daily, self.seq_len)
            pattern_yearly = self.patternQueue(pattern_index_yearly, self.seq_len)

            pattern = self.patternQueue((pattern_index + self.seq_len) % self.pattern_len, self.pred_len)
            
            # 初始化三个空的列表来保存每个通道的中间变量
            reconstructed_signal_list = []
            weights_list = []
            current_error_list =[]
            pred_error_pattern_list = []
            pattern_list = []  # List to store the intermediate pattern values

            # print(y1.shape)
            # 遍历每个通道
            for i in range(x.shape[2]):
            # 获取当前通道的pred y1, y4 和 error
                current_pred = pred[:, :, i]
                error = torch.stack([x1[:, :, i], x4[:, :, i]], dim=2)
                current_error = torch.stack([y1[:, :, i], y4[:, :, i]], dim=2)# 形状 [B, L, K]
                current_pattern = pattern[:, :, i] # 形状 [B, L]
        
                current_error_list.append(error)
                pattern_list.append(torch.stack([pattern_daily[:, :, i], pattern_yearly[:, :, i]], dim=2))
                
                # 通过 CWS 计算当前通道的重构信号和权重
                weights = self.cswnet(current_pred.unsqueeze(-1), current_error, current_pattern.unsqueeze(-1), K = self.pattern_num)
                pred_error_pattern = torch.cat([current_pred.unsqueeze(-1), current_error, current_pattern.unsqueeze(-1)], dim=2)
                 
                weights_avg = torch.mean(weights, dim=1)  # [B, 1+K+1]
                weights_avg = weights_avg.unsqueeze(1).expand(-1, self.pred_len, -1)  # 沿着时间维度求均值——因为权重的优化是一个和时间无关的过程
                
                # 计算重构信号
                channel_reconstructed = (weights_avg * pred_error_pattern).sum(dim=2)  # 沿着通道维度求和
                weights = torch.unique(weights_avg, dim=1) # 保留时间维度上的不重复样本
                weights_list.append(weights.squeeze(1)) #去掉第二个维度，得到形状[B, 1+K+1]
                pred_error_pattern_list.append(pred_error_pattern.squeeze(1))
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1)) 
                
                # 将每个通道的中间拼接起来，得到 [B, L, enc_in]
                y = torch.cat(reconstructed_signal_list, dim=2)
                weights=torch.cat(weights_list, dim=1)
                patterns = torch.cat(pattern_list, dim=2)
                error=torch.cat(current_error_list, dim=2)
                pred_error_perio=torch.cat(pred_error_pattern_list, dim=2)
            
            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean

        elif self.cycle_pattern == 'daily+weekly+monthly':
            pattern_index_daily = x_mark[..., 0] * self.pattern_len * 30
            pattern_index_daily = pattern_index_daily[:, -1]
            pattern_index_weekly = x_mark[..., 1] * 7  # 每周周期
            pattern_index_weekly = pattern_index_weekly[:, -1]
            pattern_index_monthly = x_mark[..., 2] * 30  # 每月周期
            pattern_index_monthly = pattern_index_monthly[:, -1]
            pattern_index = pattern_index_daily + pattern_index_weekly +pattern_index_monthly # 合并每日，每周和每月周期

            # instance norm
            if self.use_revin:
                seq_mean = torch.mean(x, dim=1, keepdim=True)
                seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
                x = (x - seq_mean) / torch.sqrt(seq_var)
        
            pred = self.patch_tst(x.permute(0, 2, 1)).permute(0, 2, 1)
            x1 = x - self.patternQueue(pattern_index_daily, self.seq_len) # remove the pattern of the input data
            y1 = self.patch_tst(x1.permute(0, 2, 1)).permute(0, 2, 1)
            x2 = x - self.patternQueue(pattern_index_weekly, self.seq_len)
            y2 = self.patch_tst(x2.permute(0, 2, 1)).permute(0, 2, 1)
            x3 = x - self.patternQueue(pattern_index_monthly, self.seq_len) # remove the pattern of the input data
            y3 = self.patch_tst(x3.permute(0, 2, 1)).permute(0, 2, 1)

            # Save patternQueue outputs as intermediate variables
            pattern_daily = self.patternQueue(pattern_index_daily, self.seq_len)
            pattern_weekly = self.patternQueue(pattern_index_weekly, self.seq_len)
            pattern_monthly = self.patternQueue(pattern_index_monthly, self.seq_len)

            pattern = self.patternQueue((pattern_index + self.seq_len) % self.pattern_len, self.pred_len)
            
            # 初始化三个空的列表来保存每个通道的中间变量
            reconstructed_signal_list = []
            weights_list = []
            current_error_list = []
            pred_error_pattern_list = []
            pattern_list = []

            # 遍历每个通道
            for i in range(x.shape[2]):
            # 获取当前通道的pred y1, y2, y3 和 error
                current_pred = pred[:, :, i]
                error = torch.stack([x1[:, :, i], x2[:, :, i], x3[:, :, i]], dim=2)
                current_error = torch.stack([y1[:, :, i], y2[:, :, i], y3[:, :, i]], dim=2)# 形状 [B, L, K]
                current_pattern = pattern[:, :, i] # 形状 [B, L]
        
                current_error_list.append(error)
                pattern_list.append(torch.stack([pattern_daily[:, :, i], pattern_weekly[:, :, i], pattern_monthly[:, :, i]], dim=2))

                # 通过 CWS 计算当前通道的重构信号和权重
                weights = self.cswnet(current_pred.unsqueeze(-1), current_error, current_pattern.unsqueeze(-1), K = self.pattern_num)
                pred_error_pattern = torch.cat([current_pred.unsqueeze(-1), current_error, current_pattern.unsqueeze(-1)], dim=2)
                 
                weights_avg = torch.mean(weights, dim=1)  # [B, 1+K+1]
                weights_avg = weights_avg.unsqueeze(1).expand(-1, self.pred_len, -1)  # 沿着时间维度求均值——因为权重的优化是一个和时间无关的过程
                
                # 计算重构信号
                channel_reconstructed = (weights_avg * pred_error_pattern).sum(dim=2)  # 沿着通道维度求和
                weights = torch.unique(weights_avg, dim=1) # 保留时间维度上的不重复样本
                weights_list.append(weights.squeeze(1)) #去掉第二个维度，得到形状[B, 1+K+1]
                pred_error_pattern_list.append(pred_error_pattern.squeeze(1))
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1)) 
                
                # 将每个通道的中间拼接起来，得到 [B, L, enc_in]
                y = torch.cat(reconstructed_signal_list, dim=2)
                weights=torch.cat(weights_list, dim=1)
                patterns = torch.cat(pattern_list, dim=2)
                error=torch.cat(current_error_list, dim=2)
                pred_error_perio=torch.cat(pred_error_pattern_list, dim=2)
            
            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean

        elif self.cycle_pattern == 'daily+weekly+yearly':
            pattern_index_daily = x_mark[..., 0] * self.pattern_len * 366
            pattern_index_daily = pattern_index_daily[:, -1]
            pattern_index_weekly = x_mark[..., 1] * 7  # 每周周期
            pattern_index_weekly = pattern_index_weekly[:, -1]
            pattern_index_yearly = x_mark[..., 3] * 366  # 每年周期
            pattern_index_yearly = pattern_index_yearly[:, -1]
            pattern_index = pattern_index_daily + pattern_index_weekly + pattern_index_yearly # 合并每日，每周和每年周期

            # instance norm
            if self.use_revin:
                seq_mean = torch.mean(x, dim=1, keepdim=True)
                seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
                x = (x - seq_mean) / torch.sqrt(seq_var)
        
            pred = self.patch_tst(x.permute(0, 2, 1)).permute(0, 2, 1)
            x1 = x - self.patternQueue(pattern_index_daily, self.seq_len) # remove the pattern of the input data
            y1 = self.patch_tst(x1.permute(0, 2, 1)).permute(0, 2, 1)
            x2 = x - self.patternQueue(pattern_index_weekly, self.seq_len)
            y2 = self.patch_tst(x2.permute(0, 2, 1)).permute(0, 2, 1)
            x4 = x - self.patternQueue(pattern_index_yearly, self.seq_len) # remove the pattern of the input data
            y4 = self.patch_tst(x4.permute(0, 2, 1)).permute(0, 2, 1)

            # Save patternQueue outputs as intermediate variables
            pattern_daily = self.patternQueue(pattern_index_daily, self.seq_len)
            pattern_weekly = self.patternQueue(pattern_index_weekly, self.seq_len)
            pattern_yearly = self.patternQueue(pattern_index_yearly, self.seq_len)

            pattern = self.patternQueue((pattern_index + self.seq_len) % self.pattern_len, self.pred_len)
            
            # 初始化三个空的列表来保存每个通道的中间变量
            reconstructed_signal_list = []
            weights_list = []
            current_error_list =[]
            pred_error_pattern_list = []
            pattern_list = [] 

            # print(y1.shape)
            # 遍历每个通道
            for i in range(x.shape[2]):
            # 获取当前通道的pred y1, y2, y4 和 error
                current_pred = pred[:, :, i]
                error = torch.stack([x1[:, :, i], x2[:, :, i], x4[:, :, i]], dim=2)#
                current_error = torch.stack([y1[:, :, i], y2[:, :, i], y4[:, :, i]], dim=2)# 形状 [B, L, K]
                current_pattern = pattern[:, :, i] # 形状 [B, L]
        
                current_error_list.append(error)
                pattern_list.append(torch.stack([pattern_daily[:, :, i], pattern_weekly[:, :, i], pattern_yearly[:, :, i]], dim=2))

                # 通过 CWS 计算当前通道的重构信号和权重
                weights = self.cswnet(current_pred.unsqueeze(-1), current_error, current_pattern.unsqueeze(-1), K = self.pattern_num)
                pred_error_pattern = torch.cat([current_pred.unsqueeze(-1), current_error, current_pattern.unsqueeze(-1)], dim=2)
                 
                weights_avg = torch.mean(weights, dim=1)  # [B, 1+K+1]
                weights_avg = weights_avg.unsqueeze(1).expand(-1, self.pred_len, -1)  # 沿着时间维度求均值——因为权重的优化是一个和时间无关的过程
                
                # 计算重构信号
                channel_reconstructed = (weights_avg * pred_error_pattern).sum(dim=2)  # 沿着通道维度求和
                weights = torch.unique(weights_avg, dim=1) # 保留时间维度上的不重复样本
                weights_list.append(weights) 
                pred_error_pattern_list.append(pred_error_pattern.unsqueeze(-1))
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1)) 

                # 将每个通道的中间拼接起来，得到 [B, L, enc_in]
                y = torch.cat(reconstructed_signal_list, dim=2)
                weights=torch.cat(weights_list, dim=1)
                patterns = torch.cat(pattern_list, dim=2)
                error=torch.cat(current_error_list, dim=2)
                pred_error_perio=torch.cat(pred_error_pattern_list, dim=2)

            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean

        
        elif self.cycle_pattern == 'daily+monthly+yearly':
            pattern_index_daily = x_mark[..., 0] * self.pattern_len * 366
            pattern_index_daily = pattern_index_daily[:, -1]
            pattern_index_monthly = x_mark[..., 2] * 31  # 每月周期
            pattern_index_monthly = pattern_index_monthly[:, -1]
            pattern_index_yearly = x_mark[..., 3] * 366  # 每年周期
            pattern_index_yearly = pattern_index_yearly[:, -1]
            pattern_index = pattern_index_daily  + pattern_index_monthly + pattern_index_yearly # 合并每日，每月和每年周期

            # instance norm
            if self.use_revin:
                seq_mean = torch.mean(x, dim=1, keepdim=True)
                seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
                x = (x - seq_mean) / torch.sqrt(seq_var)
        
            pred = self.patch_tst(x.permute(0, 2, 1)).permute(0, 2, 1)
            x1 = x - self.patternQueue(pattern_index_daily, self.seq_len) # remove the pattern of the input data
            y1 = self.patch_tst(x1.permute(0, 2, 1)).permute(0, 2, 1)
            x3 = x - self.patternQueue(pattern_index_monthly, self.seq_len) # remove the pattern of the input data
            y3 = self.patch_tst(x3.permute(0, 2, 1)).permute(0, 2, 1)
            x4 = x - self.patternQueue(pattern_index_yearly, self.seq_len) # remove the pattern of the input data
            y4 = self.patch_tst(x4.permute(0, 2, 1)).permute(0, 2, 1)

            pattern_daily = self.patternQueue(pattern_index_daily, self.seq_len)
            pattern_monthly = self.patternQueue(pattern_index_monthly, self.seq_len)
            pattern_yearly = self.patternQueue(pattern_index_yearly, self.seq_len)

            pattern = self.patternQueue((pattern_index + self.seq_len) % self.pattern_len, self.pred_len)
            
            # 初始化三个空的列表来保存每个通道的中间变量
            reconstructed_signal_list = []
            weights_list = []
            current_error_list = []
            pred_error_pattern_list = []
            pattern_list = []  # List to store the intermediate pattern values

            # print(y1.shape)
            # 遍历每个通道
            for i in range(x.shape[2]):
            # 获取当前通道的pred y1, y3，y4和 error
                current_pred = pred[:, :, i]
                error = torch.stack([x1[:, :, i], x3[:, :, i], x4[:, :, i]], dim=2)
                current_error = torch.stack([y1[:, :, i], y3[:, :, i], y4[:, :, i]], dim=2)# 形状 [B, L, K]
                current_pattern = pattern[:, :, i] # 形状 [B, L]

                current_error_list.append(error)
                pattern_list.append(torch.stack([pattern_daily[:, :, i], pattern_monthly[:, :, i], pattern_yearly[:, :, i]], dim=2))

                # 通过 CWS 计算当前通道的重构信号和权重
                weights = self.cswnet(current_pred.unsqueeze(-1), current_error, current_pattern.unsqueeze(-1), K = self.pattern_num)
                pred_error_pattern = torch.cat([current_pred.unsqueeze(-1), current_error, current_pattern.unsqueeze(-1)], dim=2)
                 
                weights_avg = torch.mean(weights, dim=1)  # [B, 1+K+1]
                weights_avg = weights_avg.unsqueeze(1).expand(-1, self.pred_len, -1)  # 沿着时间维度求均值——因为权重的优化是一个和时间无关的过程
                
                # 计算重构信号
                channel_reconstructed = (weights_avg * pred_error_pattern).sum(dim=2)  # 沿着通道维度求和
                weights = torch.unique(weights_avg, dim=1) # 保留时间维度上的不重复样本
                weights_list.append(weights.squeeze(1)) #去掉第二个维度，得到形状[B, 1+K+1]
                pred_error_pattern_list.append(pred_error_pattern.squeeze(1))
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1)) 
                
                # 将每个通道的中间拼接起来，得到 [B, L, enc_in]
                y = torch.cat(reconstructed_signal_list, dim=2)
                weights=torch.cat(weights_list, dim=1)
                patterns = torch.cat(pattern_list, dim=2)
                error=torch.cat(current_error_list, dim=2)
                pred_error_perio=torch.cat(pred_error_pattern_list, dim=2)
            
            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean

        elif self.cycle_pattern == 'daily+weekly+monthly+yearly':
            pattern_index_daily = x_mark[..., 0] * self.pattern_len * 366
            pattern_index_daily = pattern_index_daily[:, -1]
            pattern_index_weekly = x_mark[..., 1] * 7  # 每周周期
            pattern_index_weekly = pattern_index_weekly[:, -1]
            pattern_index_monthly = x_mark[..., 2] * 31  # 每月周期
            pattern_index_monthly = pattern_index_monthly[:, -1]
            pattern_index_yearly = x_mark[..., 3] * 366  # 每年周期
            pattern_index_yearly = pattern_index_yearly[:, -1]
            pattern_index = pattern_index_daily + pattern_index_weekly + pattern_index_monthly + pattern_index_yearly # 合并每日，每周，每月和每年周期

            # instance norm
            if self.use_revin:
                seq_mean = torch.mean(x, dim=1, keepdim=True)
                seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
                x = (x - seq_mean) / torch.sqrt(seq_var)
        
            pred = self.patch_tst(x.permute(0, 2, 1)).permute(0, 2, 1)
            x1 = x - self.patternQueue(pattern_index_daily, self.seq_len) # remove the pattern of the input data
            y1 = self.patch_tst(x1.permute(0, 2, 1)).permute(0, 2, 1)
            x2 = x - self.patternQueue(pattern_index_weekly, self.seq_len)
            y2 = self.patch_tst(x2.permute(0, 2, 1)).permute(0, 2, 1)
            x3 = x - self.patternQueue(pattern_index_monthly, self.seq_len) # remove the pattern of the input data
            y3 = self.patch_tst(x3.permute(0, 2, 1)).permute(0, 2, 1)
            x4 = x - self.patternQueue(pattern_index_yearly, self.seq_len) # remove the pattern of the input data
            y4 = self.patch_tst(x4.permute(0, 2, 1)).permute(0, 2, 1)

            # Save patternQueue outputs as intermediate variables
            pattern_daily = self.patternQueue(pattern_index_daily, self.seq_len)
            pattern_weekly = self.patternQueue(pattern_index_weekly, self.seq_len)
            pattern_monthly = self.patternQueue(pattern_index_monthly, self.seq_len)
            pattern_yearly = self.patternQueue(pattern_index_yearly, self.seq_len)

            pattern = self.patternQueue((pattern_index + self.seq_len) % self.pattern_len, self.pred_len)

            # Initialize lists to store intermediate variables
            reconstructed_signal_list = []
            weights_list = []
            current_error_list = []
            pred_error_pattern_list = []
            pattern_list = []  # List to store the intermediate pattern values
            
            # 遍历每个通道
            for i in range(x.shape[2]):
            # 获取当前通道的pred y1, y2, y3，y4和 error
                current_pred = pred[:, :, i]
                error = torch.stack([x1[:, :, i], x2[:, :, i], x3[:, :, i], x4[:, :, i]], dim=2)
                current_error = torch.stack([y1[:, :, i], y2[:, :, i], y3[:, :, i], y4[:, :, i]], dim=2)# 形状 [B, L, K]
                current_pattern = pattern[:, :, i] # 形状 [B, L]
                
                current_error_list.append(error)
                # Append patterns to the pattern_list
                pattern_list.append(torch.stack([pattern_daily[:, :, i], pattern_weekly[:, :, i], pattern_monthly[:, :, i], pattern_yearly[:, :, i]], dim=2))
                
                # 通过 CWS 计算当前通道的重构信号和权重
                weights = self.cswnet(current_pred.unsqueeze(-1), current_error, current_pattern.unsqueeze(-1), K = self.pattern_num)
                pred_error_pattern = torch.cat([current_pred.unsqueeze(-1), current_error, current_pattern.unsqueeze(-1)], dim=2)
                 
                weights_avg = torch.mean(weights, dim=1)  # [B, 1+K+1]
                weights_avg = weights_avg.unsqueeze(1).expand(-1, self.pred_len, -1)  # 沿着时间维度求均值——因为权重的优化是一个和时间无关的过程
                
                # 计算重构信号
                channel_reconstructed = (weights_avg * pred_error_pattern).sum(dim=2)  # 沿着通道维度求和
                weights = torch.unique(weights_avg, dim=1) # 保留时间维度上的不重复样本
                weights_list.append(weights.squeeze(1)) #去掉第二个维度，得到形状[B, 1+K+1]
                pred_error_pattern_list.append(pred_error_pattern)
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1)) 

                # 将每个通道的中间拼接起来，得到 [B, L, enc_in]
                y = torch.cat(reconstructed_signal_list, dim=2)
                weights=torch.cat(weights_list, dim=1)
                patterns = torch.cat(pattern_list, dim=2)
                error=torch.cat(current_error_list, dim=2)
                pred_error_perio=torch.cat(pred_error_pattern_list, dim=2)
            
            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean

        else:
            raise Exception("please specify timestamps_related_pattern")

        return y, weights, patterns, error, pred_error_perio
