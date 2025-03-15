import torch
import torch.nn as nn

from argparse import Namespace
import torch.nn.functional as F

# from baselines.CycleNet.arch.DRPK_net import DRNet
class DRNet(nn.Module):
    def __init__(self, hidden_dim, K):
        super(DRPKNet, self).__init__()
        # 根据K调整网络结构
        # 计算拼接后的输入特征维度：imfs + error => 1 + K + 1
        input_dim = K + 1  #imfs(K), error(1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  
        self.relu = nn.ReLU()  # 从x个信号到x个权重

        # 输出的维度为: K+2，其中K是imfs的数量，+2是包括pred和error的两个额外特征
        self.fc2 = nn.Linear(hidden_dim, K+1)  # K+2，考虑pred和error的维度

    def forward(self,imfs,error,K):
        # 将输入信号拼接
        error = error.unsqueeze(-1)
        x = torch.cat([imfs, error], dim=2)
        # 通过网络获取权重
        x = self.fc1(x.permute(1,0,2))
        x = x.view(-1, x.shape[2])  # 将 x 变为 [B * L, 3]
        x = self.bn1(x)
        x = self.relu(x)
        weights = self.fc2(x).view(-1, imfs.shape[1],  imfs.shape[2]+1)
        error_weights = torch.sigmoid(weights[:, :, -1]) ## error单独处理
        other_weights = F.softmax(weights[:, :, :K], dim = 2) * K 
        weights = torch.cat((other_weights, error_weights.unsqueeze(-1)), dim=2)     
        # 计算重构信号
        # 初始化一个列表来存储每个通道的加权信号
        reconstructed_channels = []

        # 遍历每个通道
        for i in range(imfs.shape[2]):
            imf = imfs[:, :, i]  # 取出第 i 个通道的所有样本和时间步，形状为 [B, L]
    
            # 计算加权信号
            weighted_imf = weights[:, :, i+1] * imf
    
            # 将每个加权信号添加到 reconstructed_channels 列表中
            reconstructed_channels.append(weighted_imf.unsqueeze(-1))  # 在最后一维添加一个维度，以便拼接

        # 拼接所有通道，得到形状 [B, L, C]，其中 C 为通道数
        reconstructed_signal = torch.cat(reconstructed_channels, dim=2)

        return reconstructed_signal, weights

class DRPKNet(nn.Module):
    def __init__(self, hidden_dim, K):
        super(DRPKNet, self).__init__()
        # 根据K调整网络结构
        # 计算拼接后的输入特征维度：pred + imfs + error => 1 + K + 1
        input_dim = 1 + K + 1  # pred(1), imfs(K), error(1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  
        self.relu = nn.ReLU()  # 从x个信号到x个权重
        
        # 输出的维度为: K+2，其中K是imfs的数量，+2是包括pred和error的两个额外特征
        self.fc2 = nn.Linear(hidden_dim, K+2)  # K+2，考虑pred和error的维度

    def forward(self, pred, imfs, error, K):
        # 将输入信号拼接
        pred = pred.unsqueeze(-1)
        error = error.unsqueeze(-1)
        x = torch.cat([pred, imfs, error], dim=2)
        
        # 通过网络获取权重
        x = self.fc1(x.permute(1, 0, 2))
        x = x.view(-1, x.shape[2])  # 将 x 变为 [B * L, 3]
        x = self.bn1(x)
        x = self.relu(x)
        weights = self.fc2(x).view(-1, imfs.shape[1],  K+2)
        
        # 计算每个通道的权重
        pred_weights = torch.sigmoid(weights[:, :, 0])  # pred的权重
        error_weights = torch.sigmoid(weights[:, :, -1])  # error的权重
        other_weights = F.softmax(weights[:, :, :K], dim=2) * K  # imf的权重

        # 合并所有权重
        weights = torch.cat((pred_weights.unsqueeze(-1), other_weights, error_weights.unsqueeze(-1)), dim=2)     

        # 计算重构信号
        reconstructed_channels = []

        # 遍历每个通道
        for i in range(imfs.shape[2]):
            imf = imfs[:, :, i]  # 取出第 i 个通道的所有样本和时间步，形状为 [B, L]
    
            # 计算加权信号
            weighted_imf = weights[:, :, i+1] * imf
    
            # 将每个加权信号添加到 reconstructed_channels 列表中
            reconstructed_channels.append(weighted_imf.unsqueeze(-1))  # 在最后一维添加一个维度，以便拼接

        # 拼接所有通道，得到形状 [B, L, C]，其中 C 为通道数
        reconstructed_signal = torch.cat(reconstructed_channels, dim=2)

        # 添加先验部分
        reconstructed_signal = torch.cat([reconstructed_signal, weights[:, :, 0].unsqueeze(-1) * pred], dim=2)

        # 添加误差部分
        reconstructed_signal = torch.cat([reconstructed_signal, weights[:, :, -1].unsqueeze(-1) * error], dim=2)

        return reconstructed_signal, weights


class RecurrentCycle(torch.nn.Module):
    """
    From the author:
    # Thanks for the contribution of wayhoww.
    # The new implementation uses index arithmetic with modulo to directly gather cyclic data in a single operation,
    # while the original implementation manually rolls and repeats the data through looping.
    # It achieves a significant speed improvement (2x ~ 3x acceleration).
    # See https://github.com/ACAT-SCUT/CycleNet/pull/4 for more details.
    """
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len    
        return self.data[gather_index.long()]


class Model(nn.Module):
    """
        Paper: CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns
        Link: https://arxiv.org/pdf/2409.18479
        Official Code: https://github.com/ACAT-SCUT/CycleNet
        Venue:  NIPS 2024
        Task: Long-term Time Series Forecasting
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin
        self.cycle_pattern = configs.cycle_pattern
        self.pattern_nums = configs.pattern_nums
        self.drn = DRPKNet(hidden_dim = 64, K=self.pattern_nums)  

        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

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
            cycle_index_daily = x_mark[..., 1] * self.cycle_len  # 每日周期
            cycle_index = cycle_index_daily[:, -1]
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
            error = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

            # 初始化一个空的列表来保存每个通道的重构信号
            reconstructed_signal_list = []

            # 遍历每个通道
            for i in range(y1.shape[2]):
            # 获取当前通道的pred y1, y2, y3 和 error
                current_pred = pred[:, :, i]
                imfs = y1[:, :, i]
                current_error = error[:, :, i]  # 形状 [B, L]
        
                # 通过 DRNet 计算当前通道的重构信号和权重
                reconstructed_signal, weights = self.drn(current_pred, imfs, current_error, imfs.shape[2])
                pred_imfs_error = torch.cat([current_pred.unsqueeze(-1), imfs, current_error.unsqueeze(-1)], dim=2)  # 形状 [256, 336, 3]

                # 计算重构信号
                channel_reconstructed = (weights * pred_imfs_error).sum(dim=2)  # 沿着通道维度求和

                # 将每个通道的重构信号添加到列表中
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1))  # 形状为 [B, L, 1]

                # 将每个通道的重构信号拼接起来，得到 [B, L, 9]
                reconstructed_signal = torch.cat(reconstructed_signal_list, dim=2)

                # add back the cycle of the output data
                    # y = y1 + y2 + error
                y = reconstructed_signal
            
            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean

        
        elif self.cycle_pattern == 'daily+weekly':
            cycle_index_daily = x_mark[..., 1] * self.cycle_len * 7
            cycle_index_daily = cycle_index_daily[:, -1]
            cycle_index_weekly = x_mark[..., 2] * 7  # 每周周期
            cycle_index_weekly = cycle_index_weekly[:, -1]
            cycle_index = cycle_index_daily + cycle_index_weekly  # 合并每日和每周周期

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

            # remove the cycle of the input data
            x2 = x - self.cycleQueue(cycle_index_weekly, self.seq_len)

            # forecasting with channel independence (parameters-sharing)
            y2 = self.model(x2.permute(0, 2, 1)).permute(0, 2, 1)

            error = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

            # 初始化一个空的列表来保存每个通道的重构信号
            reconstructed_signal_list = []

            # 遍历每个通道
            for i in range(y1.shape[2]):
            # 获取当前通道的pred y1, y2, y3 和 error
                current_pred = pred[:, :, i]
                imfs = torch.stack([y1[:, :, i], y2[:, :, i]], dim=2)  # 形状 [B, L, 2]
                current_error = error[:, :, i]  # 形状 [B, L]
        
                # 通过 DRNet 计算当前通道的重构信号和权重
                reconstructed_signal, weights = self.drn(current_pred, imfs, current_error, imfs.shape[2])
                pred_imfs_error = torch.cat([current_pred.unsqueeze(-1), imfs, current_error.unsqueeze(-1)], dim=2)  # 形状 [256, 336, 3]

                # 计算重构信号
                channel_reconstructed = (weights * pred_imfs_error).sum(dim=2)  # 沿着通道维度求和

                # 将每个通道的重构信号添加到列表中
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1))  # 形状为 [B, L, 1]

                # 将每个通道的重构信号拼接起来，得到 [B, L, 9]
                reconstructed_signal = torch.cat(reconstructed_signal_list, dim=2)

                # add back the cycle of the output data
                    # y = y1 + y2 + error
                y = reconstructed_signal
            
            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean

        elif self.cycle_pattern == 'daily+weekly+monthly':
            cycle_index_daily = x_mark[..., 1] * self.cycle_len * 7  # 每日周期
            cycle_index_weekly = x_mark[..., 2] * 7  # 每周周期
            cycle_index_monthly = x_mark[..., 3] * self.cycle_len * 30  # 每月周期，假设每月30天
            cycle_index_daily = cycle_index_daily[:, -1]
            cycle_index_weekly = cycle_index_weekly[:, -1]
            cycle_index_monthly = cycle_index_monthly[:, -1]
            cycle_index = cycle_index_daily + cycle_index_weekly + cycle_index_monthly  # 合并每日、每周和每月周期

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

            # remove the cycle of the input data
            x2 = x - self.cycleQueue(cycle_index_weekly, self.seq_len)

            # forecasting with channel independence (parameters-sharing)
            y2 = self.model(x2.permute(0, 2, 1)).permute(0, 2, 1)

            x3 = x - self.cycleQueue(cycle_index_monthly, self.seq_len)
        
            # forecasting with channel independence (parameters-sharing)
            y3 = self.model(x3.permute(0, 2, 1)).permute(0, 2, 1)

            error = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

            # 初始化一个空的列表来保存每个通道的重构信号
            reconstructed_signal_list = []

            # 遍历每个通道
            for i in range(y1.shape[2]):
            # 获取当前通道的pred y1, y2, y3 和 error
                current_pred = pred[:, :, i]
                imfs = torch.stack([y1[:, :, i], y2[:, :, i], y3[:, :, i]], dim=2)  # 形状 [B, L, 2]
                current_error = error[:, :, i]  # 形状 [B, L]
        
                # 通过 DRNet 计算当前通道的重构信号和权重
                reconstructed_signal, weights = self.drn(current_pred, imfs, current_error, imfs.shape[2])
                pred_imfs_error = torch.cat([current_pred.unsqueeze(-1), imfs, current_error.unsqueeze(-1)], dim=2)  # 形状 [256, 336, 3]

                # 计算重构信号
                channel_reconstructed = (weights * pred_imfs_error).sum(dim=2)  # 沿着通道维度求和

                # 将每个通道的重构信号添加到列表中
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1))  # 形状为 [B, L, 1]

                # 将每个通道的重构信号拼接起来，得到 [B, L, 9]
                reconstructed_signal = torch.cat(reconstructed_signal_list, dim=2)

                # add back the cycle of the output data
                    # y = y1 + y2 + error
                y = reconstructed_signal
            
            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean

        elif self.cycle_pattern == 'daily+weekly+monthly+yearly':
            # cycle_index_hourly = x_mark[..., 0] * self.cycle_len  # 每小时周期
            cycle_index_daily = x_mark[..., 1] * self.cycle_len * 7  # 每日周期
            cycle_index_weekly = x_mark[..., 2] * 7  # 每周周期
            cycle_index_monthly = x_mark[..., 3] * self.cycle_len * 30  # 每月周期，假设每月30天
            cycle_index_yearly = x_mark[..., 4] * self.cycle_len * 366  # 每年周期，假设每年366天
            # cycle_index_hourly =cycle_index_hourly[:, -1]
            cycle_index_daily = cycle_index_daily[:, -1]
            cycle_index_weekly = cycle_index_weekly[:, -1]
            cycle_index_monthly = cycle_index_monthly[:, -1]
            cycle_index_yearly = cycle_index_yearly[:, -1]
            cycle_index = cycle_index_daily + cycle_index_weekly + cycle_index_monthly + cycle_index_yearly  # 合并每日、每周、每月和每年周期

            # instance norm
            if self.use_revin:
                seq_mean = torch.mean(x, dim=1, keepdim=True)
                seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
                x = (x - seq_mean) / torch.sqrt(seq_var)
        
            pred = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)
            # remove the cycle of the input data
            # x0 = x - self.cycleQueue(cycle_index_hourly, self.seq_len)

            # forecasting with channel independence (parameters-sharing)
            # y0 = self.model(x0.permute(0, 2, 1)).permute(0, 2, 1)

            # remove the cycle of the input data
            x1 = x - self.cycleQueue(cycle_index_daily, self.seq_len)

            # forecasting with channel independence (parameters-sharing)
            y1 = self.model(x1.permute(0, 2, 1)).permute(0, 2, 1)

            # remove the cycle of the input data
            x2 = x - self.cycleQueue(cycle_index_weekly, self.seq_len)

            # forecasting with channel independence (parameters-sharing)
            y2 = self.model(x2.permute(0, 2, 1)).permute(0, 2, 1)

            x3 = x - self.cycleQueue(cycle_index_monthly, self.seq_len)
        
            # forecasting with channel independence (parameters-sharing)
            y3 = self.model(x3.permute(0, 2, 1)).permute(0, 2, 1)

            x4 = x - self.cycleQueue(cycle_index_yearly, self.seq_len)
        
            # forecasting with channel independence (parameters-sharing)
            y4 = self.model(x4.permute(0, 2, 1)).permute(0, 2, 1)

            error = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

            # 初始化一个空的列表来保存每个通道的重构信号
            reconstructed_signal_list = []

            # 遍历每个通道
            for i in range(y1.shape[2]):
            # 获取当前通道的pred y1, y2, y3 和 error
                current_pred = pred[:, :, i]
                imfs = torch.stack([y1[:, :, i], y2[:, :, i], y3[:, :, i], y4[:, :, i]], dim=2)  # 形状 [B, L, 2]
                current_error = error[:, :, i]  # 形状 [B, L]
        
                # 通过 DRNet 计算当前通道的重构信号和权重
                reconstructed_signal, weights = self.drn(current_pred, imfs, current_error, imfs.shape[2])
                pred_imfs_error = torch.cat([current_pred.unsqueeze(-1), imfs, current_error.unsqueeze(-1)], dim=2)  # 形状 [256, 336, 3]

                # 计算重构信号
                channel_reconstructed = (weights * pred_imfs_error).sum(dim=2)  # 沿着通道维度求和

                # 将每个通道的重构信号添加到列表中
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1))  # 形状为 [B, L, 1]

                # 将每个通道的重构信号拼接起来，得到 [B, L, 9]
                reconstructed_signal = torch.cat(reconstructed_signal_list, dim=2)

                # add back the cycle of the output data
                    # y = y1 + y2 + error
                y = reconstructed_signal
            
            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean
        
        elif self.cycle_pattern == 'hourly+daily+weekly+monthly+yearly':
            cycle_index_hourly = x_mark[..., 0] * self.cycle_len  # 每小时周期
            cycle_index_daily = x_mark[..., 1] * self.cycle_len * 7  # 每日周期
            cycle_index_weekly = x_mark[..., 2] * 7  # 每周周期
            cycle_index_monthly = x_mark[..., 3] * self.cycle_len * 30  # 每月周期，假设每月30天
            cycle_index_yearly = x_mark[..., 4] * self.cycle_len * 366  # 每年周期，假设每年366天
            cycle_index_hourly =cycle_index_hourly[:, -1]
            cycle_index_daily = cycle_index_daily[:, -1]
            cycle_index_weekly = cycle_index_weekly[:, -1]
            cycle_index_monthly = cycle_index_monthly[:, -1]
            cycle_index_yearly = cycle_index_yearly[:, -1]
            cycle_index = cycle_index_hourly + cycle_index_daily + cycle_index_weekly + cycle_index_monthly + cycle_index_yearly  # 合并每日、每周、每月和每年周期

            # instance norm
            if self.use_revin:
                seq_mean = torch.mean(x, dim=1, keepdim=True)
                seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
                x = (x - seq_mean) / torch.sqrt(seq_var)
        
            pred = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)
            # remove the cycle of the input data
            x0 = x - self.cycleQueue(cycle_index_hourly, self.seq_len)

            # forecasting with channel independence (parameters-sharing)
            y0 = self.model(x0.permute(0, 2, 1)).permute(0, 2, 1)

            # remove the cycle of the input data
            x1 = x - self.cycleQueue(cycle_index_daily, self.seq_len)

            # forecasting with channel independence (parameters-sharing)
            y1 = self.model(x1.permute(0, 2, 1)).permute(0, 2, 1)

            # remove the cycle of the input data
            x2 = x - self.cycleQueue(cycle_index_weekly, self.seq_len)

            # forecasting with channel independence (parameters-sharing)
            y2 = self.model(x2.permute(0, 2, 1)).permute(0, 2, 1)

            x3 = x - self.cycleQueue(cycle_index_monthly, self.seq_len)
        
            # forecasting with channel independence (parameters-sharing)
            y3 = self.model(x3.permute(0, 2, 1)).permute(0, 2, 1)

            x4 = x - self.cycleQueue(cycle_index_yearly, self.seq_len)
        
            # forecasting with channel independence (parameters-sharing)
            y4 = self.model(x4.permute(0, 2, 1)).permute(0, 2, 1)

            error = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

            # 初始化一个空的列表来保存每个通道的重构信号
            reconstructed_signal_list = []

            # 遍历每个通道
            for i in range(y1.shape[2]):
            # 获取当前通道的pred y1, y2, y3 和 error
                current_pred = pred[:, :, i]
                imfs = torch.stack([y0[:, :, i], y1[:, :, i], y2[:, :, i], y3[:, :, i], y4[:, :, i]], dim=2)  # 形状 [B, L, 2]
                current_error = error[:, :, i]  # 形状 [B, L]
        
                # 通过 DRNet 计算当前通道的重构信号和权重
                reconstructed_signal, weights = self.drn(current_pred, imfs, current_error, imfs.shape[2])
                pred_imfs_error = torch.cat([current_pred.unsqueeze(-1), imfs, current_error.unsqueeze(-1)], dim=2)  # 形状 [256, 336, 3]

                # 计算重构信号
                channel_reconstructed = (weights * pred_imfs_error).sum(dim=2)  # 沿着通道维度求和

                # 将每个通道的重构信号添加到列表中
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1))  # 形状为 [B, L, 1]

                # 将每个通道的重构信号拼接起来，得到 [B, L, 9]
                reconstructed_signal = torch.cat(reconstructed_signal_list, dim=2)

                # add back the cycle of the output data
                    # y = y1 + y2 + error
                y = reconstructed_signal
            
            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean
        
        else:
            raise Exception("please specify cycle pattern, daily OR weekly OR monthly OR daily&weekly&monthly OR daily&weekly&monthly&yearly")

        return y
