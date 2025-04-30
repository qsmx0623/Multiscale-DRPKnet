import torch
import torch.nn as nn
from layers import WCC_Combiner


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
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.pattern_len = configs.pattern
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin
        self.cycle_pattern = configs.cycle_pattern
        self.pattern_num = configs.pattern_nums
        self.hidden_dim = configs.batch_size
        self.wccnet = WCC_Combiner.WCC_no_pk(hidden_dim=self.hidden_dim, K=self.pattern_num)

        self.patternQueue = RecurrentPattern(pattern_len=self.pattern_len, channel_size=self.enc_in)

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
        if self.cycle_pattern == 'daily':
            pattern_index_daily = x_mark[..., 0] * self.pattern_len  # 每日周期
            pattern_index_daily = pattern_index_daily[:, -1]

            # instance norm
            if self.use_revin:
                seq_mean = torch.mean(x, dim=1, keepdim=True)
                seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
                x = (x - seq_mean) / torch.sqrt(seq_var)

            # 获取 y1 和 error
            x1 = x - self.patternQueue(pattern_index_daily, self.seq_len)  # 去除输入中的模式
            y1 = self.model(x1.permute(0, 2, 1)).permute(0, 2, 1)

            pattern_daily = self.patternQueue(pattern_index_daily, self.seq_len)
            pattern = self.patternQueue((pattern_index_daily + self.seq_len) % self.pattern_len, self.pred_len)

            # 初始化空的列表来保存每个通道的中间变量
            reconstructed_signal_list = []
            weights_list = []
            current_error_list = []
            error_pattern_list = []
            pattern_list = []

            # 遍历每个通道
            for i in range(x.shape[2]):
                error = x1[:, :, i]  # 计算当前通道的 error
                current_error = y1[:, :, i] # 形状 [B, L]
                current_pattern = pattern[:, :, i]  # 计算当前通道的 pattern

                current_error_list.append(error)
                pattern_list.append(pattern_daily[:, :, i])

                # 通过 WCC 计算当前通道的重构信号和权重
                # weights = self.wccnet(current_error.unsqueeze(-1), current_pattern.unsqueeze(-1), K=self.pattern_num)
                weights = torch.full((current_error.size(0), current_error.size(1), self.pattern_num + 1), 1 / (self.pattern_num + 1))
                error_pattern = torch.cat([current_error.unsqueeze(-1), current_pattern.unsqueeze(-1)], dim=2)

                weights_avg = torch.mean(weights, dim=1)  # [B, 3]
                weights_avg = weights_avg.unsqueeze(1).expand(-1, self.pred_len, -1)  # 沿着时间维度求均值

                # 计算重构信号
                device = error_pattern.device  # 获取 weights_avg 的设备，确保所有张量都在相同的设备上
                weights_avg = weights_avg.to(device)
                channel_reconstructed = (weights_avg * error_pattern).sum(dim=2)  # 沿着通道维度求和
                weights = torch.unique(weights_avg, dim=1)  # 保留时间维度上的不重复样本
                weights_list.append(weights.squeeze(1))  # 去掉第二个维度，得到形状 [B, 1+K+1]
                error_pattern_list.append(error_pattern.squeeze(1))
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1))

            # 将每个通道的中间拼接起来，得到 [B, L, enc_in]
            y = torch.cat(reconstructed_signal_list, dim=2)
            weights = torch.cat(weights_list, dim=1)
            patterns = pattern_list
            error = current_error_list
            error_perio = torch.cat(error_pattern_list, dim=2)

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
        
            x1 = x - self.patternQueue(pattern_index_daily, self.seq_len) # remove the pattern of the input data
            y1 = self.model(x1.permute(0, 2, 1)).permute(0, 2, 1)
            x2 = x - self.patternQueue(pattern_index_weekly, self.seq_len)
            y2 = self.model(x2.permute(0, 2, 1)).permute(0, 2, 1)

            # Save patternQueue outputs as intermediate variables
            pattern_daily = self.patternQueue(pattern_index_daily, self.seq_len)
            pattern_weekly = self.patternQueue(pattern_index_weekly, self.seq_len)

            pattern = self.patternQueue((pattern_index + self.seq_len) % self.pattern_len, self.pred_len)
            
            # 初始化空的列表来保存每个通道的中间变量
            reconstructed_signal_list = []
            weights_list = []
            current_error_list =[]
            error_pattern_list = []
            pattern_list = []  # List to store the intermediate pattern values

            # 遍历每个通道
            for i in range(x.shape[2]):
            # 获取当前通道的pred y1, y2 和 error
                error = torch.stack([x1[:, :, i], x2[:, :, i]], dim=2)
                current_error = torch.stack([y1[:, :, i], y2[:, :, i]], dim=2)# 形状 [B, L, K]
                current_pattern = pattern[:, :, i] # 形状 [B, L]
        
                current_error_list.append(error)
                # Append patterns to the pattern_list
                pattern_list.append(torch.stack([pattern_daily[:, :, i], pattern_weekly[:, :, i]], dim=2))

                # 通过 WCC 计算当前通道的重构信号和权重
                # weights = self.wccnet(current_error, current_pattern.unsqueeze(-1), K = self.pattern_num)
                weights = torch.full((current_error.size(0), current_error.size(1), self.pattern_num +1 ), 1 / (self.pattern_num + 1))
                error_pattern = torch.cat([current_error, current_pattern.unsqueeze(-1)], dim=2)
                 
                weights_avg = torch.mean(weights, dim=1)  # [B, 1+K+1]
                weights_avg = weights_avg.unsqueeze(1).expand(-1, self.pred_len, -1)  # 沿着时间维度求均值——因为权重的优化是一个和时间无关的过程
                
                # 计算重构信号
                device = error_pattern.device  # 获取 weights_avg 的设备，确保所有张量都在相同的设备上
                weights_avg = weights_avg.to(device)
                channel_reconstructed = (weights_avg * error_pattern).sum(dim=2)  # 沿着通道维度求和
                weights = torch.unique(weights_avg, dim=1) # 保留时间维度上的不重复样本
                weights_list.append(weights.squeeze(1)) #去掉第二个维度，得到形状[B, 1+K+1]
                error_pattern_list.append(error_pattern.squeeze(1))
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1)) 

                # 将每个通道的中间拼接起来，得到 [B, L, enc_in]
                y = torch.cat(reconstructed_signal_list, dim=2)
                weights=torch.cat(weights_list, dim=1)
                patterns = torch.cat(pattern_list, dim=2)
                error=torch.cat(current_error_list, dim=2)
                error_perio=torch.cat(error_pattern_list, dim=2)
            
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
        
            x1 = x - self.patternQueue(pattern_index_daily, self.seq_len) # remove the pattern of the input data
            y1 = self.model(x1.permute(0, 2, 1)).permute(0, 2, 1)
            x4 = x - self.patternQueue(pattern_index_yearly, self.seq_len) # remove the pattern of the input data
            y4 = self.model(x4.permute(0, 2, 1)).permute(0, 2, 1)

            pattern_daily = self.patternQueue(pattern_index_daily, self.seq_len)
            pattern_yearly = self.patternQueue(pattern_index_yearly, self.seq_len)

            pattern = self.patternQueue((pattern_index + self.seq_len) % self.pattern_len, self.pred_len)
            
            # 初始化三个空的列表来保存每个通道的中间变量
            reconstructed_signal_list = []
            weights_list = []
            current_error_list =[]
            error_pattern_list = []
            pattern_list = []  # List to store the intermediate pattern values

            # print(y1.shape)
            # 遍历每个通道
            for i in range(x.shape[2]):
            # 获取当前通道的pred y1, y4 和 error
                error = torch.stack([x1[:, :, i], x4[:, :, i]], dim=2)
                current_error = torch.stack([y1[:, :, i], y4[:, :, i]], dim=2)# 形状 [B, L, K]
                current_pattern = pattern[:, :, i] # 形状 [B, L]
        
                current_error_list.append(error)
                pattern_list.append(torch.stack([pattern_daily[:, :, i], pattern_yearly[:, :, i]], dim=2))
                
                # 通过 WCC 计算当前通道的重构信号和权重
                # weights = self.wccnet(current_error, current_pattern.unsqueeze(-1), K = self.pattern_num)
                weights = torch.full((current_error.size(0), current_error.size(1), self.pattern_num + 1), 1 / (self.pattern_num + 1))
                error_pattern = torch.cat([current_error, current_pattern.unsqueeze(-1)], dim=2)
                 
                weights_avg = torch.mean(weights, dim=1)  # [B, 1+K+1]
                weights_avg = weights_avg.unsqueeze(1).expand(-1, self.pred_len, -1)  # 沿着时间维度求均值——因为权重的优化是一个和时间无关的过程
                
                # 计算重构信号
                device = error_pattern.device  # 获取 weights_avg 的设备，确保所有张量都在相同的设备上
                weights_avg = weights_avg.to(device)
                channel_reconstructed = (weights_avg * error_pattern).sum(dim=2)  # 沿着通道维度求和
                weights = torch.unique(weights_avg, dim=1) # 保留时间维度上的不重复样本
                weights_list.append(weights.squeeze(1)) #去掉第二个维度，得到形状[B, 1+K+1]
                error_pattern_list.append(error_pattern.squeeze(1))
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1)) 
                
                # 将每个通道的中间拼接起来，得到 [B, L, enc_in]
                y = torch.cat(reconstructed_signal_list, dim=2)
                weights=torch.cat(weights_list, dim=1)
                patterns = torch.cat(pattern_list, dim=2)
                error=torch.cat(current_error_list, dim=2)
                error_perio=torch.cat(error_pattern_list, dim=2)
            
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
        
            x1 = x - self.patternQueue(pattern_index_daily, self.seq_len) # remove the pattern of the input data
            y1 = self.model(x1.permute(0, 2, 1)).permute(0, 2, 1)
            x2 = x - self.patternQueue(pattern_index_weekly, self.seq_len)
            y2 = self.model(x2.permute(0, 2, 1)).permute(0, 2, 1)
            x4 = x - self.patternQueue(pattern_index_yearly, self.seq_len) # remove the pattern of the input data
            y4 = self.model(x4.permute(0, 2, 1)).permute(0, 2, 1)

            # Save patternQueue outputs as intermediate variables
            pattern_daily = self.patternQueue(pattern_index_daily, self.seq_len)
            pattern_weekly = self.patternQueue(pattern_index_weekly, self.seq_len)
            pattern_yearly = self.patternQueue(pattern_index_yearly, self.seq_len)

            pattern = self.patternQueue((pattern_index + self.seq_len) % self.pattern_len, self.pred_len)
            
            # 初始化三个空的列表来保存每个通道的中间变量
            reconstructed_signal_list = []
            weights_list = []
            current_error_list =[]
            error_pattern_list = []
            pattern_list = [] 

            # print(y1.shape)
            # 遍历每个通道
            for i in range(x.shape[2]):
            # 获取当前通道的pred y1, y2, y4 和 error
                error = torch.stack([x1[:, :, i], x2[:, :, i], x4[:, :, i]], dim=2)#
                current_error = torch.stack([y1[:, :, i], y2[:, :, i], y4[:, :, i]], dim=2)# 形状 [B, L, K]
                current_pattern = pattern[:, :, i] # 形状 [B, L]
        
                current_error_list.append(error)
                pattern_list.append(torch.stack([pattern_daily[:, :, i], pattern_weekly[:, :, i], pattern_yearly[:, :, i]], dim=2))

                # 通过 WCC 计算当前通道的重构信号和权重
                # weights = self.wccnet(current_error, current_pattern.unsqueeze(-1), K = self.pattern_num)
                weights = torch.full((current_error.size(0), current_error.size(1), self.pattern_num + 1), 1 / (self.pattern_num + 1))
                error_pattern = torch.cat([current_error, current_pattern.unsqueeze(-1)], dim=2)
                 
                weights_avg = torch.mean(weights, dim=1)  # [B, 1+K+1]
                weights_avg = weights_avg.unsqueeze(1).expand(-1, self.pred_len, -1)  # 沿着时间维度求均值——因为权重的优化是一个和时间无关的过程
                
                # 计算重构信号
                device = error_pattern.device  # 获取 weights_avg 的设备，确保所有张量都在相同的设备上
                weights_avg = weights_avg.to(device)
                channel_reconstructed = (weights_avg * error_pattern).sum(dim=2)  # 沿着通道维度求和
                weights = torch.unique(weights_avg, dim=1) # 保留时间维度上的不重复样本
                weights_list.append(weights) 
                error_pattern_list.append(error_pattern.unsqueeze(-1))
                reconstructed_signal_list.append(channel_reconstructed.unsqueeze(-1)) 

                # 将每个通道的中间拼接起来，得到 [B, L, enc_in]
                y = torch.cat(reconstructed_signal_list, dim=2)
                weights=torch.cat(weights_list, dim=1)
                patterns = torch.cat(pattern_list, dim=2)
                error=torch.cat(current_error_list, dim=2)
                error_perio=torch.cat(error_pattern_list, dim=2)

            # instance denorm
            if self.use_revin:
                y = y * torch.sqrt(seq_var) + seq_mean

        return y, weights, patterns, error, error_perio
