import torch
import torch.nn as nn
import torch.nn.functional as F

class WCCNet(nn.Module):
    def __init__(self, hidden_dim, K):
        super(WCCNet, self).__init__()
        # 根据K调整网络结构
        # 计算拼接后的输入特征维度：pred + cycle + error => 1 + K+1
        input_dim = out_dim = 1+K+1  # pred(1), error(K), cycle(Q)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  
        self.relu = nn.ReLU()  # 从x个信号到x个权重
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, pred, error, basic_pattern, K):
        # 将输入信号拼接
        x = torch.cat([pred, error, basic_pattern], dim=2)
        # 通过网络获取权重
        x = self.fc1(x.permute(1, 0, 2))
        x = x.view(-1, x.shape[2])  # 将 x 变为 [B * L, 1+K+1]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        weights = x.view(-1, basic_pattern.shape[1], 1+K+1)

        # 计算每个通道的权重
        pred_weights = torch.sigmoid(weights[:, :, 0])  # pred的权重
        other_weights = F.softmax(weights[:, :, 1:K+1], dim=-1) * K 
        basic_pattern_weight = torch.sigmoid(weights[:, :, -1]) # basic_pattern的权重
        # weights = F.softmax(weights, dim=-1)

        # 合并所有权重
        weights = torch.cat((pred_weights.unsqueeze(-1), other_weights, basic_pattern_weight.unsqueeze(-1)), dim=2)   
        # weights = torch.sigmoid(weights)
        return weights

class WCC_no_pk(nn.Module):
    def __init__(self, hidden_dim, K):
        super(WCC_no_pk, self).__init__()
        # 现在只有 error 和 basic_pattern 作为输入特征
        input_dim = out_dim = K + 1  # 只保留 error (K) 和 basic_pattern (1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, error, basic_pattern, K):
        # 将输入信号拼接，只拼接 error 和 basic_pattern
        x = torch.cat([error, basic_pattern], dim=2)
        
        # 通过网络获取权重
        x = self.fc1(x.permute(1, 0, 2))
        x = x.view(-1, x.shape[2])  # 将 x 变为 [B * L, K+1]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        weights = x.view(-1, basic_pattern.shape[1], K + 1)

        # 计算每个通道的权重
        other_weights = F.softmax(weights[:, :, :K], dim=-1) * K  # 计算 error 部分的权重
        basic_pattern_weight = torch.sigmoid(weights[:, :, -1])  # basic_pattern 的权重

        # 合并所有权重
        weights = torch.cat((other_weights, basic_pattern_weight.unsqueeze(-1)), dim=2)
        return weights