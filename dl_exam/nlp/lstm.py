"""
Author: Redal
Date: 2025-10-14
Todo: lstm.py for creating lstm and GRU model
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# 标准LSTM与PyTorch nn.LSTM功能一致,用于教学/自定义扩展
# 包含: 遗忘门, 输入门, 输出门+记忆单元(Cell State); 相关公式包括: 
# f_t = \sigma(W_{xf}·x_t + W_{hf}·h_{t-1} + b_f)
# i_t &= \sigma(W_{xi}·x_t + W_{hi}·h_{t-1} + b_i) 
# \tilde{C}_t &= \tanh(W_{xc}·x_t + W_{hc}·h_{t-1} + b_c) 
# C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t 
# o_t &= \sigma(W_{xo}·x_t + W_{ho}·h_{t-1} + b_o) 
# h_t &= o_t \odot \tanh(C_t)
class VanillaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, 
                 num_layers=1, bias=True, dropout=0.0, 
                 bidirectional=False):
        super().__init__()
        # 初始化LSTM层所需超参数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        # 使用PyTorch内置LSTM(高效且支持cuDNN)
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bias=self.bias,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)
    def forward(self, x, h0=None, c0=None):
        batch_size = x.size(0)
        if h0 is None:
            h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, 
                             self.hidden_size, device=x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers * self.num_directions, batch_size,
                             self.hidden_size, device=x.device)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        return output, (hn, cn)


# 带窥孔连接Peephole Connections的LSTM
class PeepholeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 从输入到各门的权重值
        self.W_xi = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_ci = nn.Parameter(torch.randn(hidden_size)) # 窥孔链接cell -> input gate
        
        self.W_xf = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_cf = nn.Parameter(torch.randn(hidden_size)) # 窥孔链接cell -> forget gate

        self.W_xo = nn.Linear(input_size, hidden_size, bias=False)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_co = nn.Parameter(torch.randn(hidden_size)) # 窥孔链接cell -> output gate
        # 初始参数化输入, 遗忘, 记忆元与输出门的偏置参数bias
        self.bias_i = nn.Parameter(torch.randn(hidden_size))
        self.bias_f = nn.Parameter(torch.randn(hidden_size))
        self.bias_c = nn.Parameter(torch.randn(hidden_size))
        self.bias_o = nn.Parameter(torch.randn(hidden_size))
    def forward(self, x, h_prev=None, c_prev=None):
        # h_prev前一时间步的隐藏状态，默认为None
        # c_prev前一时间步的细胞状态，默认为None
        batch_size, seq_len, _ = x.shape
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        if c_prev is None:
            c_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        outputs = []
        h_t, c_t = h_prev, c_prev
        for t in range(seq_len):
            # x形状[batch_size, input_size]
            x_t = x[:, t, :]
            # 输入门, 遗忘门, 候选记忆元, 更新cell
            i_t = torch.sigmoid(self.W_xi(x_t) + self.W_hi(h_t) + self.W_ci * c_t, self.bias_i)
            f_t = torch.sigmoid(self.W_xf(x_t) + self.W_hf(h_t) + self.W_cf * c_t, self.bias_f)
            c_hat = torch.tanh(self.W_xc(x_t) + self.W_hc(h_t) + self.bias_c)
            c_t = f_t * c_t + i_t * c_hat
            # 输出门, 更新隐藏状态
            o_t = torch.sigmoid(self.W_xo(x_t) + self.W_ho(h_t) + self.W_co * c_t, self.bias_o)
            h_t = o_t * torch.tanh(c_t)
        # 输出output形状[Batch_size, seq_len, hidden_size]
        output = torch.cat(outputs, dim=1)
        return output, (h_t, c_t)

        
# 耦合输入-遗忘门 LSTM：f_t = 1 - i_t
class CoupledLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size


        