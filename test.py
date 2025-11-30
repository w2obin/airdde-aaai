import torch
from torch import nn

# --------------------------------------------------
# 你的 LocalMemoryModule (保持一致即可)
# --------------------------------------------------
import math
class LocalMemoryModule(nn.Module):
    def __init__(self, num_nodes, d_model, tau=3, k_neighbors=8):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.tau = tau
        self.k_neighbors = k_neighbors
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, h_e, x_orig):
        batch_size, seq_len, num_nodes, d_model = h_e.shape

        if x_orig.dim() == 4:
            x_reshape = x_orig.permute(1, 0, 2, 3)
        elif x_orig.dim() == 3:
            x_reshape = x_orig.permute(1, 0, 2).reshape(batch_size, seq_len, num_nodes, -1)
        else:
            raise ValueError(f"x_orig dim {x_orig.dim()} not supported")

        wind_vars = x_reshape[:, :, :, 4:6]
        last_wind = wind_vars[:, -1]
        b, n, _ = last_wind.shape
        wind_flat = last_wind.reshape(b, n, -1)

        dist = torch.cdist(wind_flat, wind_flat)
        sim = -dist
        k = min(self.k_neighbors, n)
        topk_idx = sim.topk(k=k, dim=-1).indices

        t0 = seq_len - 1
        t_start = max(0, t0 - self.tau + 1)
        hist = h_e[:, t_start:t0 + 1].permute(0, 2, 1, 3)
        tau_eff = hist.size(2)

        batch_idx = torch.arange(b, device=h_e.device).view(b, 1, 1).expand(b, n, k)
        neighbor_hist = hist[batch_idx, topk_idx]
        neighbor_hist = neighbor_hist.reshape(b, n, k * tau_eff, d_model)

        q = h_e[:, t0]
        q = self.q_proj(q).unsqueeze(2)
        k_feat = self.k_proj(neighbor_hist)
        v_feat = self.v_proj(neighbor_hist)

        scale = math.sqrt(d_model)
        attn_scores = (q * k_feat).sum(-1) / scale
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)
        context = (attn_weights * v_feat).sum(2)
        h_l = self.mlp(context)
        return h_l


# --------------------------------------------------
# 1. 构造随机输入，模拟真实场景
# --------------------------------------------------
T = 12          # 序列长度
B = 4           # batch size
N = 10          # 节点数
F = 8           # 特征数（至少包含 wind_speed, wind_dir）
d_model = 32    # hidden dim

# h_e: (T, B, N, d)
h_e = torch.randn(T, B, N, d_model)

# x_orig: (T, B, N*F)
x_orig = torch.randn(T, B, N * F)

# --------------------------------------------------
# 2. 初始化 LocalMemoryModule
# --------------------------------------------------
local_mem = LocalMemoryModule(num_nodes=N, d_model=d_model, tau=4, k_neighbors=5)

# --------------------------------------------------
# 3. 调用
# --------------------------------------------------
h_l = local_mem(h_e, x_orig)

print("Local memory output shape:", h_l.shape)
print("Example values (first batch, first node):\n", h_l[0, 0])
