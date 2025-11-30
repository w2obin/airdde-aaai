import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import ChebConv
from torch_geometric.utils import dense_to_sparse
from torchdiffeq import odeint_adjoint as odeint


class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(2*cheb_k*dim_in, dim_out)) # 2 is the length of support
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)
        
    def forward(self, x, supports):
        x_g = []        
        support_set = []
        for support in supports:
            support_ks = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2]) 
            support_set.extend(support_ks)
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1) # B, N, 2 * cheb_k * dim_in
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out
        return x_gconv
    
class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k)
        self.update = AGCN(dim_in+self.hidden_dim, dim_out, cheb_k)

    def forward(self, x, state, supports):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        # print(x.shape, state.shape)
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    
class STEncoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(STEncoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, x, init_state, supports):
        # x: (B, T, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, supports)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden
    
    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states

class STDecoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(STDecoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, xt, init_state, supports):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], supports)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden



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

        # 做了一个动态优化
        # 从所有节点中，为每个地点 i 按“风驱动相似度/邻近性”选择 k 个最有可能将污染传输到 i 的邻居
        # sim（相似度）越大，说明两个节点的风速/风向越接近，也越可能有：距离相近、污染传输相关
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

        # local attention
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

class DiffeqSolver:
    def __init__(self, method, odeint_rtol=1e-5,
                 odeint_atol=1e-5, adjoint=True):
        self.ode_method = method
        self.odeint = odeint
        self.rtol = odeint_rtol
        self.atol = odeint_atol

    def solve(self, odefunc, first_point, time_steps_to_pred):
        pred_y = self.odeint(odefunc,
                             first_point,
                             time_steps_to_pred,
                             rtol=self.rtol,
                             atol=self.atol,
                             method=self.ode_method)
        return pred_y


class ODEFunc(nn.Module):
    def __init__(self, gcn_hidden_dim, input_dim, adj_mx, edge_index, edge_attr,
                 K_neighbour, num_nodes, device, num_layers=2,
                 activation='tanh', filter_type="diff_adv", estimate=False):
        super(ODEFunc, self).__init__()
        self.device = device

        self._activation = torch.tanh if activation == 'tanh' else torch.relu

        self.num_nodes = num_nodes
        self.gcn_hidden_dim = gcn_hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        

        self._filter_type = filter_type

        self.adj_mx = adj_mx
        self.edge_index = torch.tensor(edge_index, dtype=torch.int64).to(self.device)
        self.edge_attr = edge_attr
        self.K_neighbour = K_neighbour

        self.diff_edge_attr = self.edge_attr[:, 0]
        self.adv_edge_attr = None

        self.source_sink = None

        self.source_sink_pred = nn.Linear(128+128,64)
        self.source_embed = nn.Linear(self.gcn_hidden_dim+1,1)
        self.norm = nn.LayerNorm(self.num_nodes)

        self.residual = nn.Identity()

        self.diff_cheb_conv = self.laplacian_operator()
        self.adv_cheb_conv = self.laplacian_operator()
        
        # batch_size, node_num, hidden
        self.previous_x = torch.randn(64,187,64).to(self.device)

    def create_adv_matrix(self, last_wind_vars, wind_mean, wind_std):
        batch_size = last_wind_vars.shape[0]
        edge_src, edge_target = self.edge_index
        node_src = last_wind_vars[:, edge_src, :]
        node_target = last_wind_vars[:, edge_target, :]

        src_wind_speed = node_src[:, :, 0] * wind_std[0] + wind_mean[0]    # km/h
        src_wind_dir = node_src[:, :, 1] * wind_std[1] + wind_mean[1]
        dist = self.edge_attr[:, 1].unsqueeze(dim=0).repeat(batch_size, 1)
        dist_dir = self.edge_attr[:, 2].unsqueeze(dim=0).repeat(batch_size, 1)

        src_wind_dir = (src_wind_dir + 180) % 360
        theta = torch.abs(dist_dir - src_wind_dir)
        adv_edge_attr = F.relu(3 * src_wind_speed * torch.cos(theta) / dist)  # B x M

        return adv_edge_attr

    def create_equation(self, last_wind_vars, wind_mean, wind_std):
        self.adv_edge_attr = self.create_adv_matrix(last_wind_vars, wind_mean, wind_std)

    def create_source_matrix(self, features):
        source_term = self.source_sink_pred(features) # B,N,64
        self.source_sink = source_term

    def forward(self, t_local, Xt):
        grad_diff = self.ode_func_net_diff(Xt, self.diff_edge_attr)
        grad_adv = self.ode_func_net_adv(Xt, self.adv_edge_attr)
        grad_source = self.ode_func_net_source_sink(Xt, self.source_sink)
        # print(grad_diff.shape, grad_adv.shape, grad_source.shape)
        grad = 0.1 * grad_diff + grad_adv + grad_source
        return grad

    def ode_func_net_source_sink(self, x, source):
        out = torch.cat([x.unsqueeze(-1), source], dim=-1)
        out = self.norm(self.source_embed(out).squeeze(-1))
        return out

    def ode_func_net_diff(self, x, edge_attr):
        # x: B x N*var_dim
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, self.num_nodes, self.input_dim))

        x = self.diff_cheb_conv[0](x, self.edge_index, edge_attr, lambda_max=2)
        x = self._activation(x)

        for op in self.diff_cheb_conv[1:-1]:
            residual = self.residual(x)
            x = op(x, self.edge_index, edge_attr, lambda_max=2)
            x = self._activation(x) + residual

        x = self.diff_cheb_conv[-1](x, self.edge_index, edge_attr, lambda_max=2)

        return x.reshape((batch_size, self.num_nodes * self.input_dim))

    def ode_func_net_adv(self, x, edge_attr):
        batch_size = x.shape[0]
        batch = torch.arange(0, batch_size)
        batch = torch.repeat_interleave(batch, self.num_nodes).to(self.device)
        x = x.reshape(batch_size * self.num_nodes, -1)  # B*N x input_dim
        x = x + 0.01 * self.previous_x.sum(dim=1).sum(dim=-1).reshape(batch_size * self.num_nodes, -1)

        edge_indices = []
        for i in range(batch_size):
            edge_indices.append(self.edge_index + i * self.num_nodes)
        edge_index = torch.cat(edge_indices, dim=1)  # 2 x B*M
        edge_attr = edge_attr.flatten()  # B*M

        x = self.adv_cheb_conv[0](x, edge_index, edge_attr, batch=batch, lambda_max=2)
        x = self._activation(x)

        for op in self.adv_cheb_conv[1:-1]:
            residual = self.residual(x)
            x = op(x, edge_index, edge_attr, batch=batch, lambda_max=2)
            x = self._activation(x) + residual

        x = self.adv_cheb_conv[-1](x, edge_index, edge_attr, batch=batch, lambda_max=2)

        x = x.reshape(batch_size, self.num_nodes, self.input_dim)
        return x.reshape((batch_size, self.num_nodes * self.input_dim))

    @staticmethod
    def dense_to_sparse(adj: torch.Tensor):
        batch_size, num_nodes, _ = adj.size()
        edge_indices = []
        edge_attrs = []

        for i in range(batch_size):
            edge_index, edge_attr = dense_to_sparse(adj[i])
            edge_indices.append(edge_index + i * num_nodes)
            edge_attrs.append(edge_attr)

        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs, dim=0)

        return edge_index, edge_attr

    def laplacian_operator(self):
        # approximate Laplacian
        operator = nn.ModuleList()
        operator.append(
            ChebConv(in_channels=self.input_dim, out_channels=self.gcn_hidden_dim,
                     K=self.K_neighbour, normalization='sym',
                     bias=True)
        )

        for _ in range(self.num_layers - 2):
            operator.append(
                ChebConv(in_channels=self.gcn_hidden_dim, out_channels=self.gcn_hidden_dim,
                         K=self.K_neighbour, normalization='sym',bias=True)
            )

        operator.append(
            ChebConv(in_channels=self.gcn_hidden_dim, out_channels=self.input_dim,
                     K=self.K_neighbour, normalization='sym', bias=True)
        )

        return operator



class Model(nn.Module):
    def __init__(self, adj_mx, edge_index, edge_attr, node_attr, wind_mean,wind_std,
                num_nodes, input_dim, output_dim, horizon, rnn_units, num_layers=1, cheb_k=3,
                 ycov_dim=5, mem_num=20, mem_dim=64, cl_decay_steps=2000, use_curriculum_learning=True):
        super(Model, self).__init__()
        self.adj_mx = adj_mx
        self.edge_index = torch.tensor(edge_index, dtype=torch.int32).to("cuda:0")
        self.edge_attr = torch.from_numpy(edge_attr).float().to("cuda:0")
        self.node_attr = node_attr

        self.wind_mean = wind_mean
        self.wind_std = wind_std

        self.wind_mean = wind_mean[-2:]
        self.wind_std = wind_std[-2:]

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.rnn_units = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        
        # memory
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.glo_memory = self.construct_global_memory()
        self.loc_memory = LocalMemoryModule(num_nodes=self.num_nodes, d_model=self.rnn_units, tau=3, k_neighbors=8)
        self.memory_embed = nn.Sequential(nn.Linear(self.mem_dim+self.mem_dim, self.mem_dim, bias=True))

        # encoder
        self.encoder = STEncoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.num_layers)
        
        # deocoder
        self.decoder_dim = self.rnn_units + self.mem_dim
        self.decoder = STDecoder(self.num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim, self.cheb_k, self.num_layers)

        # solver
        self.phy_solver = DiffeqSolver(
            method="dopri5",
            odeint_atol=1e-2,
            odeint_rtol=1e-2,
            adjoint=True
        )

        self.phy_odefunc = ODEFunc(gcn_hidden_dim=64, input_dim=1, adj_mx=self.adj_mx, edge_index=self.edge_index, edge_attr=self.edge_attr,
                 K_neighbour=3, num_nodes=184, device="cuda:0", num_layers=3,
                 activation='tanh', filter_type="diff_adv", estimate=False)
        
        self.phy_output = nn.Linear(self.decoder_dim, self.output_dim, bias=True)
        self.y_cov_embed_layer = nn.Linear(24*5, self.decoder_dim)
        
        # output
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))

        self.setting = self.get_setting()
    
    def construct_global_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)     # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim), requires_grad=True)
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True)
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True)
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict
    
    def global_memory_modeling(self, h_t:torch.Tensor):
        query = torch.matmul(h_t, self.glo_memory['Wq'])     # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.glo_memory['Memory'].t()), dim=-1)         # alpha: (B, N, M)
        value = torch.matmul(att_score, self.glo_memory['Memory'])     # (B, N, d)
        return value, query
            
    def forward(self, x, y_cov, labels=None, batches_seen=None):
        # inputs data
        # x: T B N D
        # y_cov: T_pred B N (D-1)
        x_orig = x.clone() # T B N D
        seq_len, batch_size = x.size(0), x.size(1)
        x = x.reshape(seq_len, batch_size, self.num_nodes, self.input_dim).permute(1,0,2,3) # B T N D

        # STencoder
        node_embeddings1 = torch.matmul(self.glo_memory['We1'], self.glo_memory['Memory'])
        node_embeddings2 = torch.matmul(self.glo_memory['We2'], self.glo_memory['Memory'])
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)
        supports = [g1, g2]
        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, _ = self.encoder(x, init_state, supports) # B, T, N, hidden
        h_t = h_en[:, -1, :, :] # B, N, hidden (last state)        
        
        # global memory
        h_global, query = self.global_memory_modeling(h_t)        
        # local memory
        h_local = self.loc_memory(h_en, x_orig)
        # print(h_global.shape, h_local.shape)

        h_memory = self.memory_embed(torch.cat([h_global, h_local], dim=-1)) # B, N, hidden
        h_embed = torch.cat([h_t,h_memory], dim=-1) # B, N, hidden+hidden
        # h_embed = torch.cat([h_global, h_local], dim=-1)
        # print(h_embed.shape)
        ht_list = [h_embed]*self.num_layers
        
        # func init
        x_reshape = x_orig.reshape(seq_len, batch_size, self.num_nodes, -1)
        wind_vars = x_reshape[:, :, :, 4: 6]  # T x B x N x 2   wind speed and wind direction
        last_wind_vars = wind_vars[-1]  # B x N x 2
        self.phy_odefunc.create_equation(last_wind_vars, self.wind_mean, self.wind_std)
        
        # func inputs
        tau_back = 3  # 回溯窗口\tau 是3
        self.phy_odefunc.previous_x = h_en[:,-1-tau_back:-1,:,:]
        
        phy_input = x_reshape[-1,:,:,0].reshape(batch_size,-1)
        y_cov_embed = self.y_cov_embed_layer(y_cov.permute(1,2,3,0).reshape(batch_size, self.num_nodes, -1)).squeeze(-1)
        self.phy_odefunc.create_source_matrix(torch.cat([h_embed, y_cov_embed], dim=-1))
        time_steps_to_predict = torch.arange(start=0, end=self.horizon + 1, step=1).float()  # horizon 1 + 24
        time_steps_to_predict = time_steps_to_predict / len(time_steps_to_predict)
        
        # future evo
        phy_y = self.phy_solver.solve(self.phy_odefunc, phy_input, time_steps_to_predict)  # T x B x N*D
        phy_y = phy_y[1:]

        # STDecoder
        out = []
        for t in range(self.horizon):
            h_de, ht_list = self.decoder(torch.cat([phy_y[t, ...].unsqueeze(-1), y_cov[t, ...]], dim=-1), ht_list, supports)
            h_de = self.proj(h_de)
            out.append(h_de)
        output = torch.stack(out, dim=1)
        output = output.squeeze(-1)
        output = output.permute(1,0,2)

        return output
    
    def get_setting(self):
        setting = 'airdde'
        return setting
