from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, GCNConv, GATConv, global_mean_pool, global_add_pool, global_max_pool 
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat, einsum


class GLADMamba(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, feat_dim, str_dim, args):
        super(GLADMamba, self).__init__()

        self.embedding_dim = hidden_dim * num_gc_layers                
        args.d_model = self.embedding_dim

        if args.GNN_Encoder == 'GCN':
            self.encoder_feat = Encoder_GCN(feat_dim, hidden_dim, num_gc_layers, args)                                             
            self.encoder_str = Encoder_GCN(str_dim, hidden_dim, num_gc_layers, args)                                              
        elif args.GNN_Encoder == 'GIN':
            self.encoder_feat = Encoder_GIN(feat_dim, hidden_dim, num_gc_layers, args)                                              
            self.encoder_str = Encoder_GIN(str_dim, hidden_dim, num_gc_layers, args)                                                
        elif args.GNN_Encoder == 'GAT':
            self.encoder_feat = Encoder_GAT(feat_dim, hidden_dim, num_gc_layers, args)                                            
            self.encoder_str = Encoder_GAT(str_dim, hidden_dim, num_gc_layers, args)                                            

        self.proj_head_feat_g = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_str_g = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                            nn.Linear(self.embedding_dim, self.embedding_dim))  

        self.out_norm = RMSNorm(args.d_model)
        self.graph_mamba = MambaBlock(args)  

        self.proj_head_feat_n = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_str_n = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                            nn.Linear(self.embedding_dim, self.embedding_dim)) 
        
        self.RQ_act=nn.LeakyReLU()
        self.RQ_lin_f=nn.Linear(feat_dim, self.embedding_dim)
        self.RQ_lin_s=nn.Linear(str_dim, self.embedding_dim)           
 
        self.init_emb() 


    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)        
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def forward(self, data, x_f, x_s, edge_index, batch, num_graphs, args): 
        g_f, n_f = self.encoder_feat(x_f, edge_index, batch)                    
        g_s, n_s = self.encoder_str(x_s, edge_index, batch)                     

        g_f_1 = self.proj_head_feat_g(g_f)                                        
        g_s_1 = self.proj_head_str_g(g_s) 
        n_f_1 = self.proj_head_feat_n(n_f)
        n_s_1 = self.proj_head_str_n(n_s)

        RQ_f = self.RQ_lin_f(data.XLX_f)
        RQ_s = self.RQ_lin_s(data.XLX_s)
        RQ_f = self.RQ_act(RQ_f)
        RQ_s = self.RQ_act(RQ_s)

        g_f_2 = self.graph_mamba(g_f_1, RQ_f, args)
        g_s_2 = self.graph_mamba(g_s_1, RQ_s, args)
        n_f_2 = self.graph_mamba(n_f_1, n_s_1, args)
        n_s_2 = self.graph_mamba(n_s_1, n_f_1, args)

        g_f_3 = g_f_1 + g_f_2
        g_s_3 = g_s_1 + g_s_2
        n_f_3 = n_f_1 + n_f_2
        n_s_3 = n_s_1 + n_s_2 

        g_f_3_norm = self.out_norm(g_f_3)
        g_s_3_norm = self.out_norm(g_s_3)
        n_f_3_norm = self.out_norm(n_f_3)
        n_s_3_norm = self.out_norm(n_s_3)

        return g_f_3_norm, g_s_3_norm, n_f_3_norm, n_s_3_norm         

    @staticmethod
    def calc_loss_n(x, x_aug, batch, temperature=0.2):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)


        node_belonging_mask = batch.repeat(batch_size,1)
        node_belonging_mask = node_belonging_mask == node_belonging_mask.t()

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature) * node_belonging_mask
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-12)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-12)

        loss_0 = - torch.log(loss_0)
        loss_1 = - torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        loss = global_mean_pool(loss, batch)
        return loss

    @staticmethod
    def calc_loss_g(x, x_aug, temperature=0.2):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

        loss_0 = - torch.log(loss_0)
        loss_1 = - torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        return loss


class MambaBlock(nn.Module):
    def __init__(self,							
                 args
                ):
        super().__init__()

        self.args = args
        self.norm = RMSNorm(args.d_model)
        self.conv1d = nn.Conv1d(
            in_channels=args.d_model,
            out_channels=args.d_model,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_model,
            padding=args.d_conv - 1,
        )

        self.in_proj = nn.Linear(args.d_model, args.d_model * 2, bias=args.bias)
        self.x_proj = nn.Linear(args.d_model, args.dt_rank + args.d_state * 2, bias=False)     
        self.dt_proj = nn.Linear(args.dt_rank, args.d_model, bias=True)
        self.out_proj = nn.Linear(args.d_model, args.d_model, bias=args.bias)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_model)                 
        self.A_log = torch.nn.Parameter(torch.log(A))
        self.D = torch.nn.Parameter(torch.ones(args.d_model))         


    def forward(self, x_1, x_2, args):
        x_1 = self.norm(x_1)
        x_2 = self.norm(x_2)

        (b, d) = x_1.shape                                                 
        expanded_x_1 = x_1.unsqueeze(1)
        x_1 = expanded_x_1.expand(b, args.l, d)                   
        (b, l, d) = x_1.shape                                              

        (b, d) = x_2.shape
        expanded_x_2 = x_2.unsqueeze(1)
        x_2 = expanded_x_2.expand(b, args.l, d)
        (b, l, d) = x_2.shape 
        
        ##---------- Linear ----------##
        x_and_res_1 = self.in_proj(x_1)
        (x_1, res_1) = x_and_res_1.split(split_size=[self.args.d_model, self.args.d_model], dim=-1)

        x_and_res_2 = self.in_proj(x_2)
        (x_2, res_2) = x_and_res_2.split(split_size=[self.args.d_model, self.args.d_model], dim=-1)

        ##----------- Conv -----------##
        x_1 = rearrange(x_1, 'b l d_in -> b d_in l')
        x_1 = self.conv1d(x_1)[:, :, :l]                                        
        x_1 = rearrange(x_1, 'b d_in l -> b l d_in')

        x_2 = rearrange(x_2, 'b l d_in -> b d_in l')
        x_2 = self.conv1d(x_2)[:, :, :l]
        x_2 = rearrange(x_2, 'b d_in l -> b l d_in')
       
        ##----------- SiLU ----------##
        x_1 = F.silu(x_1)
        x_2 = F.silu(x_2)

        y = self.ssm(x_1, x_2)                     

        ##------ SiLU, Multiplication ---##
        y = y * F.silu(res_1)

        ##------- Linear ----------##
        output = self.out_proj(y)

        output = self.norm(output)
        return output[:,-1,:]  

    
    def ssm (self, x_1, x_2):
        (d_in, n) = self.A_log.shape                           
        A = -torch.exp(self.A_log.float())        				
        D = self.D.float()                                    
    
        x_dbl_1 = self.x_proj(x_1)				      		
        x_dbl_2 = self.x_proj(x_2)                           

        (delta, B, C) = x_dbl_2.split(split_size=[self.args.dt_rank, n, n], dim=-1)          
        delta = F.softplus(self.dt_proj(delta))   				

        y = self.selective_scan(x_1, delta, A, B, C, D)           
        return y

    def selective_scan(self, u , delta, A, B, C, D): 			
        (b, l, d_in) = u.shape                      
        n = A.shape[1]                

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))              
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')       

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = [] 

        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]                   
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')      
            
            ys.append(y)
        y = torch.stack(ys, dim=1)     				
        y = y + u * D
        return y


class RMSNorm(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5
                ):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class Encoder_GIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, args):
        super(Encoder_GIN, self).__init__()

        self.num_gc_layers = num_gc_layers                  
        self.convs = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))             
            else:   
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))    
            conv = GINConv(nn)                                                          
            self.convs.append(conv)             
        self.pool_type = args.graph_level_pool


    def forward(self, x, edge_index, batch):
        xs = []                                                  
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)

        if self.pool_type == 'global_max_pool':
            xpool = [global_max_pool(x, batch) for x in xs]                                
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]

        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)                                                     


class Encoder_GCN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, args):
        super(Encoder_GCN, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                conv = GCNConv(dim, dim)
            else:
                conv = GCNConv(num_features, dim)
            self.convs.append(conv)

        self.pool_type = args.graph_level_pool


    def forward(self, x, edge_index, batch):
        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)

        if self.pool_type == 'global_max_pool':
            xpool = [global_max_pool(x, batch) for x in xs]                               
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]

        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)


class Encoder_GAT(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, args):
        super(Encoder_GAT, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                conv = GATConv(dim, dim)
            else:
                conv = GATConv(num_features, dim)
            self.convs.append(conv)
        self.pool_type = args.graph_level_pool


    def forward(self, x, edge_index, batch):
        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)

        if self.pool_type == 'global_max_pool':
            xpool = [global_max_pool(x, batch) for x in xs]                                 
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]

        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)