import torch.nn as nn
import torch.nn.utils.weight_norm as wn
import torch
from layers.utils import Coefnet, MLP_bottle

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.d_model = d_model = configs.d_model
        self.k = heads = configs.n_heads
        self.e_layers = e_layers = configs.e_layers
        self.d_ff = d_ff = configs.d_ff

        self.dropout = dropout = configs.dropout

        self.N = configs.N
        block_nums = configs.block_nums
        bottle = configs.bottleneck
        map_bottleneck = configs.map_bottleneck

        self.coefnet = Coefnet(blocks=block_nums,d_model=d_model,heads=heads)
            
        self.pred_len = pred_len = configs.pred_len
        self.seq_len = seq_len = configs.seq_len
        
        self.MLP_x = MLP_bottle(seq_len,heads * int(seq_len/heads),int(seq_len/bottle))
        self.MLP_y = MLP_bottle(pred_len,heads * int(pred_len/heads),int(pred_len/bottle))
        self.MLP_sx = MLP_bottle(heads * int(seq_len/heads),seq_len,int(seq_len/bottle))
        self.MLP_sy = MLP_bottle(heads * int(pred_len/heads),pred_len,int(pred_len/bottle))
        
        self.project1 = wn(nn.Linear(seq_len,d_model))
        self.project2 = wn(nn.Linear(seq_len,d_model))
        self.project3 = wn(nn.Linear(pred_len,d_model))
        self.project4 = wn(nn.Linear(pred_len,d_model))
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.L1Loss(reduction='none')
        
        self.device = device = configs.gpu
                        
        # smooth array
        arr = torch.zeros((seq_len+pred_len-2,seq_len+pred_len))
        for i in range(seq_len+pred_len-2):
            arr[i,i]=-1
            arr[i,i+1] = 2
            arr[i,i+2] = -1
        self.smooth_arr = arr.to(device)

        if configs.freq== 'h':
            T_fea_dim = 4
        elif configs.freq== 't':
            T_fea_dim = 5

        self.map_MLP = MLP_bottle(T_fea_dim,self.N*(self.seq_len+self.pred_len),map_bottleneck,bias=True)
        self.epsilon = 1E-5
        
    def forward(self, x, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        mean_x = x.mean(dim=1,keepdim=True)
        std_x = x.std(dim=1,keepdim=True)
        feature = (x - mean_x) / (std_x + self.epsilon)
        B,L,C = feature.shape
        feature = feature.permute(0,2,1)
        feature = self.project1(feature)   #(B,C,d)
        
        m = self.map_MLP(x_mark_enc[:,0,:]).reshape(B,self.seq_len + self.pred_len,self.N)
        m = m / torch.sqrt(torch.sum(m**2,dim=1,keepdim=True)+self.epsilon)
        
        raw_m1 = m[:,:self.seq_len].permute(0,2,1)  #(B,L,N)
        raw_m2 = m[:,self.seq_len:].permute(0,2,1)   #(B,L',N)
        m1 = self.project2(raw_m1)    #(B,N,d)
        
        score,attn_x1,attn_x2 = self.coefnet(m1,feature)    #(B,k,C,N)

        base = self.MLP_y(raw_m2).reshape(B,self.N,self.k,-1).permute(0,2,1,3)   #(B,k,N,L/k)
        out = torch.matmul(score,base).permute(0,2,1,3).reshape(B,C,-1)  #(B,C,k * (L/k))
        out = self.MLP_sy(out).reshape(B,C,-1).permute(0,2,1)   #（BC,L）
        
        output = out * (std_x + self.epsilon) + mean_x 
        
        return output  
        
        
