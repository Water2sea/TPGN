import torch
import torch.nn as nn

import torch.nn.utils.weight_norm as wn
import torch.nn.functional as F

import numpy as np
from functools import partial

from scipy.special import eval_legendre
from sympy import Poly, legendre, Symbol, chebyshevt


def legendreDer(k, x):
    def _legendre(k, x):
        return (2*k+1) * eval_legendre(k, x)
    out = 0
    for i in np.arange(k-1,-1,-2):
        out += _legendre(i, x)
    return out


def phi_(phi_c, x, lb = 0, ub = 1):
    mask = np.logical_or(x<lb, x>ub) * 1.0
    return np.polynomial.polynomial.Polynomial(phi_c)(x) * (1-mask)


def get_phi_psi(k, base):
    
    x = Symbol('x')
    phi_coeff = np.zeros((k,k))
    phi_2x_coeff = np.zeros((k,k))
    if base == 'legendre':
        for ki in range(k):
            coeff_ = Poly(legendre(ki, 2*x-1), x).all_coeffs()
            phi_coeff[ki,:ki+1] = np.flip(np.sqrt(2*ki+1) * np.array(coeff_).astype(np.float64))
            coeff_ = Poly(legendre(ki, 4*x-1), x).all_coeffs()
            phi_2x_coeff[ki,:ki+1] = np.flip(np.sqrt(2) * np.sqrt(2*ki+1) * np.array(coeff_).astype(np.float64))
        
        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))
        for ki in range(k):
            psi1_coeff[ki,:] = phi_2x_coeff[ki,:]
            for i in range(k):
                a = phi_2x_coeff[ki,:ki+1]
                b = phi_coeff[i, :i+1]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_)<1e-8] = 0
                proj_ = (prod_ * 1/(np.arange(len(prod_))+1) * np.power(0.5, 1+np.arange(len(prod_)))).sum()
                psi1_coeff[ki,:] -= proj_ * phi_coeff[i,:]
                psi2_coeff[ki,:] -= proj_ * phi_coeff[i,:]
            for j in range(ki):
                a = phi_2x_coeff[ki,:ki+1]
                b = psi1_coeff[j, :]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_)<1e-8] = 0
                proj_ = (prod_ * 1/(np.arange(len(prod_))+1) * np.power(0.5, 1+np.arange(len(prod_)))).sum()
                psi1_coeff[ki,:] -= proj_ * psi1_coeff[j,:]
                psi2_coeff[ki,:] -= proj_ * psi2_coeff[j,:]

            a = psi1_coeff[ki,:]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_)<1e-8] = 0
            norm1 = (prod_ * 1/(np.arange(len(prod_))+1) * np.power(0.5, 1+np.arange(len(prod_)))).sum()

            a = psi2_coeff[ki,:]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_)<1e-8] = 0
            norm2 = (prod_ * 1/(np.arange(len(prod_))+1) * (1-np.power(0.5, 1+np.arange(len(prod_))))).sum()
            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki,:] /= norm_
            psi2_coeff[ki,:] /= norm_
            psi1_coeff[np.abs(psi1_coeff)<1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff)<1e-8] = 0

        phi = [np.poly1d(np.flip(phi_coeff[i,:])) for i in range(k)]
        psi1 = [np.poly1d(np.flip(psi1_coeff[i,:])) for i in range(k)]
        psi2 = [np.poly1d(np.flip(psi2_coeff[i,:])) for i in range(k)]
    
    elif base == 'chebyshev':
        for ki in range(k):
            if ki == 0:
                phi_coeff[ki,:ki+1] = np.sqrt(2/np.pi)
                phi_2x_coeff[ki,:ki+1] = np.sqrt(2/np.pi) * np.sqrt(2)
            else:
                coeff_ = Poly(chebyshevt(ki, 2*x-1), x).all_coeffs()
                phi_coeff[ki,:ki+1] = np.flip(2/np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))
                coeff_ = Poly(chebyshevt(ki, 4*x-1), x).all_coeffs()
                phi_2x_coeff[ki,:ki+1] = np.flip(np.sqrt(2) * 2 / np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))
                
        phi = [partial(phi_, phi_coeff[i,:]) for i in range(k)]
        
        x = Symbol('x')
        kUse = 2*k
        roots = Poly(chebyshevt(kUse, 2*x-1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
        # not needed for our purpose here, we use even k always to avoid
        wm = np.pi / kUse / 2
        
        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))

        psi1 = [[] for _ in range(k)]
        psi2 = [[] for _ in range(k)]

        for ki in range(k):
            psi1_coeff[ki,:] = phi_2x_coeff[ki,:]
            for i in range(k):
                proj_ = (wm * phi[i](x_m) * np.sqrt(2)* phi[ki](2*x_m)).sum()
                psi1_coeff[ki,:] -= proj_ * phi_coeff[i,:]
                psi2_coeff[ki,:] -= proj_ * phi_coeff[i,:]

            for j in range(ki):
                proj_ = (wm * psi1[j](x_m) * np.sqrt(2) * phi[ki](2*x_m)).sum()        
                psi1_coeff[ki,:] -= proj_ * psi1_coeff[j,:]
                psi2_coeff[ki,:] -= proj_ * psi2_coeff[j,:]

            psi1[ki] = partial(phi_, psi1_coeff[ki,:], lb = 0, ub = 0.5)
            psi2[ki] = partial(phi_, psi2_coeff[ki,:], lb = 0.5, ub = 1)

            norm1 = (wm * psi1[ki](x_m) * psi1[ki](x_m)).sum()
            norm2 = (wm * psi2[ki](x_m) * psi2[ki](x_m)).sum()

            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki,:] /= norm_
            psi2_coeff[ki,:] /= norm_
            psi1_coeff[np.abs(psi1_coeff)<1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff)<1e-8] = 0

            psi1[ki] = partial(phi_, psi1_coeff[ki,:], lb = 0, ub = 0.5+1e-16)
            psi2[ki] = partial(phi_, psi2_coeff[ki,:], lb = 0.5+1e-16, ub = 1)
        
    return phi, psi1, psi2


def get_filter(base, k):
    
    def psi(psi1, psi2, i, inp):
        mask = (inp<=0.5) * 1.0
        return psi1[i](inp) * mask + psi2[i](inp) * (1-mask)
    
    if base not in ['legendre', 'chebyshev']:
        raise Exception('Base not supported')
    
    x = Symbol('x')
    H0 = np.zeros((k,k))
    H1 = np.zeros((k,k))
    G0 = np.zeros((k,k))
    G1 = np.zeros((k,k))
    PHI0 = np.zeros((k,k))
    PHI1 = np.zeros((k,k))
    phi, psi1, psi2 = get_phi_psi(k, base)
    if base == 'legendre':
        roots = Poly(legendre(k, 2*x-1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        wm = 1/k/legendreDer(k,2*x_m-1)/eval_legendre(k-1,2*x_m-1)
        
        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki](x_m/2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m/2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki]((x_m+1)/2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m+1)/2) * phi[kpi](x_m)).sum()
                
        PHI0 = np.eye(k)
        PHI1 = np.eye(k)
                
    elif base == 'chebyshev':
        x = Symbol('x')
        kUse = 2*k
        roots = Poly(chebyshevt(kUse, 2*x-1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
        # not needed for our purpose here, we use even k always to avoid
        wm = np.pi / kUse / 2

        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki](x_m/2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m/2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki]((x_m+1)/2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m+1)/2) * phi[kpi](x_m)).sum()

                PHI0[ki, kpi] = (wm * phi[ki](2*x_m) * phi[kpi](2*x_m)).sum() * 2
                PHI1[ki, kpi] = (wm * phi[ki](2*x_m-1) * phi[kpi](2*x_m-1)).sum() * 2
                
        PHI0[np.abs(PHI0)<1e-8] = 0
        PHI1[np.abs(PHI1)<1e-8] = 0

    H0[np.abs(H0)<1e-8] = 0
    H1[np.abs(H1)<1e-8] = 0
    G0[np.abs(G0)<1e-8] = 0
    G1[np.abs(G1)<1e-8] = 0
        
    return H0, H1, G0, G1, PHI0, PHI1


def train(model, train_loader, optimizer, epoch, device, verbose = 0,
    lossFn = None, lr_schedule=None, 
    post_proc = lambda args: args):
        
    if lossFn is None:
        lossFn = nn.MSELoss()

    model.train()
    
    total_loss = 0.

    for batch_idx, (data, target) in enumerate(train_loader):
        
        bs = len(data)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        
        target = post_proc(target)
        output = post_proc(output)
        loss = lossFn(output.view(bs, -1), target.view(bs, -1))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.sum().item()
    if lr_schedule is not None: lr_schedule.step()
    
    if verbose>0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        
    return total_loss/len(train_loader.dataset)


def test(model, test_loader, device, verbose=0, lossFn=None,
        post_proc = lambda args: args):
    
    model.eval()
    if lossFn is None:
        lossFn = nn.MSELoss()
    
    
    total_loss = 0.
    predictions = []
    
    with torch.no_grad():
        for data, target in test_loader:
            bs = len(data)

            data, target = data.to(device), target.to(device)
            output = model(data)
            output = post_proc(output)
            
            loss = lossFn(output.view(bs, -1), target.view(bs, -1))
            total_loss += loss.sum().item()
    
    return total_loss/len(test_loader.dataset)


# Till EoF
# taken from FNO paper:
# https://github.com/zongyi-li/fourier_neural_operator

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x
    
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class MLP(nn.Module):
    def __init__(self,input_len,output_len):
        super().__init__()
        self.linear1 = nn.Sequential(
            wn(nn.Linear(input_len, output_len)),
            nn.ReLU(),
            wn(nn.Linear(output_len,output_len))
        )

        self.linear2 = nn.Sequential(
            wn(nn.Linear(output_len, output_len)),
            nn.ReLU(),
            wn(nn.Linear(output_len, output_len))
        )

        self.skip = wn(nn.Linear(input_len, output_len))
        self.act = nn.ReLU()
        
    def forward(self,x):
        x = self.act(self.linear1(x)+self.skip(x))
        x = self.linear2(x)
        
        return x
    
class MLP_bottle(nn.Module):
    def __init__(self,input_len,output_len,bottleneck,bias=True):
        super().__init__()
        self.linear1 = nn.Sequential(
            wn(nn.Linear(input_len, bottleneck,bias=bias)),
            nn.ReLU(),
            wn(nn.Linear(bottleneck,bottleneck,bias=bias))
        )

        self.linear2 = nn.Sequential(
            wn(nn.Linear(bottleneck, bottleneck)),
            nn.ReLU(),
            wn(nn.Linear(bottleneck, output_len))
        )

        self.skip = wn(nn.Linear(input_len, bottleneck,bias=bias))
        self.act = nn.ReLU()
        
    def forward(self,x):
        x = self.act(self.linear1(x)+self.skip(x))
        x = self.linear2(x)
        
        return x

class Coefnet(nn.Module):
    def __init__(self, blocks,d_model,heads,norm_layer=None, projection=None):
        super().__init__()
        layers = [BCAB(d_model,heads) for i in range(blocks)]
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        # heads = heads if blocks > 0 else 1
        self.last_layer = last_layer(d_model,heads)

    def forward(self, basis, series):
        attns1 = []
        attns2 = []
        for layer in self.layers:
            basis,series,basis_attn,series_attn = layer(basis,series)   #basis(B,N,d)  series(B,C,d)
            attns1.append(basis_attn)
            attns2.append(series_attn)
        
        coef = self.last_layer(series,basis)  #(B,k,C,N)
        
        return coef,attns1,attns2

class BCAB(nn.Module):
    def __init__(self, d_model,heads=8,index=0,d_ff=None,
                     dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.cross_attention_basis = channel_AutoCorrelationLayer(d_model,heads,dropout=dropout)
        self.conv1_basis = wn(nn.Linear(d_model,d_ff))
        self.conv2_basis = wn(nn.Linear(d_ff,d_model))

        self.dropout_basis = nn.Dropout(dropout)
        self.activation_basis = F.relu if activation == "relu" else F.gelu
        
        self.cross_attention_ts = channel_AutoCorrelationLayer(d_model,heads,dropout=dropout)
        self.conv1_ts = wn(nn.Linear(d_model,d_ff))
        self.conv2_ts = wn(nn.Linear(d_ff,d_model))

        self.dropout_ts = nn.Dropout(dropout)
        self.activation_ts = F.relu if activation == "relu" else F.gelu
        self.layer_norm11 = nn.LayerNorm(d_model)
        self.layer_norm12 = nn.LayerNorm(d_model)
        self.layer_norm21 = nn.LayerNorm(d_model)
        self.layer_norm22 = nn.LayerNorm(d_model)

    def forward(self, basis,series):
        basis_raw = basis
        series_raw = series
        basis_add, basis_attn = self.cross_attention_basis(
            basis_raw, series_raw, series_raw,
        )
        basis_out = basis_raw + self.dropout_basis(basis_add)
        basis_out = self.layer_norm11(basis_out)

        y_basis = basis_out
        y_basis = self.dropout_basis(self.activation_basis(self.conv1_basis(y_basis)))
        y_basis = self.dropout_basis(self.conv2_basis(y_basis))
        basis_out = basis_out + y_basis
        
        basis_out = self.layer_norm12(basis_out)
        
        series_add,series_attn = self.cross_attention_ts(
            series_raw, basis_raw, basis_raw
        )
        series_out = series_raw + self.dropout_ts(series_add)
        
        series_out = self.layer_norm21(series_out)

        y_ts = series_out
        y_ts = self.dropout_ts(self.activation_ts(self.conv1_ts(y_ts)))
        y_ts = self.dropout_ts(self.conv2_ts(y_ts))
        series_out = series_out + y_ts
        series_out = series_raw
        
        series_out = self.layer_norm22(series_out)

        return basis_out, series_out, basis_attn, series_attn
    
class channel_AutoCorrelationLayer(nn.Module):
    def __init__(self,d_model,n_heads, mask=False,d_keys=None,
                 d_values=None,dropout=0):
        super().__init__()
        
        self.mask = mask

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.query_projection = wn(nn.Linear(d_model,d_keys * n_heads))
        self.key_projection = wn(nn.Linear(d_model, d_keys * n_heads))
        self.value_projection = wn(nn.Linear(d_model, d_values * n_heads))
        self.out_projection = wn(nn.Linear(d_values * n_heads, d_model))
        self.n_heads = n_heads
        self.scale = d_keys ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        num = len(queries.shape)
        if num == 2:
            L, _ = queries.shape
            S, _ = keys.shape
            H = self.n_heads

            queries = self.query_projection(queries).view(L, H, -1).permute(1,0,2)
            keys = self.key_projection(keys).view(S, H, -1).permute(1,0,2)
            values = self.value_projection(values).view(S, H, -1).permute(1,0,2)
            # queries = queries.view(L, H, -1).permute(1,0,2)
            # keys = keys.view(S, H, -1).permute(1,0,2)
            # values = values.view(S, H, -1).permute(1,0,2)

            dots = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)
            attn = self.dropout(attn)

            out = torch.matmul(attn, values)    #(H,L,D)

            out = out.permute(1,0,2).reshape(L,-1)
        else:
            B,L, _ = queries.shape
            B,S, _ = keys.shape
            H = self.n_heads

            queries = self.query_projection(queries).view(B,L, H, -1).permute(0,2,1,3)
            keys = self.key_projection(keys).view(B,S, H, -1).permute(0,2,1,3)
            values = self.value_projection(values).view(B,S, H, -1).permute(0,2,1,3)

            dots = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)
            
            attn = self.dropout(attn)

            out = torch.matmul(attn, values)    #(H,L,D)

            out = out.permute(0,2,1,3).reshape(B,L,-1)
            
        return self.out_projection(out),attn
    
class last_layer(nn.Module):
    def __init__(self,d_model,n_heads, mask=False,d_keys=None,
                 d_values=None,dropout=0):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.query_projection = wn(nn.Linear(d_model,d_keys * n_heads))
        self.key_projection = wn(nn.Linear(d_model, d_keys * n_heads))
        self.n_heads = n_heads
        self.scale = d_keys ** -0.5

    def forward(self, queries, keys):
        B,L, _ = queries.shape
        B,S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B,L, H, -1).permute(0,2,1,3)
        keys = self.key_projection(keys).view(B,S, H, -1).permute(0,2,1,3)

        dots = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale   #(B,H,L,S)

        return dots
    
