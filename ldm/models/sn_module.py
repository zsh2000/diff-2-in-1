
from functools import partial
import math
from typing import Iterable
from black import diff
from torch import nn, einsum
import numpy as np
import torch as th
import torch.nn as nn
import functools
import torch.nn.functional as F

import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange
import copy
from torchvision import transforms
from torchvision.transforms import InterpolationMode
class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
def resize_fn(img, size):
    return transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img))
import math
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos = None, query_pos = None):
        output = tgt
        
        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)

        return output

    
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos = None, query_pos = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt
# Projection of x onto y
def proj(x, y):
  return torch.mm(y, x.t()) * y / torch.mm(y, y.t())

# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
  for y in ys:
    x = x - proj(x, y)
  return x
def power_iteration(W, u_, update=True, eps=1e-12):
  # Lists holding singular vectors and values
  us, vs, svs = [], [], []
  for i, u in enumerate(u_):
    # Run one step of the power iteration
    with torch.no_grad():
      v = torch.matmul(u, W)
      # Run Gram-Schmidt to subtract components of all other singular vectors
      v = F.normalize(gram_schmidt(v, vs), eps=eps)
      # Add to the list
      vs += [v]
      # Update the other singular vector
      u = torch.matmul(v, W.t())
      # Run Gram-Schmidt to subtract components of all other singular vectors
      u = F.normalize(gram_schmidt(u, us), eps=eps)
      # Add to the list
      us += [u]
      if update:
        u_[i][:] = u
    # Compute this singular value and add it to the list
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
  return svs, us, vs

# Spectral normalization base class 
class SN(object):
  def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
    # Number of power iterations per step
    self.num_itrs = num_itrs
    # Number of singular values
    self.num_svs = num_svs
    # Transposed?
    self.transpose = transpose
    # Epsilon value for avoiding divide-by-0
    self.eps = eps
    # Register a singular vector for each sv
    for i in range(self.num_svs):
      self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
      self.register_buffer('sv%d' % i, torch.ones(1))
  
  # Singular vectors (u side)
  @property
  def u(self):
    return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

  # Singular values; 
  # note that these buffers are just for logging and are not used in training. 
  @property
  def sv(self):
   return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]
   
  # Compute the spectrally-normalized weight
  def W_(self):
    W_mat = self.weight.view(self.weight.size(0), -1)
    if self.transpose:
      W_mat = W_mat.t()
    # Apply num_itrs power iterations
    for _ in range(self.num_itrs):
      svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps) 
    # Update the svs
    if self.training:
      with torch.no_grad(): # Make sure to do this in a no_grad() context or you'll get memory leaks!
        for i, sv in enumerate(svs):
          self.sv[i][:] = sv     
    return self.weight / svs[0]

# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
  def __init__(self, in_features, out_features, bias=True,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Linear.__init__(self, in_features, out_features, bias)
    SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
  def forward(self, x):
    return F.linear(x, self.W_(), self.bias)

# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True, 
             num_svs=1, num_itrs=1, eps=1e-12):
    nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, 
                     padding, dilation, groups, bias)
    SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)    
  def forward(self, x):
    return F.conv2d(x, self.W_(), self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)

class SegBlock(nn.Module):
    def __init__(self, in_channels, out_channels, con_channels,
                which_conv=nn.Conv2d, which_linear=None, activation=None, 
                upsample=None):
        super(SegBlock, self).__init__()
        
        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_linear = which_conv, which_linear
        self.activation = activation
        self.upsample = upsample
        
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels, 
                                            kernel_size=1, padding=0)
       
        self.register_buffer('stored_mean1', torch.zeros(in_channels))
        self.register_buffer('stored_var1',  torch.ones(in_channels)) 
        self.register_buffer('stored_mean2', torch.zeros(out_channels))
        self.register_buffer('stored_var2',  torch.ones(out_channels)) 
        
        self.upsample = upsample

    def forward(self, x, y=None):
        x = F.batch_norm(x, self.stored_mean1, self.stored_var1, None, None,
                          self.training, 0.1, 1e-4)
        h = self.activation(x)
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = F.batch_norm(h, self.stored_mean2, self.stored_var2, None, None,
                          self.training, 0.1, 1e-4)
        
        h = self.activation(h)
        h = self.conv2(h)
        if self.learnable_sc:       
            x = self.conv_sc(x)
        return h + x

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):

        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs).double()
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:

                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x.double() * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):

    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : False,
                'input_dims' : 2,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class SNmodule(nn.Module):
    
    def __init__(self,
        embedding_dim=512,
        num_heads=8,
        num_layers=3,
        hidden_dim=2048,
        dropout_rate=0):
        super().__init__()

        low_feature_channel = 2048
        mid_feature_channel = 176
        high_feature_channel = 64
        highest_feature_channel = 64

        top_feature_channel = 40

        top_top_feature_channel = 24
        
        self.relu = nn.ReLU(inplace=True)

        self.low_feature_conv = nn.Sequential(
            nn.Conv2d(1280*6, low_feature_channel, kernel_size=1, bias=False),

        )
        self.mid_feature_conv = nn.Sequential(
            nn.Conv2d((1280*5+640), mid_feature_channel, kernel_size=1, bias=False),

        )
        self.mid_feature_mix_conv = SegBlock(
                                in_channels=low_feature_channel+mid_feature_channel,
                                out_channels=mid_feature_channel,
                                con_channels=128,
                                which_conv=functools.partial(SNConv2d,
                                    kernel_size=3, padding=1,
                                    num_svs=1, num_itrs=1,
                                    eps=1e-04),
                                which_linear=functools.partial(SNLinear,
                                    num_svs=1, num_itrs=1,
                                    eps=1e-04),
                                activation=nn.ReLU(inplace=True),
                                upsample=False,
                            )
        self.high_feature_conv = nn.Sequential(
            nn.Conv2d((1280+640*4+320), high_feature_channel, kernel_size=1, bias=False),
        )
        self.high_feature_mix_conv = SegBlock(
                                in_channels=mid_feature_channel+high_feature_channel,
                                out_channels=high_feature_channel,
                                con_channels=128,
                                which_conv=functools.partial(SNConv2d,
                                    kernel_size=3, padding=1,
                                    num_svs=1, num_itrs=1,
                                    eps=1e-04),
                                which_linear=functools.partial(SNLinear,
                                    num_svs=1, num_itrs=1,
                                    eps=1e-04),
                                activation=nn.ReLU(inplace=True),
                                upsample=False,
                            )
        self.highest_feature_conv = nn.Sequential(
            nn.Conv2d((640+320*6), highest_feature_channel, kernel_size=1, bias=False),
        )
        self.highest_feature_mix_conv = SegBlock(
                                in_channels=high_feature_channel+highest_feature_channel,
                                out_channels=highest_feature_channel,
                                con_channels=128,
                                which_conv=functools.partial(SNConv2d,
                                    kernel_size=3, padding=1,
                                    num_svs=1, num_itrs=1,
                                    eps=1e-04),
                                which_linear=functools.partial(SNLinear,
                                    num_svs=1, num_itrs=1,
                                    eps=1e-04),
                                activation=nn.ReLU(inplace=True),
                                upsample=False,
                            )
        
        feature_dim=low_feature_channel+mid_feature_channel+high_feature_channel+highest_feature_channel
        query_dim=feature_dim*16
        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate)
        self.transfromer_decoder = TransformerDecoder(decoder_layer, num_layers)
        self.mlp = MLP(embedding_dim, embedding_dim, feature_dim, 3)
        context_dim=768
        
        self.to_k = nn.Linear(query_dim, embedding_dim, bias=False)
        self.to_q = nn.Linear(context_dim, embedding_dim, bias=False)

        self.bn_up0 = nn.BatchNorm2d(64)
        self.conv_up1 = nn.ConvTranspose2d(64, 40, 2, 2, 0)
        self.bn_up1 = nn.BatchNorm2d(40)
        self.conv_up2 = nn.ConvTranspose2d(64+40, 24, 2, 2, 0)
        self.bn_up2 = nn.BatchNorm2d(24)


    def forward(self,diffusion_feature):

        low_feat, mid_feat, high_feat, image_feature = self._prepare_features(diffusion_feature)

        image_feature_0 = self.relu(self.bn_up0(image_feature))
        image_feature_1_before_relu = self.bn_up1(self.conv_up1(image_feature_0))
        image_feature_1 = self.relu(image_feature_1_before_relu)

        direct_up_1 = F.interpolate(image_feature_0, size=128, mode='bilinear', align_corners=False)
        image_feature_2_before_relu = self.bn_up2(self.conv_up2(torch.cat((image_feature_1, direct_up_1), 1)))
        
        return low_feat, mid_feat, high_feat, image_feature_1_before_relu, image_feature_2_before_relu

    def _prepare_features(self, features, upsample='bilinear'):
        self.low_feature_size = 16
        self.mid_feature_size = 32
        self.high_feature_size = 64
        
        low_features = [
            F.interpolate(i, size=self.low_feature_size, mode=upsample, align_corners=False) for i in features["low"]
        ]
#        print(low_features, "low_features")
        low_features = torch.cat(low_features, dim=1)
        
        mid_features = [
             F.interpolate(i, size=self.mid_feature_size, mode=upsample, align_corners=False) for i in features["mid"]
        ]
        mid_features = torch.cat(mid_features, dim=1)
        
        high_features = [
             F.interpolate(i, size=self.high_feature_size, mode=upsample, align_corners=False) for i in features["high"]
        ]
        high_features = torch.cat(high_features, dim=1)
        highest_features=torch.cat(features["highest"],dim=1)
        features_dict = {
            'low': low_features,
            'mid': mid_features,
            'high': high_features,
            'highest':highest_features,
        }
        
#        print(features_dict["low"].shape, features_dict["mid"].shape, features_dict["high"].shape, features_dict["highest"].shape)        
        low_feat_out = self.low_feature_conv(features_dict['low'])
        low_feat = F.interpolate(low_feat_out, size=self.mid_feature_size, mode='bilinear', align_corners=False)
        
        mid_feat = self.mid_feature_conv(features_dict['mid'])
        mid_feat = torch.cat([low_feat, mid_feat], dim=1)
        mid_feat_out = self.mid_feature_mix_conv(mid_feat, y=None)
        mid_feat = F.interpolate(mid_feat_out, size=self.high_feature_size, mode='bilinear', align_corners=False)
        
        high_feat = self.high_feature_conv(features_dict['high'])
        high_feat = torch.cat([mid_feat, high_feat], dim=1)
        high_feat = self.high_feature_mix_conv(high_feat, y=None)
        
        highest_feat=self.highest_feature_conv(features_dict['highest'])
        highest_feat=torch.cat([high_feat,highest_feat],dim=1)
        highest_feat=self.highest_feature_mix_conv(highest_feat,y=None)
        
        return low_feat_out, mid_feat_out, high_feat, highest_feat
   
