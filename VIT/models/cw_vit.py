# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 19:48:33 2021

@author: Administrator
"""
import torch
import torch.nn as nn
import math
import numpy as np
from vit_pytorch.vit import ViT,Transformer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_


    
def outer_sampling(x,window_edge,global_edge,mid1,mid2, ratio=1):
    #using dynamic programming : the ratio is untrainable  使用动态规划,但是衰减系数固定
    #a1 a2为纵坐标两个边界, b1 b2为纵坐标两个边界
    #e1,e2,e3,e4为整体矩阵的四个边界
    #mid1 mid2 中心点坐标
    batch_size,_,_ = x.shape
    a1,a2,b1,b2 = window_edge
    e1,e2,e3,e4 = global_edge
    index = []
    direction = [0,1]
    l1 = a1
    l2 = b1-1
    start=0
    edge = e4-e3
    i=0
    ratio_list = []
    if type(ratio) == torch.Tensor:
        ratio_list = ratio
    while(1):
        l1 += direction[0]
        l2 += direction[1]
        if l1==a1 and l2==b1:
            if start==1:break
            else:start=1
        if l1>a2:
            direction = [0,-1]
            l1-=1
            continue
        elif l2<b1:
            direction = [-1,0]
            l2+=1
            continue
        elif l2>b2:
            direction = [1,0]
            l2-=1
            continue
        if l1<e1 or l1>e2 or l2<e3 or l2>e4:
            continue
        if len(ratio_list)>0:
            ratio = ratio_list[:,i].reshape(batch_size,1)
        #数据池化方式1
        m1 = 0 if l1-mid1==0 else int(-(l1-mid1)/np.abs(l1-mid1))
        m2 = 0 if l2-mid2==0 else int(-(l2-mid2)/np.abs(l2-mid2))
        if len(x.shape)==3:
            if np.abs(l1-mid1)>np.abs(l2-mid2):
                x[:,:,l1*edge+l2] = (x[:,:,(l1+m1)*edge+l2]+x[:,:,(l1+m1)*edge+l2+m2])*ratio/2 + x[:,:,l1*edge+l2]
            elif np.abs(l1-mid1)<np.abs(l2-mid2):
                x[:,:,l1*edge+l2] = (x[:,:,(l1)*edge+l2+m2]+x[:,:,(l1+m1)*edge+l2+m2])*ratio/2 + x[:,:,l1*edge+l2]
            else:
                x[:,:,l1*edge+l2] = x[:,:,(l1+m1)*edge+l2+m2]*ratio + x[:,:,l1*edge+l2]
            index += [l1*edge+l2]
        else:
            if np.abs(l1-mid1)>np.abs(l2-mid2):
                x[l1*edge+l2] = (x[(l1+m1)*edge+l2]+x[(l1+m1)*edge+l2+m2])*ratio/2 + x[l1*edge+l2]
            elif np.abs(l1-mid1)<np.abs(l2-mid2):
                x[l1*edge+l2] = (x[(l1)*edge+l2+m2]+x[(l1+m1)*edge+l2+m2])*ratio/2 + x[l1*edge+l2]
            else:
                x[l1*edge+l2] = x[(l1+m1)*edge+l2+m2]*ratio + x[l1*edge+l2]
            index += [l1*edge+l2]
            
        i += 1
    return x, index, x[:,:,index].transpose(1,2)



class cw_vit(nn.Module):
    def __init__(self, img_size, in_chans, args, pool = 'mean', dim_head = 64):
        # num_classes=10, depth=7, mlp_dim=200,
        #  dim=200, heads=3, dim_head = 64, dropout = 0., emb_dropout = 0.,
        # pool = 'cls', pos_embed='sin', set_embed_linear=False, ratio=1,
        # ratio_trainable=False
        super().__init__()
        self.num_classes=args.CLASSES_NUM
        self.heads = args.heads
        self.depth=args.depth
        self.mlp_dim=args.mlp_dim
        self.dim=args.dim
        self.pos_embed=args.pos_embed
        self.set_embed_linear = args.set_embed_linear
        self.dropout = args.dropout
        self.emb_dropout = nn.Dropout(args.emb_dropout)
        self.ratio = args.ratio
        self.ratio_trainable = args.ratio_trainable
        self.heads = args.heads
        self.img_size = img_size
        self.rearrange = Rearrange('b c h w -> b c (h w)')
        self.in_chans = in_chans
        self.dim_head = dim_head
        self.pool = pool
        self.device = args.device
        self.has_pe = args.has_pe
        self.cw_pe = args.cw_pe
        
        
        if not self.set_embed_linear:
            self.dim = self.in_chans
        # self.rearrange2 = Rearrange('b c a -> b a c')
        self.num_patches = 4*(self.img_size-1)
        self.embed_linear = nn.Linear(self.in_chans, self.dim)
        
        
        self.set_pos_embedding()
        #-------------------------decay ratio from middle to edge---------------------------------
        if self.ratio_trainable:
            self.ratio = nn.Parameter(data=torch.Tensor(self.ratio), requires_grad=True)
            

        self.transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head,
                                       self.mlp_dim, self.dropout)
        
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes)
        )
    
    def set_pos_embedding(self):
        #------------------------pos_embed in [sin random trainable_sin]-------------------------
        if self.has_pe==0:
            self.pos_embed='random'
            
            
        if self.pos_embed=='sin':
            self.pos_embedding = nn.Parameter(data=get_sinusoid_encoding(
                n_position=self.num_patches + 1, d_hid=self.dim), requires_grad=False)
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        
        elif self.pos_embed=='random':
            self.pos_embedding = nn.Parameter(torch.randn(1,self.num_patches + 1, self.dim), requires_grad=False)
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        
        elif self.pos_embed=='trainable_sin':
            self.pos_embedding = nn.Parameter(
                data=get_sinusoid_encoding(
                    n_position=self.num_patches + 1, d_hid=self.dim), requires_grad=True)
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
            
    def reset_decay_ratio(self):
        self.ratio = nn.Parameter(data=self.ratio.mean(0).repeat((self.img_size-1)*4)\
                                  .reshape((self.img_size-1)*4,-1).detach(), requires_grad=True)
        
    # @torch.no_grad()
    def clockwise_input(self, inputs, edge, mid=None):
        x = inputs.clone()
        x = self.rearrange(x)
        if mid==None:
            mid1 = mid2 = int(edge/2)
        else:
            mid1,mid2 = mid
        
        # if self.ratio_trainable:
        #     i = int(edge/2)
        #     a1 = mid1-i
        #     a2 = mid1+i
        #     b1 = mid2-i
        #     b2 = mid2+i
        #     x,index,output = outer_sampling2(x,[a1,a2,b1,b2],mid1,mid2,ratio=self.ratio)
        # else:
        for i in range(1,int(edge/2)+1):
            a1 = mid1-i
            a2 = mid1+i
            b1 = mid2-i
            b2 = mid2+i
            e1,e2,e3,e4 = 0,edge,0,edge
            is_list = type(self.ratio) == list
            with torch.no_grad():
                x,index,output = outer_sampling(x,[a1,a2,b1,b2],[e1,e2,e3,e4],mid1,mid2,
                                                ratio=self.ratio[i-1] if is_list else self.ratio)
        return x,index,output
        
                
    def forward(self, x):
        _,_,x = self.clockwise_input(x, edge=self.img_size)
        if self.set_embed_linear:
            x = self.embed_linear(x)
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.emb_dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class hs_cw_vit(cw_vit):
    #cw_vit with hierarchical structure
    def __init__(self, img_size, in_chans, args, pool = 'cls', dim_head = 64):
        # num_classes=10, depth=7, mlp_dim=200,
        #  dim=200, heads=3, dim_head = 64, dropout = 0., emb_dropout = 0.,
        # pool = 'cls', pos_embed='sin', set_embed_linear=False, ratio=1,
        # ratio_trainable=False
        super().__init__(img_size, in_chans, args, pool, dim_head)
        self.num_patches = self.img_size**2
        self.set_pos_embedding()

        
        # trunc_normal_(self.dist_toke3n, std=.02)
    def set_pos_embedding(self):
        self.ratio_mlp_head = []
        self.ratio_transformer = Transformer(self.dim, 7, 3, self.dim_head,
                                       self.mlp_dim, self.dropout)
        #------------------------pos_embed in [sin random trainable_sin]-------------------------  
        if self.pos_embed=='sin':
            self.pos_embedding = nn.Parameter(data=get_sinusoid_encoding(
                n_position=self.num_patches + 1, d_hid=self.dim), requires_grad=False)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        
        elif self.pos_embed=='random':
            self.pos_embedding = nn.Parameter(torch.randn(1,self.num_patches + 1, self.dim))
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        
        elif self.pos_embed=='trainable_sin':
            self.pos_embedding = nn.Parameter(
                data=get_sinusoid_encoding(
                    n_position=self.num_patches + 1, d_hid=self.dim), requires_grad=True)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
            
        for edge in range(3,self.img_size+1,2):
            num_patches = (edge-1)*4
            if edge!=3:
                self.ratio_mlp_head += [nn.Sequential(
                    nn.LayerNorm(self.dim),
                    nn.Linear(self.dim, num_patches)
                        ).to(self.device)]
    
    def clockwise_pos_embedding(self, all_index):
        self.cls_token.data = torch.ones(1, 1, self.dim).to(self.device)
        pos_embedding_ = self.pos_embedding.to('cpu')
        for index in all_index:
            pos_embedding_.data[:,index] = get_sinusoid_encoding(
                n_position=len(index), d_hid=self.dim)
        self.pos_embedding.data = pos_embedding_.to(self.device)

    # @torch.no_grad()
    def clockwise_input(self, inputs, edge, mid=None):
        x = inputs.clone()
        x = self.rearrange(x)
        if mid==None:
            mid1 = mid2 = int(edge/2)
        else:
            mid1,mid2 = mid
        
        ratio = 0
        all_index = []
        for i in range(1,int(edge/2)+1):
            a1 = mid1-i
            a2 = mid1+i
            b1 = mid2-i
            b2 = mid2+i
            e1,e2,e3,e4 = 0,edge,0,edge
            x,index,output = outer_sampling(x,[a1,a2,b1,b2],[e1,e2,e3,e4],mid1,mid2,
                                            ratio=ratio)
            if i==1:
                index = [mid1*edge+mid2]+index
            all_index += [index]
            ratio = self.ratio_transformer(x[:,:,index].transpose(1,2))
            ratio = ratio.mean(dim = 1) if self.pool == 'mean' else ratio[:, 0]
            ratio = self.to_latent(ratio)
            if i!=int(edge/2):
                ratio = self.ratio_mlp_head[i-1](ratio)
            all_index += [index]
        return x,all_index,output
        
                
    def forward(self, x):
        _,all_index,x = self.clockwise_input(x, edge=self.img_size)
        if self.set_embed_linear:
            x = self.embed_linear(x)
        b, n, _ = x.shape
        
        
        if self.cw_pe:
            self.clockwise_pos_embedding(all_index)
            self.cw_pe=0
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.emb_dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)



def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

        



