# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 19:48:33 2021

@author: Administrator
"""
import torch
import torch.nn as nn
from vit_pytorch.vit import Transformer, pair
from einops import repeat,rearrange
from einops.layers.torch import Rearrange
try:
    from .cw_vit import cw_vit,outer_sampling,get_sinusoid_encoding
except:
    from cw_vit import cw_vit,outer_sampling,get_sinusoid_encoding

class decay_vit(cw_vit):
    def __init__(self, img_size, in_chans, args, pool = 'mean', dim_head = 64):
        super().__init__(img_size, in_chans, args, pool, dim_head)
        #----------------------map trainable ratio----------------------
        self.num_patches = self.img_size**2
        self.set_pos_embedding()
        self.has_map_ratio = args.has_map_ratio
        # self.cls_token_add_mean = args.cls_token_add_mean
        if self.has_map_ratio:
            self.map_ratio = nn.Parameter(data=torch.ones(self.num_patches), requires_grad=True)



    def clockwise_input(self, inputs, edge, mid=None):
        x = inputs.clone()
        x = self.rearrange(x)
        if mid==None:
            mid1 = mid2 = int(edge/2)
        else:
            mid1,mid2 = mid
        
        all_index = [[mid1*edge+mid2]]
        for i in range(1,int(edge/2)+1):
            a1 = mid1-i
            a2 = mid1+i
            b1 = mid2-i
            b2 = mid2+i
            e1,e2,e3,e4 = 0,edge,0,edge
            is_list = type(self.ratio) == list
            x,index,output = outer_sampling(x,[a1,a2,b1,b2],[e1,e2,e3,e4],mid1,mid2,
                                            ratio=self.ratio[i-1] if is_list else self.ratio)
            all_index += [index]
        return x,all_index,output
    
    
    def clockwise_pos_embedding(self, all_index):
        self.cls_token.data = torch.ones(1, 1, self.dim).to(self.device)
        pos_embedding_ = self.pos_embedding.to('cpu')
        for index in all_index:
            pos_embedding_.data[:,index] = get_sinusoid_encoding(
                n_position=len(index), d_hid=self.dim)
        self.pos_embedding.data = pos_embedding_.to(self.device)
        
    def forward(self, x):
        x,all_index,_ = self.clockwise_input(x, edge=self.img_size)
        if self.has_map_ratio:
            x = x*self.map_ratio
        x = x.transpose(1,2)
        if self.set_embed_linear:
            x = self.embed_linear(x)
        b, n, d = x.shape
        
        if self.cw_pe:
            self.clockwise_pos_embedding(all_index)
            self.cw_pe=0
        if self.has_pe:
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
        else:
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
            # if self.cls_token_add_mean:  
            #     cls_tokens += x.mean(1).reshape(b, 1, d)
            x = torch.cat((cls_tokens, x), dim=1)
            
            
            
        x = self.emb_dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        
        x = self.to_latent(x)
        return self.mlp_head(x)
    
        


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim,
                 depth, heads, mlp_dim, pool = 'mean', channels = 3,
                 dim_head = 64, dropout = 0., emb_dropout = 0.,
                 set_embed_linear=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.set_embed_linear = set_embed_linear
        
        if not self.set_embed_linear:
            dim = channels*patch_height*patch_width

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
        self.embed_linear = nn.Linear(patch_dim, dim)

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.pos_embedding = nn.Parameter(data=get_sinusoid_encoding(
                n_position=num_patches + 1, d_hid=dim), requires_grad=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.rearrange(img)
        b, n, _ = x.shape
        if self.set_embed_linear:
            x = self.embed_linear(x)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

