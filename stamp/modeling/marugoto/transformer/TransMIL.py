"""
In parts from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange



class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, norm_layer=nn.LayerNorm, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            norm_layer(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=512 // 8, norm_layer=nn.LayerNorm, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = heads != 1 or dim_head != dim

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = norm_layer(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            dots.masked_fill_(mask, mask_value)

        # improve numerical stability of softmax
        dots = dots - torch.amax(dots, dim=-1, keepdim=True)

        attn = F.softmax(dots, dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, norm_layer=nn.LayerNorm, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, norm_layer=norm_layer, dropout=dropout),
                FeedForward(dim, mlp_dim, norm_layer=norm_layer, dropout=dropout)
            ]))
        self.norm = norm_layer(dim)

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return self.norm(x)


class TransMIL(nn.Module):
    def __init__(self, *, num_classes, input_dim=768, dim=512, depth=2, heads=8, mlp_dim=512, pool='cls',
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.fc = nn.Sequential(nn.Linear(input_dim, dim, bias=True), nn.GELU())
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, nn.LayerNorm, dropout)

        self.pool = pool
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, lens):
        x = x[:, :torch.amax(lens)] # remove unnecessary padding
        b, n, d = x.shape

        # map input sequence to latent space of TransMIL
        x = self.dropout(self.fc(x))

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        lens = lens + 1 # account for cls token

        mask = None
        if torch.amin(lens) != torch.amax(lens):
            mask = torch.arange(0, torch.max(lens), dtype=torch.int32, device=x.device).repeat(b, 1) >= lens[..., None]
            mask = mask[:, None, :, None].to(torch.float)
            mask = (mask @ mask.mT).to(torch.bool)
        
        x = self.transformer(x, mask)

        if mask is not None and self.pool == 'mean':
            x = torch.cumsum(x, dim=1)[torch.arange(b), lens-1]
            x = x / lens[..., None]
        else:
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.mlp_head(x)
        return x
