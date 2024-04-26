"""
In parts from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange, einsum



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
        self.mlp = nn.Sequential(
            norm_layer(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(x)


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
        dots = (q @ k.mT) * self.scale

        if mask is not None:
            mask_value = torch.finfo(dots.dtype).min
            dots.masked_fill_(mask, mask_value)

        # improve numerical stability of softmax
        dots = dots - torch.amax(dots, dim=-1, keepdim=True)
        attn = F.softmax(dots, dim=-1)

        out = attn @ v
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
    def __init__(self, *, num_classes, input_dim=768, dim=512, depth=2, heads=8, mlp_dim=2048, pool='cls',
                 dim_head=64, dropout=0., emb_dropout=0., pos_emb_density: int = 10):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.pos_emb_density = pos_emb_density
        self.pos_emb = nn.Parameter(torch.randn(pos_emb_density, pos_emb_density, dim))
        #  slide embedding only allows 25 slides per patient
        self.register_buffer("slide_emb", self.sinusoid_encoding(25, dim))

        self.fc = nn.Sequential(nn.Linear(input_dim, dim, bias=True), nn.GELU())
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, nn.LayerNorm, dropout)

        self.pool = pool
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, lens, slide_ids, coords):
        # remove unnecessary padding
        max_idx = torch.amax(lens)
        x, slide_ids, coords = x[:, :max_idx], slide_ids[:, :max_idx], coords[:, :max_idx] 
        b, n, d = x.shape

        # map input sequence to latent space of TransMIL
        x = self.dropout(self.fc(x))

        # apply positional embedding
        # note: feature vectors added during zero padding get assigned the embedding
        # for the top left patch, however they are masked out in the transformer,
        # thus they don't influence the gradient of this embedding vector
        pos_emb = self.bilinear_interp(coords)
        slide_emb = self.slide_emb[slide_ids]
        x = x + pos_emb + slide_emb

        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        lens = lens + 1 # account for cls token

        # mask indicating zero padded feature vectors
        mask = None
        if torch.amin(lens) != torch.amax(lens):
            mask = torch.arange(0, torch.max(lens), dtype=torch.int32, device=x.device).repeat(b, 1) < lens[..., None]
            mask = mask[:, None, :, None].to(torch.float)
            mask = ~(mask @ mask.mT).to(torch.bool)
        
        x = self.transformer(x, mask)

        if mask is not None and self.pool == 'mean':
            x = torch.cumsum(x, dim=1)[torch.arange(b), lens-1]
            x = x / lens[..., None]
        else:
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)

    def bilinear_interp(self, coords):
        """
        Bilinear interpolation for getting the positional embeddings for each
        patch from the learnable pos. embedding matrix.
        """
        (b, l), d = coords.shape[:2], self.pos_emb.shape[-1]
        coords = coords.reshape(-1, 2)
        device = coords.device

        offset = torch.tensor([
            [0, 0], # upper left
            [0, 1], # upper right
            [1, 0], # lower left
            [1, 1], # lower right
        ], dtype=torch.int32, device=device)

        coords = coords * (self.pos_emb_density - 1)
        corner_idcs = torch.floor(coords).to(device, torch.int32).unsqueeze(1) + offset
        corner_idcs.clip_(0, self.pos_emb_density - 1)

        corner_idcs = corner_idcs.reshape(-1, 2)
        corners = self.pos_emb[corner_idcs[:, 0], corner_idcs[:, 1]]
        corner_idcs = corner_idcs.reshape(-1, 4, 2)
        corners = corners.reshape(-1, 2, 2, d)

        in_cell_pos = coords - corner_idcs[:, 0]
        del_x = torch.cat((1 - in_cell_pos[:, :1], in_cell_pos[:, :1]), dim=-1)
        del_y = torch.cat((1 - in_cell_pos[:, 1:], in_cell_pos[:, 1:]), dim=-1)

        interp = einsum(del_x, corners, del_y, "b h, b h w d, b w -> b d")
        return interp.reshape(b, l, d)

    def sinusoid_encoding(self, num_tokens, token_len):
        """
        Sinusoidal embedding used for distinguishing between patches of different slides.
        """
        position_angle_vec = 1 / torch.pow(100, 2 * (torch.arange(token_len) // 2) / token_len)
        sinusoid_table = torch.outer(torch.arange(num_tokens), position_angle_vec)
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])

        sinusoid_table.sub_(sinusoid_table.mean(axis=-1, keepdim=True))
        sinusoid_table.div_(sinusoid_table.std(axis=-1, keepdim=True))

        return sinusoid_table
