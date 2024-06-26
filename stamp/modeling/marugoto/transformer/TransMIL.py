"""
In parts from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange, einsum

import matplotlib.pyplot as plt
from pathlib import Path



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


# class Attention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=512 // 8, norm_layer=nn.LayerNorm, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         project_out = heads != 1 or dim_head != dim

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.norm = norm_layer(dim)

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x, mask=None):
#         x = self.norm(x)

#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
#         dots = (q @ k.mT) * self.scale

#         if mask is not None:
#             mask_value = torch.finfo(dots.dtype).min
#             dots.masked_fill_(mask, mask_value)

#         # improve numerical stability of softmax
#         dots = dots - torch.amax(dots, dim=-1, keepdim=True)
#         attn = F.softmax(dots, dim=-1)

#         out = attn @ v
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out), attn


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=512 // 8, norm_layer=nn.LayerNorm, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = norm_layer(dim)
        self.mhsa = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)


    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.repeat(self.heads, 1, 1)

        x = self.norm(x)        
        attn_output, attn_output_weights = self.mhsa(x, x, x, attn_mask=mask)
        return attn_output, attn_output_weights


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, norm_layer=nn.LayerNorm, dropout=0.):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, norm_layer=norm_layer, dropout=dropout),
                FeedForward(dim, mlp_dim, norm_layer=norm_layer, dropout=dropout)
            ]))
        self.norm = norm_layer(dim)

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x_attn, attn_mask = attn(x, mask=mask)
            x = x_attn + x
            x = ff(x) + x
        return self.norm(x), attn_mask


class InterpolationEmbedding2d(nn.Module): 
    def __init__(self, dim: int, grid_size: int = 10):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        # self.embedding = nn.Parameter(torch.zeros(grid_size, grid_size, dim))
        self.embedding = nn.Parameter(torch.randn(grid_size, grid_size, dim) * 1e-5)
        self.register_buffer("offset", torch.tensor([
                                                        [0, 0], # upper left
                                                        [0, 1], # upper right
                                                        [1, 0], # lower left
                                                        [1, 1], # lower right
                                                    ], dtype=torch.int32)
        )
    
    def forward(self, coords):
        """
        Bilinear interpolation for getting the positional embeddings for each
        patch from the learnable pos. embedding matrix.
        """
        (b, l), d = coords.shape[:2], self.dim
        coords = coords.reshape(-1, 2)
    
        coords = coords * (self.grid_size - 1)
        corner_idcs = torch.floor(coords).to(torch.int32).unsqueeze(1) + self.offset
        corner_idcs.clip_(0, self.grid_size - 1)

        corner_idcs = corner_idcs.reshape(-1, 2)
        corners = self.embedding[corner_idcs[:, 0], corner_idcs[:, 1]]
        corner_idcs = corner_idcs.reshape(-1, 4, 2)
        corners = corners.reshape(-1, 2, 2, d)      

        in_cell_pos = coords - corner_idcs[:, 0]
        del_x = torch.cat((1 - in_cell_pos[:, :1], in_cell_pos[:, :1]), dim=-1)
        del_y = torch.cat((1 - in_cell_pos[:, 1:], in_cell_pos[:, 1:]), dim=-1)
        interp = einsum(del_x, corners, del_y, "b h, b h w d, b w -> b d")

        return interp.reshape(b, l, d)
    
    def __repr__(self):
        return f"InterpolationEmbedding2d(dim={self.dim}, grid_size={self.grid_size})"

    def plot(self, path=None, interpolate: bool = False):
        fig = plt.figure(figsize=(self.grid_size, self.grid_size))
        device = self.offset.device

        with torch.no_grad():
            if interpolate:
                step_size = 15
                samples = (self.grid_size - 1) * step_size + 1
                x_coords, y_coords = torch.meshgrid(torch.linspace(0, 1, samples, device=device), torch.linspace(0, 1, samples, device=device), indexing="ij")
                x_coords = x_coords.reshape(-1, 1)
                y_coords = y_coords.reshape(-1, 1)
                grid = torch.cat((x_coords, y_coords), dim=1)[None]
                emb = self(grid)
                emb = emb.reshape(samples, samples, -1)
            else:
                step_size = 1
                samples = self.grid_size
                emb = self.embedding

            emb = torch.nn.functional.normalize(emb, 2, dim=-1)
            emb1, emb2 = emb[::step_size, ::step_size].reshape(-1, 512), emb.reshape(-1, 512)
            cossim = (emb1 @ emb2.T).reshape(self.grid_size, self.grid_size, samples, samples)

            # emb1, emb2 = emb[::step_size, ::step_size].reshape(-1, 1, 512), emb.reshape(-1, 512)
            # l2sim = torch.linalg.norm(emb1 - emb2, 2, dim=-1)
            # cossim = l2sim.reshape(self.grid_size, self.grid_size, samples, samples)

        fig, axs = plt.subplots(self.grid_size, self.grid_size, figsize=(10, 10))
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        for ax in axs.flat:
            ax.axis('off')
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                axs[i, j].imshow(cossim[i, j].cpu(), cmap='viridis')
        if path is not None and (path := Path(path)).exists():
            if interpolate:
                path /= "pos_emb_cossim_interp.png"
            else:
                path /= "pos_emb_cossim.png"
            plt.savefig(path, dpi=300)
        plt.close()
        return axs
    

class SinusoidEmbedding(nn.Module):
    def __init__(self, num_tokens: int, dim: int, normalize: bool = True):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim

        position_angle_vec = 1 / torch.pow(100, 2 * (torch.arange(dim) // 2) / dim)
        sinusoid_table = torch.outer(torch.arange(num_tokens), position_angle_vec)
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])

        if normalize:
            sinusoid_table.sub_(sinusoid_table.mean(axis=-1, keepdim=True))
            sinusoid_table.div_(sinusoid_table.std(axis=-1, keepdim=True))

        self.register_buffer("embedding", sinusoid_table)
    
    def forward(self, token_ids):
        return self.embedding[token_ids]

    def __repr__(self):
        return f"SinusoidEmbedding(num_tokens={self.num_tokens}, dim={self.dim})"
    
    def plot(self, path=None):
        fig = plt.figure(figsize=(self.dim / self.num_tokens, 2))
        ax = plt.imshow(self.embedding.cpu(), interpolation="none")
        plt.tight_layout()

        if path is not None and (path := Path(path)).exists():
            path /= "sinusoid_embedding.png"
            plt.savefig(path, dpi=300)
        plt.close()
        return ax


class TransMIL(nn.Module):
    def __init__(self, *, 
        num_classes: int, input_dim: int = 768, dim: int = 512,
        depth: int = 2, heads: int = 8, dim_head: int = 64, mlp_dim: int = 2048,
        pool: str ='cls', dropout: int = 0., emb_dropout: int = 0.,
        use_pos_embedding: bool = False, emb_grid_size: int = 3
    ):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.cls_token = nn.Parameter(torch.randn(dim))

        self.use_pos_embedding = use_pos_embedding
        if use_pos_embedding:
            self.pos_emb = InterpolationEmbedding2d(dim, grid_size=emb_grid_size)
            # sinusoidal embedding used for distinguishing between patches of different slides
            self.slide_emb = SinusoidEmbedding(25, dim)

        self.fc = nn.Sequential(nn.Linear(input_dim, dim, bias=True), nn.GELU())
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, nn.LayerNorm, dropout)

        self.pool = pool
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, lens, slide_ids=None, coords=None, return_last_attn: bool = False):
        # remove unnecessary padding
        max_idx = torch.amax(lens)
        x = x[:, :max_idx]
        b, n, d = x.shape

        # map input sequence to latent space of TransMIL
        x = self.dropout(self.fc(x))

        # apply positional embedding
        # note: feature vectors added during zero padding get assigned the embedding
        # for the top left patch, however they are masked out in the transformer,
        # thus they don't influence the gradient of this embedding vector
        if self.use_pos_embedding:
            assert slide_ids is not None, "Slide ids needed for positional embedding"
            assert coords is not None, "Tile coordinates needed for positional embedding"
            slide_ids, coords =  slide_ids[:, :max_idx], coords[:, :max_idx]
            pos_emb = self.pos_emb(coords)
            slide_emb = self.slide_emb(slide_ids)
            x = x + pos_emb # + slide_emb
            

        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        lens = lens + 1 # account for cls token

        # mask indicating zero padded feature vectors
        mask = None
        if torch.amin(lens) != torch.amax(lens):
            # mask = torch.arange(0, torch.max(lens), dtype=torch.int32, device=x.device).repeat(b, 1) < lens[..., None]
            # mask = mask[:, None, :, None].to(torch.float)
            # mask = ~(mask @ mask.mT).to(torch.bool)
            mask = torch.arange(0, torch.max(lens), dtype=torch.int32, device=x.device).repeat(b, 1) < lens[..., None]
            mask = ~mask[:, None, :].repeat(1, (n + 1), 1)
        
        x, attn_mask = self.transformer(x, mask)

        if mask is not None and self.pool == 'mean':
            x = torch.cumsum(x, dim=1)[torch.arange(b), lens-1]
            x = x / lens[..., None]
        else:
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        
        return (self.mlp_head(x), attn_mask) if return_last_attn else self.mlp_head(x)
