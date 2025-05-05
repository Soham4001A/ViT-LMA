# vit_lma.py (Modified version incorporating LMA)

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Import your LMA components (assuming they are in 'latent.py' in the same directory/path)
try:
    from .latent import InitialLatentTransform, LatentLayer
except ImportError:
    # Fallback if running as a script and latent.py is in the same dir
    from .latent import InitialLatentTransform, LatentLayer

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    # Unchanged from original
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    # Unchanged from original
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    # Unchanged from original (This is standard MHA, won't be used by LMA ViT)
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
     # Unchanged from original (Standard Transformer Encoder, won't be used by LMA ViT)
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
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
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
        
# --------------------------------------------------------------------------
# LMA ViT Implementation
# --------------------------------------------------------------------------
class ViT_LMA(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, lma_cfg, pool = 'cls', channels = 3, dropout = 0., emb_dropout = 0.):
        """
        ViT using pure latent LMA encoder with inverse transform before head.

        Args:
            image_size, patch_size, num_classes, dim, depth: Standard ViT parameters.
            lma_cfg (dict or OmegaConf): Configuration object for LMA containing:
                - d_new (int): Latent dimension.
                - num_heads_stacking (int): Heads for stacking (Stage 2a).
                - num_heads_latent (int): Heads for latent attention.
                - target_l_new (int, optional): Target latent sequence length.
                - ff_latent_hidden (int): Hidden dim for latent FFN.
                - qkv_bias (bool): Bias for linear layers in LMA.
                - dropout_prob (float): Dropout within LMA layers (inherited from main dropout).
                - Optional: norm (str), norm_eps (float) if different norm needed for latent space.
            pool, channels, dropout, emb_dropout: Standard ViT parameters.
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        seq_len = num_patches + 1 # Account for CLS token

        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # --- LMA Integration ---
        # Add necessary info to lma_cfg if missing
        lma_cfg.static_seq_len = seq_len
        lma_cfg.dropout_prob = dropout # Use main dropout for LMA internal dropout
        if not hasattr(lma_cfg, 'qkv_bias'):
             lma_cfg.qkv_bias = True # Default bias if not specified

        # 1. Initial Transform H -> Z
        self.latent_front = InitialLatentTransform(dim, lma_cfg) # expects dim=hidden_size
        latent_dim = lma_cfg.d_new

        # 2. Latent Transformer Layers
        self.latent_layers = nn.ModuleList([
            LatentLayer(
                d_new=latent_dim,
                nh_latent=lma_cfg.num_heads_latent,
                ff_hidden=lma_cfg.ff_latent_hidden,
                dropout=dropout, # Pass main dropout rate
                bias=lma_cfg.qkv_bias,
            )
            for _ in range(depth)
        ])

        # 3. Final Norm in Latent Space (Applied before inverse transform)
        # Use standard LayerNorm for simplicity, adjust if needed via config
        latent_norm_eps = getattr(lma_cfg, 'norm_eps', 1e-5)
        self.final_latent_norm = nn.LayerNorm(latent_dim, eps=latent_norm_eps)
        # --- End LMA Integration ---

        self.pool = pool
        self.to_latent = nn.Identity() # Original ViT field, kept for compatibility? Usually identity.

        # MLP Head remains the same (operates on original `dim`)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        # Optional: Add weight initialization if needed

    def forward(self, img):
        # 1. Patching and Embedding (Standard ViT)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape # n = num_patches

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1) # [B, n+1, dim]
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x) # Input dropout

        # 2. LMA Encoder Path
        # Transform H -> Z, get latent mask (likely None for ViT)
        # Input mask=None assumes no padding within the patch sequence
        z, latent_mask = self.latent_front(x, mask=None) # [B, L_new, d_new]

        # Pass through Latent Layers
        for blk in self.latent_layers:
            z = blk(z, latent_mask) # latent_mask is likely None

        # Apply final norm in latent space
        z = self.final_latent_norm(z)

        # 3. Inverse Transform Z -> H (Reconstruct token-level features)
        x_reconstructed = self.latent_front.inverse_transform(z) # [B, n+1, dim]

        # 4. Pooling / CLS Token Extraction (Standard ViT, but on reconstructed features)
        x_pooled = x_reconstructed.mean(dim = 1) if self.pool == 'mean' else x_reconstructed[:, 0]

        # 5. Final Classification Head (Standard ViT)
        x_final = self.to_latent(x_pooled) # Usually identity
        return self.mlp_head(x_final)