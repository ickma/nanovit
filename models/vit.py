import torch
import torch.nn as nn
from torch.nn import functional as F


class LinearAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(LinearAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, key_padding_mask=None):
        """
        query, key, value: (batch_size, seq_len, embed_dim)
        key_padding_mask: (batch_size, seq_len) -> True for padding tokens
        """
        batch_size, seq_len, _ = x.size()

        # Linear projections
        Q = self.q_proj(x)  # (batch_size, seq_len, embed_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        # (batch_size, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Permute to get dimensions: (batch_size, num_heads, seq_len, head_dim)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Apply the feature map (e.g., elu + 1)
        Q = F.elu(Q) + 1  # (batch_size, num_heads, seq_len, head_dim)
        K = F.elu(K) + 1

        if key_padding_mask is not None:
            # key_padding_mask: (batch_size, seq_len) -> (batch_size, 1, seq_len, 1)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(3)
            K = K.masked_fill(mask, 0)
            V = V.masked_fill(mask, 0)

        # Compute KV^T: (batch_size, num_heads, head_dim, head_dim)
        KV = torch.einsum('bnhd,bnhe->bnde', K, V)

        # Compute normalizer: Z = Q sum(K)
        K_sum = K.sum(dim=2)  # (batch_size, num_heads, head_dim)
        Z = 1 / (torch.einsum('bnhd,bnd->bnh', Q, K_sum) + 1e-6)
        Z = Z.unsqueeze(-1)  # (batch_size, num_heads, seq_len, 1)

        # Compute output: (batch_size, num_heads, seq_len, head_dim)
        output = torch.einsum('bnhd,bnde->bnhe', Q, KV)
        output = output * Z

        # Reshape and project
        # (batch_size, seq_len, num_heads, head_dim)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        return output


class Attention(nn.Module):
    def __init__(self,  head_dim, num_heads):
        super(Attention, self).__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads

        self.q_proj = nn.Linear(head_dim, head_dim)
        self.k_proj = nn.Linear(head_dim, head_dim)
        self.v_proj = nn.Linear(head_dim, head_dim)
        self.o_proj = nn.Linear(head_dim, head_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads,
                   self.head_dim//self.num_heads)
        q = q.transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads,
                   self.head_dim//self.num_heads)
        k = k.transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads,
                   self.head_dim//self.num_heads)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(
            q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate heads and project
        attn_output = attn_output.transpose(
            1, 2).contiguous().view(batch_size, seq_len, self.head_dim)
        output = self.o_proj(attn_output)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self,  embed_size, heads, linear_attn=False):
        super(TransformerEncoder, self).__init__()
        if linear_attn:
            self.attention = LinearAttention(embed_size, heads)
        else:
            self.attention = Attention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # convert x to embed_size for skip connection
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        out = x+self.attention(self.norm1(x))
        out = self.norm2(out)
        out = out+self.feed_forward(out)
        return out


class ViT(nn.Module):
    def __init__(self,  img_size=32,
                 patch_size=2,
                 heads=8,
                 depth=4,
                 emd_size=240,
                 num_classes=10,
                 linear_attn=False,
                 linear_proj=True,
                 cnn_proj=False):
        super(ViT, self).__init__()
        number_patches = (img_size//patch_size)**2
        assert emd_size % heads == 0

        self.patch_size = patch_size
        self.heads = heads
        self.num_classes = num_classes
        self.linear_attn = linear_attn

        self.cnn_proj = cnn_proj
        self.linear_proj = linear_proj
        if cnn_proj:
            self.patch_embedding = nn.Conv2d(
                3, emd_size, kernel_size=patch_size, stride=patch_size)
        if linear_proj:
            self.patch_embedding = nn.Linear(
                patch_size*patch_size*3, emd_size,
                bias=True)

        encoder_list = []
        for i in range(depth):
            encoder_list.append(TransformerEncoder(
                emd_size, heads, linear_attn))
        self.encoders = nn.ModuleList(encoder_list)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emd_size))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, number_patches+1, emd_size))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emd_size),
            nn.Linear(emd_size, num_classes)
        )

    def forward(self, x):
        if self.cnn_proj:
            out = self.patch_embedding(x)

            # Change to (batch, h, w, dim)
            out = out.permute(0, 2, 3, 1).contiguous()
            # Flatten to (batch, h*w, dim)
            out = out.view(x.size(0), -1, out.size(-1))
        elif self.linear_proj:
            patches = self.gen_patches(x)
            out = self.patch_embedding(patches)

        out = torch.cat([self.cls_token.repeat(
            out.size(0), 1, 1), out], dim=1)

        out = out + self.pos_embedding
        for encoder in self.encoders:
            out = encoder(out)

        logits = self.mlp_head(out[:, 0])
        return logits

    def gen_patches(self, x):
        # Example input with batch size (batch_size, channels, height, width)
        batch_size = x.size(0)  # Example batch size
        # Step 1: Extract 8x8 patches using unfold
        # Output shape: (batch_size, 3, 4, 4, 8, 8)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size)

        # Step 2: Reshape the patches so that each 8x8 patch is flattened into a 1D vector
        patches = patches.contiguous().view(batch_size, 3, -1,
                                            self.patch_size * self.patch_size)  # Shape: (batch_size, 3, 16, 64)

        # Step 3: Flatten the channel and spatial dimensions together
        patches = patches.permute(0, 2, 1, 3).contiguous().view(
            batch_size, -1, 3 * self.patch_size * self.patch_size)  # Shape: (batch_size, 16, 192)
        return patches.to(x.device)
