import torch
import torch.nn as nn
from torch.nn import functional as F


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
    def __init__(self,  embed_size, heads):
        super(TransformerEncoder, self).__init__()
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
    def __init__(self,  number_patches=(32//4)**2, patch_size=4, heads=8,
                 depth=5, emd_size=256, num_classes=10):
        super(ViT, self).__init__()
        assert number_patches % heads == 0

        self.patch_size = patch_size
        self.heads = heads
        self.num_classes = num_classes

        self.patch_embedding = nn.Linear(
            patch_size*patch_size*3, emd_size,
            bias=True)

        encoder_list = []
        for i in range(depth):
            encoder_list.append(TransformerEncoder(
                emd_size, heads))
        self.encoders = nn.ModuleList(encoder_list)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emd_size))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, number_patches+1, emd_size))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emd_size),
            nn.Linear(emd_size, num_classes)
        )

    def forward(self, x):

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
