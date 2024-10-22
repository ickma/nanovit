import torch
import torch.nn as nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, input_channels, embed_size, heads):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads

        self.qkv_proj = nn.Linear(input_channels, 3 * embed_size)
        self.o_proj = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv_proj(x).view(
            batch_size, seq_len, self.heads, 3 * self.embed_size // self.heads)
        q, k, v = qkv.chunk(3, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * (self.embed_size ** (-1/2))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).view(batch_size, seq_len, self.embed_size)
        out = self.o_proj(out)
        # out = F.relu(out)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, input_channels, embed_size, heads):
        super(TransformerEncoder, self).__init__()
        self.attention = Attention(input_channels, embed_size, heads)
        self.norm1 = nn.LayerNorm(input_channels)
        # convert x to embed_size for skip connection
        self.fc = nn.Linear(input_channels, embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        out = self.attention(self.norm1(x))
        out = self.fc(x)+out
        feed_forward_out = self.feed_forward(out)
        out = feed_forward_out + out
        return out


class ViT(nn.Module):
    def __init__(self,  number_patches=16, patch_size=8, heads=8, num_classes=10):
        super(ViT, self).__init__()
        assert number_patches % heads == 0

        self.transformer_input_channels = [320, 64, 128, 256, 320,320]

        self.patch_size = patch_size
        self.heads = heads
        self.num_classes = num_classes

        self.patch_embedding = nn.Linear(
            patch_size*patch_size*3, self.transformer_input_channels[0],
            bias=True)

        encoder_list = []
        for i in range(len(self.transformer_input_channels)-2):
            encoder_list.append(TransformerEncoder(
                self.transformer_input_channels[i],
                self.transformer_input_channels[i+1], heads))
        self.encoders = nn.ModuleList(encoder_list)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, number_patches, self.transformer_input_channels[0]))

        self.fc = nn.Linear(
            self.transformer_input_channels[-1]*number_patches, num_classes, bias=True)

    def forward(self, x):

     
        patches = self.gen_patches(x)

        out = self.patch_embedding(patches)

        out = out + self.pos_embedding
        for encoder in self.encoders:
            out = encoder(out)
        
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

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
