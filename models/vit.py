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
        return feed_forward_out

    def fold_conv(self, x):
        batch_size, seq_len, channels = x.shape
        side_length = int(seq_len ** 0.5)  # 4
        x = x.view(batch_size, side_length, side_length, channels)
        x = x.permute(0, 3, 1, 2)  # Change to (batch_size, channels, height, width)
        return x
    
    def unfold_conv(self, x):
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1)  # Change to (batch_size, height, width, channels)
        x = x.view(batch_size, height * width, channels)
        return x

class ViT(nn.Module):
    def __init__(self,  number_patches=16, patch_size=8, heads=8, num_classes=10):
        super(ViT, self).__init__()
        assert number_patches % heads == 0

        self.transformer_input_channels = [64, 128, 256, 512, 512]

        self.patch_size = patch_size
        self.heads = heads
        self.num_classes = num_classes
        self.pre_conv_steps = 3  # 32=>16;16=>8;8=>4
        channels = self.transformer_input_channels[0]
        for step in range(self.pre_conv_steps):
            conv = nn.Conv2d(3 if step == 0 else channels, channels,
                             kernel_size=3, stride=1, padding=1)
            bn = nn.BatchNorm2d(channels)
            max_pool = nn.MaxPool2d(2)
            self.add_module(f"conv{step}", conv)
            self.add_module(f"bn{step}", bn)
            self.add_module(f"max_pool{step}", max_pool)

        
        encoder_list = []
        for i in range(len(self.transformer_input_channels)-2):
            encoder_list.append(TransformerEncoder(
                self.transformer_input_channels[i],
                self.transformer_input_channels[i+1], heads))
        self.encoders = nn.ModuleList(encoder_list)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, number_patches, self.transformer_input_channels[0]))
        
        self.post_convs=nn.Sequential(
            nn.Conv2d(self.transformer_input_channels[-1], 128,
                             kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.transformer_input_channels[-1]),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
        )

        self.fc = nn.Linear(
            self.transformer_input_channels[-1]*number_patches, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = x
        for step in range(self.pre_conv_steps):
            out = self.get_submodule(f"conv{step}")(out)
            out = self.get_submodule(f"bn{step}")(out)
            out = F.relu(out)
            out = F.dropout(out, 0.2)
            out = self.get_submodule(f"max_pool{step}")(out)

        # out = out.view(out.size(0), self.transformer_input_channels[0], -1)
        out = self.unfold_conv(out)
        # Transpose the tensor to change dimensions from (batch_size, channels, length) to (batch_size, length, channels)
        out = out.transpose(1, 2)

        out = out + self.pos_embedding
        for encoder in self.encoders:
            out = encoder(out)
            out = F.relu(out, 0.2)

        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    
    
     def fold_conv(self, x):
        batch_size, seq_len, channels = x.shape
        side_length = int(seq_len ** 0.5)  # 4
        x = x.view(batch_size, side_length, side_length, channels)
        x = x.permute(0, 3, 1, 2)  # Change to (batch_size, channels, height, width)
        return x
    
    def unfold_conv(self, x):
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1)  # Change to (batch_size, height, width, channels)
        x = x.view(batch_size, height * width, channels)
        return x
