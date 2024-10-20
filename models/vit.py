import torch
import torch.nn as nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, embed_size, heads):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads

        self.qkv_proj = nn.Linear(embed_size, 3 * embed_size)
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
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads):
        super(TransformerEncoder, self).__init__()
        self.attention = Attention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
     

    def forward(self, x):
        x = self.attention(x) + x
        x = self.norm1(x)
        return x


class ViT(nn.Module):
    def __init__(self, cnn_model_cls, input_channels=256, input_len=16, heads=4, num_classes=10):
        super(ViT, self).__init__()
        assert input_len % heads == 0
        self.input_channels = input_channels
        self.input_len = input_len
        self.heads = heads
        self.num_classes = num_classes
        self.cnn_model = cnn_model_cls()
        self.encoder1 = TransformerEncoder(input_channels, heads)
        self.encoder2 = TransformerEncoder(input_channels, heads)
        self.encoder3 = TransformerEncoder(input_channels, heads)

        self.pos_embedding = nn.Embedding(input_len, input_channels)
        self.fc1 = nn.Linear(input_channels*input_len, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.cnn_model(x)

        out = out.view(out.size(0), self.input_channels, -1)
        # Transpose the tensor to change dimensions from (batch_size, channels, length) to (batch_size, length, channels)
        out = out.transpose(1, 2)
        out = out + self.pos_embedding(torch.arange(self.input_len,
                                                    device=x.device))
        out = self.encoder1(out)
        out = out + self.encoder2(out)
        out = out + self.encoder3(out)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out
