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
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
        )

    def forward(self, x):
        x = self.attention(x) + x
        x = self.norm1(x)
        x = self.mlp(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, embed_size=256,   heads=4):
        super(ViT, self).__init__()
        self.encoder1 = TransformerEncoder(embed_size, heads)
        self.encoder2 = TransformerEncoder(embed_size, heads)
        self.encoder3 = TransformerEncoder(embed_size, heads)
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pos_embedding = nn.Embedding(16, 256)
        self.fc1 = nn.Linear(256*16, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # in: 3,64,64, out: 32,64,64
        out = self.max_pool(out)  # in: 32,64,64, out: 32,32,32 
        out = F.relu(self.bn2(self.conv2(out)))  # in: 32,32,32, out: 64,32,32  
        out = self.max_pool(out)  # in: 64,32,32, out: 64,16,16
        out = F.relu(self.bn3(self.conv3(out)))  # in: 64,16,16, out: 128,16,16
        out = self.max_pool(out)  # in: 128,16,16, out: 128,8,8
        out = F.relu(self.bn4(self.conv4(out)))  # in: 128,8,8, out: 256,8,8
        out = self.max_pool(out)  # in: 256,8,8, out: 256,4,4

        out = out.view(out.size(0), 256, -1)
        # Transpose the tensor to change dimensions from (batch_size, 256, 16) to (batch_size, 16, 256)
        out = out.transpose(1, 2)
        out = out + self.pos_embedding(torch.arange(16,
                                                    device=x.device))
        out = self.encoder1(out)
        out = self.encoder2(out)
        out = self.encoder3(out)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out
