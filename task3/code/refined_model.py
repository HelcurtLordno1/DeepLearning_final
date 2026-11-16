"""A small spatial CNN encoder and seven-step GRU decoder, warm-start compatible."""
import torch
from torch import nn
from utils import dot_attention


class RefinedPlateSeq2Seq(nn.Module):
    def __init__(self, classes, config):
        super().__init__()
        h = config['hidden_size']
        self.attention = True
        self.alignment = config['alignment']
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, h, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.refinement = nn.Sequential(
            nn.Conv2d(h, h, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(h, h, 3, padding=1)
        )
        # The new residual branch starts at zero, retaining the old features.
        nn.init.zeros_(self.refinement[-1].weight)
        nn.init.zeros_(self.refinement[-1].bias)
        self.height_projection = nn.Linear(8*h, h)
        with torch.no_grad():
            self.height_projection.weight.copy_(torch.eye(h).repeat(1, 8)/8)
            self.height_projection.bias.zero_()
        self.column_positions = nn.Parameter(torch.zeros(1, 48, h))
        self.character_regions = nn.AdaptiveAvgPool1d(7)
        self.decoder = nn.GRU(h, h, batch_first=True)
        self.output = nn.Linear(2*h, classes)

    def forward(self, X, return_attention=False):
        features = self.cnn(X)                         # [B, H, 8, 48]
        features = features + .1*self.refinement(features)
        columns = features.permute(0, 3, 2, 1).flatten(2)
        memory = self.height_projection(columns) + self.column_positions
        context = memory.mean(dim=1)                  # [B, H]
        regions = self.character_regions(memory.transpose(1, 2)).transpose(1, 2)
        decoder_input = (1-self.alignment)*context[:, None] + self.alignment*regions
        decoded, _ = self.decoder(decoder_input, context[None])
        attended, weights = dot_attention(decoded, memory)
        logits = self.output(torch.cat([decoded, attended], dim=-1))
        if return_attention:
            return logits, weights
        return logits
