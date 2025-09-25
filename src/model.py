import torch
from torch import nn

class SimpleSAINT(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=8, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=256,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.classifier(x)
