import torch
import torch.nn as nn

class Ncf(nn.Module):
    def __init__(self, numusers: int, numitems: int, embeddingdim: int = 32):
        super().__init__()
        self.userembed = nn.Embedding(num_embeddings=numusers, embedding_dim=embeddingdim)
        self.itemembed = nn.Embedding(num_embeddings=numitems, embedding_dim=embeddingdim)
        self.fc1 = nn.Linear(embeddingdim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.outlayer = nn.Linear(32, 1)

    def forward(self, userindices: torch.Tensor, itemindices: torch.Tensor) -> torch.Tensor:
        uservec = self.userembed(userindices)
        itemvec = self.itemembed(itemindices)
        vec = torch.cat([uservec, itemvec], dim=-1)
        x = torch.relu(self.fc1(vec))
        x = torch.relu(self.fc2(x))
        out = torch.sigmoid(self.outlayer(x))
        return out.squeeze()
