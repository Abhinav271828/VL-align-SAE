import torch
from torch import nn

class SAE(nn.Module):
    def __init__(self, txt_dim, img_dim, hidden_dim, k):
        super().__init__()

        self.k = k

        self.img_encoder = nn.Sequential(nn.Linear(img_dim, hidden_dim), nn.ReLU())
        self.txt_encoder = nn.Sequential(nn.Linear(txt_dim, hidden_dim), nn.ReLU())

        self.img_decoder = nn.Linear(hidden_dim, img_dim)
        self.txt_decoder = nn.Linear(hidden_dim, txt_dim)

    def encode_image(self, x):
        latent = self.img_encoder(x)

        #values, indices = torch.topk(latent, self.k)
        #latent = torch.zeros_like(latent)
        #latent.scatter_(-1, indices, values)

        return latent
    
    def decode_image(self, x):
        return self.img_decoder(x)

    def encode_text(self, x):
        latent = self.txt_encoder(x)

        #values, indices = torch.topk(latent, self.k)
        #latent = torch.zeros_like(latent)
        #latent.scatter_(-1, indices, values)

        return latent
    
    def decode_text(self, x):
        return self.txt_decoder(x)