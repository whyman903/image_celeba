import torch
import torch.nn as nn
import torch.nn.functional as F

def _group_norm(c: int, max_groups: int = 32) -> nn.GroupNorm:
    g = min(max_groups, c)
    while c % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, c)

class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            _group_norm(ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            _group_norm(ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
    def forward(self, x):
        return x + self.block(x) * (1 / (2 ** 0.5))         # residual scaling to keep activation variance stable


class Encoder(nn.Module):
    def __init__(self, in_ch: int = 3, z_dim: int = 64):
        super().__init__()
        ch = 64
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1),
            ResBlock(ch), ResBlock(ch),
            nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1),
            ResBlock(ch * 2), ResBlock(ch * 2),
            nn.Conv2d(ch * 2, ch * 4, 4, stride=2, padding=1),
            ResBlock(ch * 4), ResBlock(ch * 4),
            nn.Conv2d(ch * 4, z_dim, 1),
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, z_dim: int = 64, out_ch: int = 3):
        super().__init__()
        ch = 256
        self.net = nn.Sequential(
            nn.Conv2d(z_dim, ch, 3, padding=1),
            ResBlock(ch), ResBlock(ch),
            nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1),
            ResBlock(ch // 2), ResBlock(ch // 2),
            nn.ConvTranspose2d(ch // 2, ch // 4, 4, stride=2, padding=1),
            ResBlock(ch // 4), ResBlock(ch // 4),
            nn.Conv2d(ch // 4, out_ch, 3, padding=1),
            nn.Tanh(),
        )
    def forward(self, z):
        return self.net(z)

class VectorQuantizer(nn.Module):
    def __init__(self, num_codebook: int, dim: int, beta: float = 0.25):
        super().__init__()
        self.codebook = nn.Embedding(num_codebook, dim)
        self.beta = beta
        nn.init.normal_(self.codebook.weight, mean=0.0, std=1.0)

    def forward(self, z):
        # z: [B, C, H, W]
        B, C, H, W = z.shape
        z_perm = z.permute(0, 2, 3, 1).contiguous()     # [B, H, W, C]
        z_flat = z_perm.view(-1, C)                     # [BHW, C]
        e = self.codebook.weight                        # [K, C]

        # Squared Euclidean distance: ||z||^2 - 2 z·e + ||e||^2
        dist = (z_flat.pow(2).sum(1, keepdim=True)
                - 2 * torch.einsum('nc,kc->nk', z_flat, e)
                + e.pow(2).sum(1))                     # [BHW, K]

        idx = dist.argmin(dim=1)                        # [BHW]
        z_q = e[idx].view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Codebook + commitment losses (VQ-VAE)
        codebook_loss = F.mse_loss(z_q, z.detach())
        commit_loss   = self.beta * F.mse_loss(z, z_q.detach())
        vq_loss = codebook_loss + commit_loss

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        with torch.no_grad():
            one_hot = F.one_hot(idx, e.size(0)).float().mean(0)  # [K]
            probs = one_hot.clamp_min(1e-10)
            perplexity = torch.exp(-(probs * probs.log()).sum())
        return z_q, vq_loss, perplexity

class VQVAE(nn.Module):
    def __init__(self, num_codebook: int = 1024, dim: int = 64):
        super().__init__()
        self.encoder = Encoder(z_dim=dim)
        self.vq = VectorQuantizer(num_codebook, dim)
        self.decoder = Decoder(z_dim=dim)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, perplexity = self.vq(z)
        recon = self.decoder(z_q)
        return recon, vq_loss, perplexity
