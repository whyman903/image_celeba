
import argparse, os, math, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as tv_utils
import matplotlib.pyplot as plt


def _group_norm(c: int, max_groups: int = 32) -> nn.GroupNorm:
    g = min(max_groups, c)
    while c % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, c)


class Encoder(nn.Module):

    def __init__(self, latent_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 64×64×64
            _group_norm(64), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),  # 128×32×32
            _group_norm(128), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),  # 256×16×16
            _group_norm(256), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),  # 512×8×8
            _group_norm(512), nn.LeakyReLU(0.2, inplace=True),
        )
        self.flatten_dim = 512 * 8 * 8
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):

    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.up = nn.Sequential(
            # 512×8×8 -> 256×16×16
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(512, 256, 3, 1, 1),
            _group_norm(256), nn.ReLU(True),

            # 256×16×16 -> 128×32×32
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, 3, 1, 1),
            _group_norm(128), nn.ReLU(True),

            # 128×32×32 -> 64×64×64
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, 1, 1),
            _group_norm(64), nn.ReLU(True),

            # 64×64×64 -> 3×128×128
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 512, 8, 8)
        return self.up(h)


class VAE(nn.Module):

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar




def vae_loss(x, recon, mu, logvar, beta: float = 1.0, loss_function: str = "mse"):
    if loss_function == "mse":
        recon_loss = F.mse_loss(recon, x, reduction="mean")
    elif loss_function == "l1":
        recon_loss = F.l1_loss(recon, x, reduction="mean")
    else:
        raise ValueError(f"Invalid loss function: {loss_function}")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss.detach(), kl.detach()




def save_samples(model: VAE, epoch: int, out_dir: Path, device):
    model.eval()
    with torch.no_grad():
        z = torch.randn(64, model.encoder.fc_mu.out_features, device=device)
        samples = model.decoder(z)
        grid = tv_utils.make_grid(samples, nrow=8, normalize=True, value_range=(-1, 1))
        sample_dir = out_dir / "samples"
        sample_dir.mkdir(exist_ok=True, parents=True)
        tv_utils.save_image(grid, sample_dir / f"sample_epoch_{epoch:03d}.png")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    dataset = datasets.CelebA(args.data_root, split="train", download=args.download, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    # Model
    model = VAE(latent_dim=args.latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-5)

    recon_losses = []
    kl_losses = []
    total_losses = []

    global_step = 0
    last_ckpt_epoch = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_recon, epoch_kl = 0.0, 0.0
        t0 = time.time()
        for x, _ in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            # KL warm-up: linearly ramp beta over warmup epochs
            if args.kl_warmup_epochs > 0:
                beta_t = args.beta * min(1.0, epoch / float(args.kl_warmup_epochs))
            else:
                beta_t = args.beta
            loss, recon_loss, kl_loss = vae_loss(x, recon, mu, logvar, beta_t, args.loss_function)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            epoch_recon += recon_loss.item() * x.size(0)
            epoch_kl += kl_loss.item() * x.size(0)
            global_step += 1

        n = len(loader.dataset)
        epoch_recon_avg = epoch_recon / n
        epoch_kl_avg = epoch_kl / n
        if args.kl_warmup_epochs > 0:
            beta_epoch = args.beta * min(1.0, epoch / float(args.kl_warmup_epochs))
        else:
            beta_epoch = args.beta
        epoch_total = epoch_recon_avg + beta_epoch * epoch_kl_avg
        
        # Store losses
        recon_losses.append(epoch_recon_avg)
        kl_losses.append(epoch_kl_avg)
        total_losses.append(epoch_total)

        print(f"Epoch {epoch}/{args.epochs} | beta: {beta_epoch:.3f} | recon: {epoch_recon / n:.4f} | kl: {epoch_kl / n:.4f} | time: {time.time() - t0:.1f}s")

        # Save checkpoint and samples at requested frequency, first epoch, and final epoch
        if epoch % args.chkpt_freq == 0 or epoch == 1 or epoch == args.epochs:
            ckpt_path = Path(args.out_dir) / f"vae_epoch_{epoch:03d}.pt"
            torch.save({"epoch": epoch, "state_dict": model.state_dict()}, ckpt_path)
            save_samples(model, epoch, Path(args.out_dir), device)
            last_ckpt_epoch = epoch

    final_path = Path(args.out_dir) / "vae_last.pt"
    torch.save({"epoch": args.epochs, "state_dict": model.state_dict()}, final_path)

    plt.figure(figsize=(10, 6))
    epochs = range(1, args.epochs + 1)
    plt.plot(epochs, recon_losses, label='Reconstruction Loss')
    plt.plot(epochs, kl_losses, label='KL Loss')
    plt.plot(epochs, total_losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training Losses')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_path = Path(args.out_dir) / 'training_losses.png'
    plt.savefig(plot_path)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="CelebA VAE trainer")
    p.add_argument("--data_root", default="data", help="Path to CelebA root directory")
    p.add_argument("--out_dir", default="chkpts", help="Directory for logs/ckpts")
    p.add_argument("--download", action="store_true", help="Download CelebA if absent")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--latent", type=int, default=256)
    p.add_argument("--beta", type=float, default=5.0, help="KL weight")
    p.add_argument("--kl_warmup_epochs", type=int, default=10, help="Epochs to linearly ramp KL weight to --beta (0 disables)")
    p.add_argument("--loss_function", type=str, default="mse", choices=["mse", "l1"], help="Reconstruction loss")
    p.add_argument("--chkpt_freq", type=int, default=25)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    train(args)
