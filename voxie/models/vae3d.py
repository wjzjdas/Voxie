from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelVAE3D(nn.Module):
    def __init__(
        self,
        num_categories: int,
        latent_dim: int = 128,
        category_embed_dim: int = 32,
    ) -> None:
        super().__init__()

        self.num_categories = num_categories
        self.latent_dim = latent_dim
        self.category_embed_dim = category_embed_dim

        # -------------------------
        # Category embedding
        # -------------------------
        self.category_embedding = nn.Embedding(num_categories, category_embed_dim)

        # -------------------------
        # Encoder
        # Input: [B, 1, 32, 32, 32]
        # Output spatial sizes:
        # 32 -> 16 -> 8 -> 4 -> 2
        # -------------------------
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),   # -> [B, 32, 16, 16, 16]
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),  # -> [B, 64, 8, 8, 8]
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1), # -> [B, 128, 4, 4, 4]
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),# -> [B, 256, 2, 2, 2]
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

        self.encoder_out_dim = 256 * 2 * 2 * 2

        self.fc_mu = nn.Linear(self.encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_out_dim, latent_dim)

        # -------------------------
        # Decoder
        # latent + category embedding
        # -------------------------
        self.decoder_input = nn.Linear(latent_dim + category_embed_dim, self.encoder_out_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),  # -> [B, 128, 4, 4, 4]
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),   # -> [B, 64, 8, 8, 8]
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),    # -> [B, 32, 16, 16, 16]
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),     # -> [B, 1, 32, 32, 32]
            # no sigmoid here; use BCEWithLogitsLoss
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, category_idx: torch.Tensor) -> torch.Tensor:
        cat_emb = self.category_embedding(category_idx)  # [B, category_embed_dim]
        z_cat = torch.cat([z, cat_emb], dim=1)
        h = self.decoder_input(z_cat)
        h = h.view(-1, 256, 2, 2, 2)
        logits = self.decoder(h)
        return logits

    def forward(
        self,
        x: torch.Tensor,
        category_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, category_idx)
        return logits, mu, logvar

def vae_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    pos_weight_value: float = 15.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    VAE loss = weighted reconstruction + beta * KL

    pos_weight_value > 1 increases the penalty for missing occupied voxels.
    This helps prevent the model from collapsing to mostly empty predictions.
    """
    pos_weight = torch.tensor(
        [pos_weight_value],
        device=logits.device,
        dtype=logits.dtype,
    )

    recon_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="mean",
        pos_weight=pos_weight,
    )

    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )

    total_loss = recon_loss + beta * kl_loss

    stats = {
        "loss": float(total_loss.item()),
        "recon_loss": float(recon_loss.item()),
        "kl_loss": float(kl_loss.item()),
        "beta": float(beta),
        "pos_weight": float(pos_weight_value),
    }
    return total_loss, stats