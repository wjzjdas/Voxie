import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from voxie.data.voxel_dataset import VoxelDataset
from voxie.models.vae3d import VoxelVAE3D, vae_loss


def run_epoch(model, loader, device, optimizer=None, beta=0.01):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        pbar = tqdm(loader, leave=False)
        for batch in pbar:
            x = batch["voxel"].to(device)
            category_idx = batch["category_idx"].to(device)

            if is_train:
                optimizer.zero_grad()

            logits, mu, logvar = model(x, category_idx)
            loss, stats = vae_loss(logits, x, mu, logvar, beta=beta)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += stats["loss"]
            total_recon += stats["recon_loss"]
            total_kl += stats["kl_loss"]
            num_batches += 1

            pbar.set_description(
                f"{'train' if is_train else 'val'} "
                f"loss {stats['loss']:.4f} recon {stats['recon_loss']:.4f}"
            )

    return {
        "loss": total_loss / num_batches,
        "recon_loss": total_recon / num_batches,
        "kl_loss": total_kl / num_batches,
    }


def main():
    os.makedirs("checkpoints", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = VoxelDataset(
        index_path="data/processed_pointcloud/index.jsonl",
        split="train",
    )
    val_dataset = VoxelDataset(
        index_path="data/processed_pointcloud/index.jsonl",
        split="val",
        category_to_idx=train_dataset.category_to_idx,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = VoxelVAE3D(
        num_categories=len(train_dataset.category_to_idx),
        latent_dim=128,
        category_embed_dim=32,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    beta = 0.01
    best_val_loss = float("inf")

    print("device:", device)
    print("train samples:", len(train_dataset))
    print("val samples:", len(val_dataset))
    print(sum(p.numel() for p in model.parameters()))

    history = {
        "train_loss": [],
        "train_recon_loss": [],
        "train_kl_loss": [],
        "val_loss": [],
        "val_recon_loss": [],
        "val_kl_loss": [],
    }
    
    for epoch in range(epochs):

        train_stats = run_epoch(model, train_loader, device, optimizer=optimizer, beta=beta)
        val_stats = run_epoch(model, val_loader, device, optimizer=None, beta=beta)
        
        history["train_loss"].append(train_stats["loss"])
        history["train_recon_loss"].append(train_stats["recon_loss"])
        history["train_kl_loss"].append(train_stats["kl_loss"])
        history["val_loss"].append(val_stats["loss"])
        history["val_recon_loss"].append(val_stats["recon_loss"])
        history["val_kl_loss"].append(val_stats["kl_loss"])

        with open("checkpoints/history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
            
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_stats['loss']:.4f} recon {train_stats['recon_loss']:.4f} | "
            f"val loss {val_stats['loss']:.4f} recon {val_stats['recon_loss']:.4f}"
        )

        torch.save(model.state_dict(), f"checkpoints/vae_epoch_{epoch}.pt")

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save(model.state_dict(), "checkpoints/vae_best.pt")
            print("Saved new best model.")


if __name__ == "__main__":
    main()