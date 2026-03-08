import json
import matplotlib.pyplot as plt


def main():
    with open("checkpoints/history.json", "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = list(range(len(history["train_loss"])))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_recon_loss"], label="Train Recon")
    plt.plot(epochs, history["val_recon_loss"], label="Val Recon")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("VAE Reconstruction Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()