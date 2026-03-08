Below is a **clean, professional `README.md`** tailored to your current project state. It explains the goal, results, structure, and how to run things. You can paste it directly into `README.md` in your repo root.

---

# Voxie: Text-Conditioned 3D Voxel Generation

Voxie is a deep learning project that learns to **generate 3D voxelized objects conditioned on object categories** using a **3D Variational Autoencoder (VAE)**. The project explores generative modeling of 3D shapes using voxel grids derived from ShapeNet datasets.

The system processes point clouds or meshes into **32×32×32 voxel grids**, trains a neural generative model, and evaluates its ability to reconstruct and generate category-specific objects.

This project was developed for **APS360 – Applied Fundamentals of Deep Learning (University of Toronto)**.

---

# Project Goals

The objective of this project is to:

* Build a pipeline for **3D shape preprocessing and voxelization**
* Implement a **baseline model** for comparison
* Train a **category-conditioned 3D VAE**
* Evaluate generative performance using **Intersection-over-Union (IoU)**
* Compare learned generative models against simple retrieval baselines

---

# Dataset

This project uses the **ShapeNet Part Segmentation Benchmark dataset**.

Dataset characteristics:

* ~16,000 3D models
* 16 object categories
* Each model represented as a **point cloud**
* Converted into **32×32×32 voxel grids**

Categories include:

```
Airplane, Bag, Cap, Car, Chair, Earphone,
Guitar, Knife, Lamp, Laptop, Motorbike,
Mug, Pistol, Rocket, Skateboard, Table
```

---

# Model Overview

## Baseline Model

The baseline model is a **random retrieval baseline**.

Given a category label:

1. Randomly select a training object from the same category
2. Compare its voxel grid to the test sample

This provides a simple reference performance.

Baseline performance:

```
Mean IoU ≈ 0.09
```

---

## Primary Model: Conditional 3D VAE

The primary model is a **3D convolutional Variational Autoencoder** conditioned on object category.

Architecture:

Encoder:

```
Input: 1×32×32×32 voxel grid

Conv3D layers:
32 → 64 → 128 → 256
```

Latent space:

```
Latent dimension = 128
Category embedding = 32
```

Decoder:

```
ConvTranspose3D layers
256 → 128 → 64 → 32 → 1
```

Training objective:

```
Loss = Reconstruction Loss + β * KL Divergence
```

Reconstruction loss uses **weighted binary cross entropy** to handle voxel sparsity.

---

# Results

Evaluation metric:

```
Intersection over Union (IoU)
```

Test results:

| Model              | Mean IoU   |
| ------------------ | ---------- |
| Baseline Retrieval | ~0.091     |
| Conditional VAE    | **~0.238** |

This represents roughly:

```
2.6× improvement over baseline
```

Example per-category performance:

| Category | Mean IoU |
| -------- | -------- |
| Laptop   | 0.368    |
| Airplane | 0.321    |
| Table    | 0.279    |
| Chair    | 0.223    |
| Knife    | 0.132    |

Large structured objects tend to achieve higher IoU than thin or irregular objects.

---

# Project Structure

```
Voxie/
│
├── checkpoints/
│   Trained model weights
│
├── configs/
│   Configuration files
│
├── data/
│   Dataset storage
│
├── scripts/
│
│   preprocess/
│       preprocess_meshes.py
│       preprocess_pointclouds.py
│
│   train/
│       train_vae.py
│
│   eval/
│       evaluate_baseline.py
│       evaluate_vae.py
│       tune_threshold.py
│
│   visualize/
│       visualize_generation.py
│       visualize_voxel.py
│       plot_training_curves.py
│
│   debug/
│       test_dataset.py
│       test_baseline.py
│       test_vae.py
│
├── voxie/
│
│   baselines/
│       random_retrieval.py
│
│   data/
│       voxel_dataset.py
│
│   models/
│       vae3d.py
│
└── README.md
```

---

# Running the Project

## Preprocess Dataset

Convert point clouds into voxel grids:

```bash
python -m scripts.preprocess.preprocess_pointclouds \
    --dataset_root data/Shapenetcore_benchmark \
    --output_dir data/processed_pointcloud \
    --splits_dir data/Shapenetcore_benchmark
```

---

## Train the VAE

```bash
python -m scripts.train.train_vae
```

---

## Evaluate Baseline

```bash
python -m scripts.eval.evaluate_baseline
```

---

## Evaluate VAE

```bash
python -m scripts.eval.evaluate_vae
```

---

## Tune Threshold

```bash
python -m scripts.eval.tune_threshold
```

---

## Visualize Reconstructions and Generations

Example:

```bash
python -m scripts.visualize.visualize_generation \
    --category Chair \
    --threshold 0.6
```

Visualization shows:

```
Ground Truth
Baseline Retrieval
VAE Reconstruction
VAE Generation
```

---

# Example Visualization

The visualization script compares:

* Ground truth voxel shape
* Baseline retrieval result
* VAE reconstruction
* VAE random generation

This helps qualitatively assess generative performance.

---

# Current Progress

✔ Data preprocessing pipeline implemented
✔ Voxel dataset loader implemented
✔ Baseline model implemented
✔ Conditional 3D VAE implemented
✔ Training and evaluation pipeline completed
✔ Visualization tools implemented

Future work could explore:

* Higher voxel resolutions (64³)
* Diffusion-based 3D generation
* Text-conditioned generation using CLIP embeddings

---

# Dependencies

Main libraries:

```
Python 3.10+
PyTorch
NumPy
Matplotlib
tqdm
```

Install dependencies:

```bash
pip install torch numpy matplotlib tqdm
```

---

# Authors

Developed by:

JDas

University of Toronto