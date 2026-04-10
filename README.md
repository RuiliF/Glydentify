# Glydentify

**Glydentify** is a deep learning framework for predicting glycosyltransferase donor substrates using structure-aware protein language models and molecular representations. This repository contains the code and resources for the paper submitted to _Communications Biology_.

## Repository Structure

```
glydentify_public/
├── src/                  # Core package source code
│   ├── model.py          # GTDonorPredictor model definition
│   ├── dataset.py        # Dataset loading and processing
│   ├── trainer.py        # Training and evaluation logic
│   ├── losses.py         # Custom loss functions (e.g., Asymmetric Loss)
│   └── utils.py          # Utility functions (including foldseek utils)
├── scripts/              # Executable scripts
│   ├── train.py          # Main training script
│   ├── inference.py      # Inference on a folder of PDB/CIF files
│   └── annotate.py       # Annotate structures with attention weights
├── data/                 # Datasets (e.g., gta/train.csv, gtb/test.csv)
├── checkpoints/          # Model checkpoints (e.g., saprot_unimol/gta/)
├── bin/                  # Directory for external binaries (Focus is on Foldseek)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RuiliF/Glydentify.git
    cd Glydentify
    ```
2.  **Environment Setup:**

    ```bash
    conda create -n glydentify python=3.10
    conda activate glydentify
    ```

    It is recommended to install PyTorch manually first to ensure the correct CUDA version for your hardware. Run the command below (or check the [official PyTorch website](https://pytorch.org/get-started/locally/) for your system):

    ```bash
    # Example for Linux with CUDA 12.9
    pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 torchaudio==2.8.0+cu129 --index-url https://download.pytorch.org/whl/cu129
    ```
    > ⚠️ **Note on reproducibility** : Prediction scores are sensitive to **PyTorch versions**. Running inference with a different PyTorch version than specified may produce ***substantially different absolute probability scores***, even on identical input files. Note that not all CUDA versions are compatible with the PyTorch version required — if you encounter installation errors, we recommend using our [Docker deployment instructions](https://github.com/RuiliF/Glydentify/edit/main/README.md#docker-setup-optional-but-recommended) below.
    
    Install rest of the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Foldseek:**
    To use SaProt-UniMol version, please download the `foldseek` binary and place it in the `bin/` directory.
    You can download it from this [Google Drive Link](https://drive.google.com/file/d/1B_9t3n_nlj8Y3Kpc_mMjtMdY0OPYa7Re/view?usp=sharing).
    _Note: This structural encoding approach is based on [SaProt](https://github.com/westlake-repl/SaProt)._

## Docker Setup (Optional but Recommended)

You can run Glydentify inside a Docker container for reproducibility and to easily manage dependencies. Before building the Docker image, **ensure you have downloaded the `foldseek` binary and placed it in the `bin/` directory as described above.**

### 1. Prerequisites (Host Machine)

To utilize GPUs inside Docker, your host machine must have:

- **Docker** installed.
- **NVIDIA GPU Drivers** installed and running.
- **NVIDIA Container Toolkit** installed to bridge your host GPU to Docker.
  - _Ubuntu/Debian installation:_ `sudo apt install nvidia-container-toolkit` followed by `sudo systemctl restart docker`.
  - _See the [official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for more OS options._

### 2. Building the Image

From the root of the repository, build the Docker image (this will bake the `checkpoints/` directly into the image):

```bash
docker build -t glydentify .
```

### 3. Running with Docker

When running the container, use the `--gpus all` flag to allow the container to access your host GPUs. Also, mount your local `data/` folder so the dataloader can read your datasets.

**Example for Training:**

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  glydentify python scripts/train.py --fold <gta|gtb> --model_type <saprot|esm2|esmc> --batch_size 16
```

**Example for Inference:**

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  glydentify python scripts/inference.py data/gta/test.csv --checkpoint checkpoints/saprot_unimol/gta/
```

## Local Usage

### Training

To train the model (supports SaProt, ESM2, ESM-C):

```bash
python scripts/train.py --fold <gta|gtb> --model_type <saprot|esm2|esmc> --batch_size 16
```

Arguments:

- `--fold`: Name of the dataset fold (expected in `data/<fold>/` or `../data/<fold>`).
- `--model_type`: Model architecture (`saprot`, `esm2`, `esmc`). Default: `saprot`.
- `--train_unimol` (optional): Fine-tune the UniMol encoder.
- `--train_seq_encoder` (optional): Fine-tune the sequence encoder (SaProt/ESM).

### Inference

To run inference on a folder of protein structures (`.pdb` or `.cif`) using a trained checkpoint:

```bash
python scripts/inference.py <input_folder> --checkpoint <path_to_checkpoint> --model_type <saprot|esm2|esmc>
```

or evaluate on the test set:

```bash
python scripts/inference.py --checkpoint checkpoints/saprot_unimol/gta/ data/gta/test.csv
```

### Structure Annotation

To visualize attention weights on the protein structure:

```bash
python scripts/annotate.py <input_folder> --checkpoint <path_to_checkpoint> --target_donor <donor_name>
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code or data, please cite our paper:

```bibtex
@article {Fang2026.03.13.711695,
	author = {Fang, Ruili and Na, Lan and Corulli, Charles J. and Prabhakar, Pradeep K. and Berardinelli, Steven J and Venkat, Aarya and Prasad, Anup and Mahmud, Rezwan and Moremen, Kelley W. and Urbanowicz, Breeanna R. and Dou, Fei and Kannan, Natarajan},
	title = {Glydentify: An explainable deep learning platform for glycosyltransferase donor substrate prediction},
	elocation-id = {2026.03.13.711695},
	year = {2026},
	doi = {10.64898/2026.03.13.711695},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/03/17/2026.03.13.711695},
	eprint = {https://www.biorxiv.org/content/early/2026/03/17/2026.03.13.711695.full.pdf},
	journal = {bioRxiv}
}
```
> This citation reflects the current bioRxiv preprint. A peer-reviewed version is under consideration; citation will be updated upon acceptance.

