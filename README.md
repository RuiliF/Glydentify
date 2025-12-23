# Glydentify

**Glydentify** is a deep learning framework for predicting glycosyltransferase donor substrates using structure-aware protein language models and molecular representations. This repository contains the code and resources for the paper submitted to *Nature Communications*.

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
    git clone https://github.com/yourusername/glydentify_public.git
    cd glydentify_public
    ```
2.  **Environment Setup:**
    It is recommended to use the provided `glydentify` Conda environment.
    ```bash
    conda create -n glydentify python=3.10
    conda activate glydentify
    pip install -r requirements.txt
    ```

3.  **Install Foldseek:**
    Please download the `foldseek` binary and place it in the `bin/` directory.
    You can download it from this [Google Drive Link](https://drive.google.com/file/d/1B_9t3n_nlj8Y3Kpc_mMjtMdY0OPYa7Re/view?usp=sharing).
    *Note: This structural encoding approach is based on [SaProt](https://github.com/westlake-repl/SaProt).*

## Usage

### Training
To train the model (supports SaProt, ESM2, ESM-C):
```bash
python scripts/train.py --fold <dataset_fold_name> --model_type <saprot|esm2|esmc> --train_unimol --train_seq_encoder --batch_size 16
```
Arguments:
- `--fold`: Name of the dataset fold (expected in `data/<fold>/` or `../data/<fold>`).
- `--model_type`: Model architecture (`saprot`, `esm2`, `esmc`). Default: `saprot`.
- `--train_unimol`: Fine-tune the UniMol encoder.
- `--train_seq_encoder`: Fine-tune the sequence encoder (SaProt/ESM).

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
python scripts/annotate.py <input_folder> --checkpoint_path <path_to_checkpoint> --target_donor <donor_name>
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code or data, please cite our paper:
> [Citation Placeholder]
