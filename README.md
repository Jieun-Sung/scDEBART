**scDEBART** is a **Transformer-based foundation model** for predicting **perturbation-induced gene expression** from single-cell transcriptomic data. It is **pretrained at scale on 66.6M of expression-change profiles**, learning to predict **log fold-changes (logFC) conditioned on basal expression**. By leveraging **large-scale scVI-denoised data**, scDEBART captures **gene co-regulation across basal states** and generalizes **perturbation responses across cell types, experimental settings, and perturbation modalities**.

> 📌 Figure: overall architecture and training pipeline (pretrain on cellxgene → fine-tune on Perturb-seq).

<p align="center">
  <img src="final_scheme.png" width="900" alt="scDEBART overview"/>
</p>

# Installation

## Option A: Create a conda environment (recommended)
We provide a conda environment specification to ensure reproducibility.

```bash
conda env create -f scDEBART.yml
conda activate scDEBART
```

## Option B: Install required packages via pip
scDEBART is implemented in PyTorch. We use PyTorch 2.5.1 with CUDA 12.1.

### Required packages
```bash
torch==2.5.1
numpy
scipy
pandas
scikit-learn
anndata
scanpy
scvi-tools
umap-learn
transformers
accelerate
datasets
tokenizers
safetensors
huggingface_hub
einops
tqdm
pyyaml
h5py
```

### Optional (recommended for training at scale)
- deepspeed

```bash
pip install deepspeed
```

# Dataset

The dataset is available on Hugging Face Datasets: `Jieun-S/scDEBART`.



# Quickstart

## 1. Pre-train with CellxGene database

1. Download cellxgene data.
```bash
python Preprocess/download_cellxgene_adata.py \
--datadir ./cellxgene
```
2. Calculate DE per dataset & Concatenate DE profiles.
```bash
python Preprocess/cal_de_per_dataset.py \
--datadir ./cellxgene \
--dataset_id <DATASET_ID> \
--outputdir ./cellxgene/processed_de_per_dataset

python Preprocess/concat_de_profiles.py \
--datadir ./cellxgene/processed_de_per_dataset \
--outputdir ./cellxgene/concat_de
```
3. Pretrain scDEBART (DeepSpeed).
```bash
deepspeed --num_gpus <N> Pretrain/pretrain_scDEBART.py \
--input_train_h5_path ./cellxgene/concat_de/all_processed_de_profiles_compressed.h5 \
--outputdir ./cellxgene/pretrain_output \
--deepspeed_config Pretrain/deepspeed_config.json
```
- Output: `./cellxgene/pretrain_output/loss.txt`
- Output: `./cellxgene/pretrain_output/model_weights/model_weights_epoch_<N>.pt`
- Output: `./cellxgene/pretrain_output/best_model_weights_epoch_22.pt`


## 2. Fine-tuning with Perturb-seq data

1. Quality control (logs append to `./perturbseq/<dataset>/filtering_metadata.txt`).
```bash
python Preprocess_perturbseq/quality_control_perturbseq_adata.py \
--dataset Replogle_K562 \
--perturbseq_adata_path ./perturbseq/Replogle_K562/Replogle_K562_perturb_processed.h5ad \
--outputdir ./perturbseq/Replogle_K562
```
- Output: `./perturbseq/Replogle_K562/Replogle_K562_perturb_filtered.h5ad`
- Output: `./perturbseq/Replogle_K562/export_for_seurat/`

2. Calculate perturb score (R).
```bash
Rscript Preprocess_perturbseq/calculate_perturb_score.R \
--dataset Replogle_K562 \
--ncore 50 \
--inputdir ./perturbseq/Replogle_K562/export_for_seurat \
--outputdir ./perturbseq/Replogle_K562/PS_score
```
- Output: `./perturbseq/Replogle_K562/PS_score/perturb_score_per_cell_per_gene.csv`

3. Calculate optimal perturb score cutoff.
```bash
python Preprocess_perturbseq/get_optimal_perturb_score.py \
--dataset Replogle_K562 \
--ncore 16 \
--input_adata ./perturbseq/Replogle_K562/Replogle_K562_perturb_filtered.h5ad \
--outputdir ./perturbseq/Replogle_K562/PS_score
```
- Output: `./perturbseq/Replogle_K562/Replogle_K562_perturb_filtered_umap.h5ad`
- Output: `./perturbseq/Replogle_K562/PS_score/optimal_ps_cutoff_summary.txt`

4. Select close clusters.
```bash
python Preprocess_perturbseq/get_close_cluster.py \
--dataset Replogle_K562 \
--input_adata ./perturbseq/Replogle_K562/Replogle_K562_perturb_filtered_umap.h5ad \
--outputdir ./perturbseq/Replogle_K562
```
- Output: `./perturbseq/Replogle_K562/rawcount_adata_final_selected_cells_only.h5ad`

5. Differential expression (Perturb-seq).
```bash
python Preprocess_perturbseq/cal_de_for_perturbseq.py \
--dataset Replogle_K562 \
--ncore 50 \
--input_adata ./perturbseq/Replogle_K562/rawcount_adata_final_selected_cells_only.h5ad \
--de_outputdir ./perturbseq/Replogle_K562/DE_results
```
- Output: `./perturbseq/Replogle_K562/DE_results/nonfiltered_de.pt`
- Output: `./perturbseq/Replogle_K562/DE_results/de_masked_high_zero_logfc_to_zero.h5`
- Output: `./perturbseq/Replogle_K562/DE_results/hvg_geneids_5000.pkl`

6. Fine-tune scDEBART (DeepSpeed).
```bash
deepspeed --num_gpus <N> Finetune_perturbseq/finetune_perturbseq.py \
--dataset Replogle_K562 \
--de_result_dir ./perturbseq/Replogle_K562/DE_results \
--pretrained_model_dir ./cellxgene/pretrain_output \
--outputdir ./perturbseq/Replogle_K562/finetuned_model/seed_30 \
--deepspeed_config Finetune_perturbseq/deepspeed_config_fortuning.json \
--train_val_test_dict_path ./perturbseq/Replogle_K562/finetuned_model/seed_30/Replogle_K562_GEARS_split_30_0.8_ensemble.pkl \
--num_HVG 5000 \
--pert_type INH
```

### Prerequisites 

#### 1. DE result directory (`--de_result_dir`)
Must contain:
- `de_masked_high_zero_logfc_to_zero.h5`
- `hvg_geneids_{num_HVG}.pkl`

#### 2. Pretrained model directory (`--pretrained_model_dir`)
Must contain:
- `best_pretrained_scDEBART_model_weights.pt`

#### 3. Train/Val/Test split file (`--train_val_test_dict_path`)
- Pickle file with keys: `train`, `val`, `test`
- Each value is a list of **ENSEMBL gene IDs**
- For Norman double perturbations, IDs can be in the form:
  - `ENSG_A+ENSG_B`

#### 4. Number of HVGs (`--num_HVG`)
- Norman dataset: `3079`
- All other datasets: `5000`

### Outputs
- Output: `./perturbseq/Replogle_K562/finetuned_model/seed_30/model_weights`
- Output: `./perturbseq/Replogle_K562/finetuned_model/seed_30/tuning_loss.txt`
- Output: `./perturbseq/Replogle_K562/Replogle_K562_test_gene_perturb_predictions.npz`
