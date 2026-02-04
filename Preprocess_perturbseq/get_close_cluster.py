import os
import argparse
import numpy as np
import pandas as pd
import anndata
from scipy.spatial.distance import cdist


def append_filter_log(log_path, lines):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def read_best_cutoff(summary_path, default=0.0):
    if not os.path.exists(summary_path):
        return default
    with open(summary_path, "r") as f:
        for line in f:
            if line.startswith("best_cutoff"):
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        return default
    return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Perturb-seq dataset name.")
    parser.add_argument("--input_adata", type=str, required=True, help="Input AnnData with UMAP.")
    parser.add_argument("--outputdir", type=str, required=True, help="Output directory for selected cells.")
    args = parser.parse_args()

    os.makedirs(args.outputdir, exist_ok=True)
    log_path = os.path.join(args.outputdir, "filtering_metadata.txt")

    adata = anndata.read_h5ad(args.input_adata)

    ps_score_path = os.path.join(args.outputdir, "PS_score", "perturb_score_per_cell_per_gene.csv")
    ps_score = pd.read_csv(ps_score_path, index_col=0)
    if "PS" not in ps_score.columns:
        raise ValueError("PS column not found in perturb_score_per_cell_per_gene.csv")

    adata.obs = adata.obs.merge(ps_score.loc[:, ["PS"]], left_index=True, right_index=True, how="left")
    adata.obs["PS"] = adata.obs["PS"].fillna(0.0)
    if "perturb_gene" not in adata.obs.columns and "perturb_gene" in ps_score.columns:
        adata.obs = adata.obs.merge(
            ps_score.loc[:, ["perturb_gene"]], left_index=True, right_index=True, how="left"
        )

    summary_path = os.path.join(args.outputdir, "PS_score", "optimal_perturb_score_cutoff_summary.txt")
    best_cutoff = read_best_cutoff(summary_path, default=0.0)

    if "X_pca" in adata.obsm:
        latent = adata.obsm["X_pca"]
    elif "X_umap" in adata.obsm:
        latent = adata.obsm["X_umap"]
    else:
        raise ValueError("Input AnnData must contain X_pca or X_umap in obsm.")

    if "perturb_gene" not in adata.obs.columns:
        raise ValueError("perturb_gene column not found in AnnData obs.")

    control_mask = adata.obs["perturb_gene"] == "non-targeting"
    control_latent = latent[control_mask.values]
    control_centroid = control_latent.mean(axis=0)

    control_dist = cdist(control_latent, control_centroid.reshape(1, -1), metric="euclidean").flatten()
    control_spread_quantile = 0.7
    control_core_mask = control_dist <= np.quantile(control_dist, control_spread_quantile)
    control_core_names = adata.obs.index[control_mask.values][control_core_mask]

    gene_spread_quantile = 0.9
    taken_mask = np.zeros(adata.n_obs, dtype=bool)
    taken_mask[adata.obs.index.isin(control_core_names)] = True

    for gene in adata.obs["perturb_gene"].unique():
        if gene == "non-targeting":
            continue
        gene_mask = adata.obs["perturb_gene"] == gene
        gene_idx = np.where(gene_mask.values)[0]
        if gene_idx.size == 0:
            continue
        ps_mask = adata.obs.loc[gene_mask, "PS"] >= best_cutoff
        if ps_mask.sum() == 0:
            continue
        gene_latent = latent[gene_mask.values][ps_mask.values]
        gene_centroid = gene_latent.mean(axis=0)
        dist_to_gene = cdist(gene_latent, gene_centroid.reshape(1, -1), metric="euclidean").flatten()
        dist_to_control = cdist(gene_latent, control_centroid.reshape(1, -1), metric="euclidean").flatten()
        keep = dist_to_gene < dist_to_control
        if keep.sum() == 0:
            continue
        gene_spread = float(np.quantile(dist_to_gene[keep], gene_spread_quantile))
        if gene_spread <= float(np.mean(dist_to_gene) + 2 * np.std(dist_to_gene)) and keep.sum() >= 5:
            kept_indices = gene_idx[ps_mask.values][keep]
            taken_mask[kept_indices] = True

    taken_adata = adata[taken_mask].copy()
    output_path = os.path.join(args.outputdir, "rawcount_adata_final_selected_cells_only.h5ad")
    taken_adata.write_h5ad(output_path)

    append_filter_log(
        log_path,
        [
            "Close cluster selection based on latent space and perturb score",
            f"Total cells before selection: {adata.n_obs}",
            f"Total cells after selection: {taken_adata.n_obs}",
            f"Selection ratio: {taken_adata.n_obs / adata.n_obs:.2%}",
            f"Best cutoff used: {best_cutoff:.2f}",
            "----------------------------------------------------------------------",
        ],
    )


if __name__ == "__main__":
    main()
