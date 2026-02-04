import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from sklearn.metrics import silhouette_score


def append_filter_log(log_path, lines):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Perturb-seq dataset name.")
    parser.add_argument("--ncore", type=int, default=1, help="Number of cores for silhouette scoring.")
    parser.add_argument("--input_adata", type=str, required=True, help="Filtered AnnData path.")
    parser.add_argument(
        "--outputdir",
        type=str,
        required=True,
        help="Output directory for perturb score results (PS_score).",
    )
    args = parser.parse_args()

    os.makedirs(args.outputdir, exist_ok=True)
    dataset_root = os.path.abspath(os.path.join(args.outputdir, os.pardir))
    log_path = os.path.join(dataset_root, "filtering_metadata.txt")

    adata = anndata.read_h5ad(args.input_adata)

    ps_score_path = os.path.join(args.outputdir, "perturb_score_per_cell_per_gene.csv")
    ps_score = pd.read_csv(ps_score_path, index_col=0)
    if "PS" not in ps_score.columns:
        raise ValueError("PS column not found in perturb_score_per_cell_per_gene.csv")

    adata.obs = adata.obs.merge(ps_score.loc[:, ["PS"]], left_index=True, right_index=True, how="left")
    adata.obs["PS"] = adata.obs["PS"].fillna(0.0)
    if "perturb_gene" not in adata.obs.columns and "perturb_gene" in ps_score.columns:
        adata.obs = adata.obs.merge(
            ps_score.loc[:, ["perturb_gene"]], left_index=True, right_index=True, how="left"
        )
        adata = adata[~adata.obs["perturb_gene"].isna()].copy()

    if "X_umap" not in adata.obsm:
        sc.pp.neighbors(adata, n_neighbors=15, use_rep="X", random_state=42)
        sc.tl.umap(adata, random_state=42)

    umap_path = os.path.join(dataset_root, f"{args.dataset}_perturb_filtered_umap.h5ad")
    adata.write_h5ad(umap_path)

    cutoff_range = np.arange(0.0, 0.8, 0.05)
    results = []
    coords = adata.obsm["X_umap"]

    for cutoff in cutoff_range:
        mask = (adata.obs["PS"] >= cutoff) | (adata.obs["perturb_gene"] == "non-targeting")
        if mask.sum() < 5:
            continue
        labels = adata.obs.loc[mask, "perturb_gene"].astype(str).values
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(coords[mask.values], labels, metric="euclidean")
        survival_ratio = float(mask.sum() / adata.n_obs)
        results.append((float(cutoff), float(score), survival_ratio))

    if not results:
        best_cutoff = 0.0
        best_score = 0.0
        best_survival = 0.0
    else:
        best_cutoff, best_score, best_survival = max(results, key=lambda x: x[1])

    summary_path = os.path.join(args.outputdir, "optimal_perturb_score_cutoff_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"best_cutoff\t{best_cutoff:.2f}\n")
        f.write(f"best_silhouette\t{best_score:.4f}\n")
        f.write(f"survival_ratio\t{best_survival:.4f}\n")

    append_filter_log(
        log_path,
        [
            "Optimal perturb score cutoff selection",
            f"best_cutoff: {best_cutoff:.2f}",
            f"best_silhouette: {best_score:.4f}",
            f"survival_ratio: {best_survival:.4f}",
            "----------------------------------------------------------------------",
        ],
    )


if __name__ == "__main__":
    main()
