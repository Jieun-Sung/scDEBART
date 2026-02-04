import os
import argparse
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
from tqdm import tqdm
from scipy import sparse
from scipy.io import mmwrite


def append_filter_log(log_path, lines):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Perturb-seq dataset name.")
    parser.add_argument(
        "--perturbseq_adata_path",
        type=str,
        required=True,
        help="Input AnnData path for perturb-seq.",
    )
    parser.add_argument(
        "--outputdir",
        type=str,
        required=True,
        help="Output directory for filtered AnnData and export files.",
    )
    args = parser.parse_args()

    os.makedirs(args.outputdir, exist_ok=True)
    log_path = os.path.join(args.outputdir, "filtering_metadata.txt")

    adata = anndata.read_h5ad(args.perturbseq_adata_path)

    # 1) QC filtering by total counts and percent mitochondrial counts (3 std from mean)
    if "mt" not in adata.var.columns:
        if "gene_name" in adata.var.columns:
            adata.var["mt"] = adata.var["gene_name"].str.upper().str.startswith("MT-")
        else:
            adata.var["mt"] = adata.var.index.str.upper().str.startswith("MT-")

    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    pct_mt = adata.obs["pct_counts_mt"]
    total_counts = adata.obs["total_counts"]
    tc_mean, tc_std = total_counts.mean(), total_counts.std()
    mt_mean, mt_std = pct_mt.mean(), pct_mt.std()
    tc_min, tc_max = tc_mean - 3 * tc_std, tc_mean + 3 * tc_std
    mt_min, mt_max = mt_mean - 3 * mt_std, mt_mean + 3 * mt_std

    qc_pass = (adata.obs["total_counts"].between(tc_min, tc_max)) & (
        adata.obs["pct_counts_mt"].between(mt_min, mt_max)
    )
    filtered_adata = adata[qc_pass].copy()

    append_filter_log(
        log_path,
        [
            "Filtering by 3 standard deviations from the mean of total counts and percent mitochondrial counts",
            f"Total cells before filtering: {adata.n_obs}",
            f"Total cells after filtering: {filtered_adata.n_obs}",
            f"Total genes before filtering: {adata.n_vars}",
            f"Total genes after filtering: {filtered_adata.n_vars}",
            "----------------------------------------------------------------------",
        ],
    )

    # 2) Select perturbed cells with target genes in var and zero expression for target gene
    perturb_genes_in_var = list(
        set(filtered_adata.obs["perturb_gene"]).intersection(
            set(filtered_adata.var.get("gene_name", filtered_adata.var_names))
        )
    )
    control_name = list(
        set(["non-targeting", "control"]).intersection(set(filtered_adata.obs["perturb_gene"]))
    )[0]

    target_list = perturb_genes_in_var + [control_name]
    filtered_adata = filtered_adata[filtered_adata.obs["perturb_gene"].isin(target_list)].copy()

    append_filter_log(
        log_path,
        [
            f"{len(perturb_genes_in_var)} / {filtered_adata.obs.perturb_gene.nunique()} perturbation genes found in var, so only processing these genes",
            f"Total cells after selecting perturbations with genes in var: {filtered_adata.n_obs}",
            "----------------------------------------------------------------------",
        ],
    )

    zero_cells = []
    gene_name_col = "gene_name" if "gene_name" in filtered_adata.var.columns else None

    for cell, row in tqdm(filtered_adata.obs.iterrows(), total=filtered_adata.n_obs):
        gene = row["perturb_gene"]
        if gene in ["non-targeting", "control"]:
            continue
        if gene_name_col:
            if gene not in filtered_adata.var[gene_name_col].values:
                continue
            gene_id = filtered_adata.var.index[filtered_adata.var[gene_name_col] == gene][0]
        else:
            if gene not in filtered_adata.var_names:
                continue
            gene_id = gene
        expr_value = filtered_adata[cell, gene_id].X
        if sparse.issparse(expr_value):
            expr_value = expr_value.toarray().flatten()[0]
        else:
            expr_value = np.array(expr_value).flatten()[0]
        if expr_value == 0:
            zero_cells.append(cell)

    final_filtered_adata = filtered_adata[filtered_adata.obs_names.isin(zero_cells)].copy()

    append_filter_log(
        log_path,
        [
            f"Total cells with zero expression of their target gene: {final_filtered_adata.n_obs}",
            f"Final number of perturb_genes: {final_filtered_adata.obs.perturb_gene.nunique() - 1} (excluding control)",
            "----------------------------------------------------------------------",
        ],
    )

    # Save filtered AnnData
    filtered_path = os.path.join(args.outputdir, f"{args.dataset}_perturb_filtered.h5ad")
    final_filtered_adata.write_h5ad(filtered_path)

    # Export for Seurat
    export_dir = os.path.join(args.outputdir, "export_for_seurat")
    os.makedirs(export_dir, exist_ok=True)

    mtx = final_filtered_adata.X
    if sparse.issparse(mtx):
        mmwrite(os.path.join(export_dir, "matrix.mtx"), mtx)
    else:
        mmwrite(os.path.join(export_dir, "matrix.mtx"), sparse.csr_matrix(mtx))

    final_filtered_adata.obs.to_csv(os.path.join(export_dir, "metadata.csv"))
    final_filtered_adata.var.to_csv(os.path.join(export_dir, "features.csv"))


if __name__ == "__main__":
    main()
