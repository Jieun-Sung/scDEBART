import argparse
import os
import time
import pickle

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch

from sklearn.metrics.pairwise import cosine_similarity


def init_logger():
    import logging

    logging.getLogger("scvi").setLevel(logging.ERROR)
    logging.getLogger("scvi._settings").setLevel(logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)


def secure_clustering(adata, dataset_id):
    try:
        import secuer as sr
    except Exception as exc:
        raise ImportError(
            "secuer is required for secure clustering. Install it before running."
        ) from exc

    feature_name = adata.var.get("feature_name", adata.var_names)
    adata.var["mt"] = feature_name.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    pct_mt = adata.obs["pct_counts_mt"]
    total_counts = adata.obs["total_counts"]
    tc_mean = total_counts.mean()
    tc_std = total_counts.std()
    mt_mean = pct_mt.mean()
    mt_std = pct_mt.std()

    tc_min, tc_max = tc_mean - 3 * tc_std, tc_mean + 3 * tc_std
    mt_min, mt_max = mt_mean - 3 * mt_std, mt_mean + 3 * mt_std

    idx = adata.obs["dataset_id"] == dataset_id
    qc_pass = np.full(adata.n_obs, True)
    qc_pass[idx] &= (adata.obs.loc[idx, "total_counts"].between(tc_min, tc_max)) & (
        adata.obs.loc[idx, "pct_counts_mt"].between(mt_min, mt_max)
    )

    filtered_adata = adata[qc_pass].copy()
    filtered_adata = filtered_adata[:, filtered_adata.var["feature_type"] == "protein_coding"]

    sc.pp.normalize_total(filtered_adata)
    sc.pp.log1p(filtered_adata)
    sc.pp.highly_variable_genes(filtered_adata, n_top_genes=2000)
    hvg_mask = filtered_adata.var["highly_variable"].copy()
    hvg_adata = filtered_adata[:, hvg_mask].copy()
    sc.pp.scale(hvg_adata, max_value=10)
    sc.tl.pca(hvg_adata, svd_solver="arpack")

    fea = hvg_adata.obsm["X_pca"]
    res = sr.secuer(fea=fea, Knn=2, multiProcessState=True, num_multiProcesses=70)

    adata_f = adata[adata.obs["soma_joinid"].isin(filtered_adata.obs["soma_joinid"])].copy()
    adata_f = adata_f[:, adata_f.var["feature_type"] == "protein_coding"]
    adata_f.obsm["X_pca"] = hvg_adata.obsm["X_pca"]
    adata_f.obs["sr_clusters"] = res
    adata_f.obs_names_make_unique()
    return adata_f


def train_scvi(adata_f):
    scvi.settings.seed = 42
    adata_f.layers["counts"] = adata_f.X.copy()
    scvi.model.SCVI.setup_anndata(
        adata_f, layer="counts", batch_key="dataset_id", labels_key="cell_type"
    )
    max_ep = 10 if adata_f.shape[0] > 50000 else 20
    model = scvi.model.SCVI(adata_f, n_latent=32)
    model.train(batch_size=512, max_epochs=max_ep, early_stopping=True, early_stopping_patience=5)
    return model


def compute_pair_de(model, adata, g1, g2, ens2idx_dict):
    init_logger()
    df = model.differential_expression(
        groupby="sr_clusters",
        group1=g1,
        group2=g2,
        mode="change",
        delta=0.25,
        all_stats=True,
        silent=True,
        adata=None,
        filter_outlier_cells=False,
        batch_correction=False,
        n_samples_overall=4000,
        batch_size=256,
    )
    idx1 = adata.obs["sr_clusters"] == g1
    idx2 = adata.obs["sr_clusters"] == g2
    expr1 = model.get_normalized_expression(
        adata=adata,
        indices=idx1,
        library_size=1e4,
        n_samples_overall=4000,
        batch_size=256,
        return_numpy=True,
    ).mean(axis=0)
    expr2 = model.get_normalized_expression(
        adata=adata,
        indices=idx2,
        library_size=1e4,
        n_samples_overall=4000,
        batch_size=256,
        return_numpy=True,
    ).mean(axis=0)
    s1 = pd.Series(expr1, index=adata.var.index)
    s2 = pd.Series(expr2, index=adata.var.index)
    df["expr1"] = s1.reindex(df.index).to_numpy()
    df["expr2"] = s2.reindex(df.index).to_numpy()
    df = pd.merge(df, adata.var[["feature_id"]], left_index=True, right_index=True)
    df["geneID"] = df["feature_id"].map(ens2idx_dict)
    df = df.dropna(subset=["geneID"])
    return {
        "group1": g1,
        "group2": g2,
        "gene_ids": torch.tensor(df["geneID"].values, dtype=torch.long),
        "proba_de": torch.tensor(df["proba_de"].values, dtype=torch.float32),
        "bayes_factor": torch.tensor(df["bayes_factor"].values, dtype=torch.float32),
        "scaled_expr_group1": torch.tensor(df["scale1"].values, dtype=torch.float32),
        "scaled_expr_group2": torch.tensor(df["scale2"].values, dtype=torch.float32),
        "normalized_expr_group1": torch.tensor(df["expr1"].values, dtype=torch.float32),
        "normalized_expr_group2": torch.tensor(df["expr2"].values, dtype=torch.float32),
        "lfc_mean": torch.tensor(df["lfc_mean"].values, dtype=torch.float32),
        "lfc_std": torch.tensor(df["lfc_std"].values, dtype=torch.float32),
        "lfC_median": torch.tensor(df["lfc_median"].values, dtype=torch.float32),
        "lfc_min": torch.tensor(df["lfc_min"].values, dtype=torch.float32),
        "lfc_max": torch.tensor(df["lfc_max"].values, dtype=torch.float32),
        "nonzero_proportion1": torch.tensor(df["non_zeros_proportion1"].values, dtype=torch.float32),
        "nonzero_proportion2": torch.tensor(df["non_zeros_proportion2"].values, dtype=torch.float32),
        "is_de_fdr_0.05": torch.tensor(df["is_de_fdr_0.05"].values, dtype=torch.bool),
        "dataset_id": adata.obs["dataset_id"].unique()[0],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute scVI-based DE profiles per dataset."
    )
    parser.add_argument("--datadir", type=str, help="Cellxgene data directory.")
    parser.add_argument("--dataset_id", type=str, help="Cellxgene dataset identifier.")
    parser.add_argument(
        "--outputdir",
        type=str,
        help="Directory to save processed DE results.",
    )
    args = parser.parse_args()

    datadir = os.path.abspath(args.datadir)
    outputdir = os.path.abspath(args.outputdir)
    os.makedirs(outputdir, exist_ok=True)

    adata_path = os.path.join(datadir, args.dataset_id, "adata.h5ad")
    if not os.path.exists(adata_path):
        raise FileNotFoundError(f"AnnData not found: {adata_path}")

    print(f"Loading AnnData from {adata_path}")
    adata = anndata.read_h5ad(adata_path)
    if "feature_id" not in adata.var.columns:
        adata.var["feature_id"] = adata.var_names

    if adata.shape[1] < 1024:
        print(
            f"Skipping dataset {args.dataset_id} with {adata.shape[0]} cells "
            f"and {adata.shape[1]} genes (too few genes)."
        )
        return

    print("Running secure clustering...")
    adata_f = secure_clustering(adata, args.dataset_id)

    print("Training scVI model...")
    model = train_scvi(adata_f)

    print("Calculating scVI latent representation...")
    latent_vec = model.get_latent_representation(batch_size=512)

    cluster_labels = adata_f.obs["sr_clusters"]
    clusters = cluster_labels.unique()
    cluster_centroids = {c: latent_vec[cluster_labels == c].mean(0) for c in clusters}

    centroid_mat = np.vstack([cluster_centroids[c] for c in clusters])
    cosine_mat = cosine_similarity(centroid_mat)
    sim_df = []
    for i, g1 in enumerate(clusters):
        for j, g2 in enumerate(clusters):
            if j > i:
                sim_df.append((g1, g2, cosine_mat[i, j]))
    sim_df = pd.DataFrame(sim_df, columns=["g1", "g2", "similarity"])

    low, high = np.percentile(sim_df["similarity"], [10, 90])
    filtered_pairs = sim_df[
        (sim_df["similarity"] >= low) & (sim_df["similarity"] <= high)
    ]
    if len(filtered_pairs) > 100000:
        filtered_pairs = filtered_pairs.sample(n=100000, random_state=42)
    filtered_pairs = filtered_pairs.reset_index(drop=True)
    all_filtered_pairs = list(zip(filtered_pairs.g1, filtered_pairs.g2))

    ens2geneid_path = os.path.join(datadir, "ens2geneid.pkl")
    ens_list = adata_f.var["feature_id"].tolist()
    if os.path.exists(ens2geneid_path):
        with open(ens2geneid_path, "rb") as f:
            ens2geneid = pd.Series(pickle.load(f))
        ens2idx_dict = {ens: idx for idx, ens in ens2geneid.items()}
        missing = [ens for ens in ens_list if ens not in ens2idx_dict]
        if missing:
            start_idx = max(ens2idx_dict.values()) + 1
            for offset, ens in enumerate(missing):
                ens2idx_dict[ens] = start_idx + offset
            with open(ens2geneid_path, "wb") as f:
                pickle.dump({idx: ens for ens, idx in ens2idx_dict.items()}, f)
    else:
        ens2idx_dict = {ens: idx for idx, ens in enumerate(ens_list)}
        with open(ens2geneid_path, "wb") as f:
            pickle.dump({idx: ens for ens, idx in ens2idx_dict.items()}, f)

    print("Computing differential expression pairs...")
    start = time.time()
    pairwise_de = []
    for g1, g2 in all_filtered_pairs:
        if g1 == g2:
            continue
        de = compute_pair_de(model, adata_f, g1, g2, ens2idx_dict)
        pairwise_de.append(de)
    end = time.time()

    output_path = os.path.join(
        outputdir, f"processed_de_{args.dataset_id}.pt"
    )
    print(f"Saving DE results to {output_path}")
    torch.save(pairwise_de, output_path)
    print(f"Completed in {round(end - start, 2)} seconds.")


if __name__ == "__main__":
    main()
