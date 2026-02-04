import os
import argparse
import numpy as np
import pandas as pd
import anndata
import h5py
import torch
import scvi
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
import scanpy as sc


def append_filter_log(log_path, lines):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def convert_pt_to_hdf5(de_res, h5_file_path, target_max_len=20000, batch_size=1000):
    if len(de_res) == 0:
        raise ValueError("Empty DE results. Nothing to convert.")
    max_len = min(target_max_len, max(len(x["gene_ids"]) for x in de_res))
    n = len(de_res)
    with h5py.File(h5_file_path, "w") as f:
        f.attrs["num_samples"] = n
        f.attrs["description"] = "Concatenated DE profiles (padded to fixed length)"
        f.attrs["padded_length"] = max_len
        f.attrs["pad_values"] = "gene_ids:-1, others:0"
        gene_ids_ds = f.create_dataset(
            "gene_ids",
            shape=(n, max_len),
            dtype="int32",
            chunks=(min(batch_size, n), max_len),
            compression="gzip",
            compression_opts=6,
            shuffle=True,
        )
        logfc_ds = f.create_dataset(
            "logfc",
            shape=(n, max_len),
            dtype="float32",
            chunks=(min(batch_size, n), max_len),
            compression="gzip",
            compression_opts=6,
            shuffle=True,
        )
        proba_ds = f.create_dataset(
            "proba_de",
            shape=(n, max_len),
            dtype="float32",
            chunks=(min(batch_size, n), max_len),
            compression="gzip",
            compression_opts=6,
            shuffle=True,
        )
        g1_ds = f.create_dataset(
            "normalized_expr_group1",
            shape=(n, max_len),
            dtype="float32",
            chunks=(min(batch_size, n), max_len),
            compression="gzip",
            compression_opts=6,
            shuffle=True,
        )
        g2_ds = f.create_dataset(
            "normalized_expr_group2",
            shape=(n, max_len),
            dtype="float32",
            chunks=(min(batch_size, n), max_len),
            compression="gzip",
            compression_opts=6,
            shuffle=True,
        )
        g1_nz = f.create_dataset(
            "nonzero_proportion1",
            shape=(n, max_len),
            dtype="float32",
            chunks=(min(batch_size, n), max_len),
            compression="gzip",
            compression_opts=6,
            shuffle=True,
        )
        g2_nz = f.create_dataset(
            "nonzero_proportion2",
            shape=(n, max_len),
            dtype="float32",
            chunks=(min(batch_size, n), max_len),
            compression="gzip",
            compression_opts=6,
            shuffle=True,
        )
        length_ds = f.create_dataset(
            "valid_lengths",
            shape=(n,),
            dtype="int32",
            chunks=(min(batch_size, n),),
            compression="gzip",
            compression_opts=6,
        )
        dsid_ds = f.create_dataset(
            "dataset_id",
            shape=(n,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            compression="gzip",
            compression_opts=6,
        )
        pert_id_ds = f.create_dataset(
            "pertubed_gene_id",
            shape=(n,),
            dtype="int32",
            compression="gzip",
            compression_opts=6,
        )
        for start in tqdm(range(0, n, batch_size), desc="Converting"):
            end = min(start + batch_size, n)
            bsize = end - start
            gene_ids_batch = np.full((bsize, max_len), -1, dtype="int32")
            logfc_batch = np.zeros((bsize, max_len), dtype="float32")
            proba_batch = np.zeros((bsize, max_len), dtype="float32")
            g1_batch = np.zeros((bsize, max_len), dtype="float32")
            g2_batch = np.zeros((bsize, max_len), dtype="float32")
            g1_nz_batch = np.zeros((bsize, max_len), dtype="float32")
            g2_nz_batch = np.zeros((bsize, max_len), dtype="float32")
            lengths = np.zeros((bsize,), dtype="int32")
            dsids = []
            pert_ids = np.zeros((bsize,), dtype="int32")
            for bi, idx in enumerate(range(start, end)):
                rec = de_res[idx]
                L = min(len(rec["gene_ids"]), max_len)
                lengths[bi] = L
                dsids.append(rec["dataset_id"])
                pert_ids[bi] = rec["pertubed_gene_id"]
                gene_ids_batch[bi, :L] = rec["gene_ids"][:L].astype("int32")
                logfc_batch[bi, :L] = rec["logfc"][:L].astype("float32")
                proba_batch[bi, :L] = rec["proba_de"][:L].astype("float32")
                g1_batch[bi, :L] = rec["norm_group1_expr"][:L].astype("float32")
                g2_batch[bi, :L] = rec["norm_group2_expr"][:L].astype("float32")
                g1_nz_batch[bi, :L] = rec["non_zero_prop1"][:L].astype("float32")
                g2_nz_batch[bi, :L] = rec["non_zero_prop2"][:L].astype("float32")
            gene_ids_ds[start:end] = gene_ids_batch
            logfc_ds[start:end] = logfc_batch
            proba_ds[start:end] = proba_batch
            g1_ds[start:end] = g1_batch
            g2_ds[start:end] = g2_batch
            g1_nz[start:end] = g1_nz_batch
            g2_nz[start:end] = g2_nz_batch
            length_ds[start:end] = lengths
            dsid_ds[start:end] = dsids
            pert_id_ds[start:end] = pert_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Perturb-seq dataset name.")
    parser.add_argument("--ncore", type=int, required=True, help="Number of cores for DE computation.")
    parser.add_argument("--input_adata", type=str, required=True, help="Input AnnData for DE.")
    parser.add_argument("--de_outputdir", type=str, required=True, help="Output directory for DE results.")
    args = parser.parse_args()

    os.makedirs(args.de_outputdir, exist_ok=True)
    dataset_root = os.path.abspath(os.path.join(args.de_outputdir, os.pardir))
    log_path = os.path.join(dataset_root, "filtering_metadata.txt")

    adata = anndata.read_h5ad(args.input_adata)
    adata.layers["counts"] = adata.X.copy()
    if "batch_info" not in adata.obs.columns:
        adata.obs["batch_info"] = "batch1"

    scvi.settings.seed = 42
    scvi.model.SCVI.setup_anndata(adata, layer="counts", labels_key="perturb_gene", batch_key="perturb_gene")
    model = scvi.model.SCVI(adata, n_latent=64)
    model.train(
        batch_size=7000,
        max_epochs=200,
        early_stopping=True,
        early_stopping_patience=5,
    )
    model.save(os.path.join(args.de_outputdir, "scvi_model"), overwrite=True, save_anndata=True)

    adata = anndata.read_h5ad(os.path.join(args.de_outputdir, "scvi_model", "adata.h5ad"))
    adata.layers["counts"] = adata.X.copy()
    adata.var["feature_id"] = adata.var_names
    model = scvi.model.SCVI.load(os.path.join(args.de_outputdir, "scvi_model"), adata=adata)

    ens2geneid = pd.read_pickle(os.path.join(".", "cellxgene", "ens2geneid.pkl"))
    ens2idx_dict = {ens: idx for idx, ens in ens2geneid.items()}
    gene_info = pd.read_csv(os.path.join(".", "cellxgene", "human_allgene_info.csv"), sep="\t")

    def compute_pair_de(g1, g2, seed, sampling_each):
        scvi.settings.seed = int(seed)
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        with torch.no_grad():
            df = model.differential_expression(
                groupby="perturb_gene",
                group1=g1,
                group2=g2,
                mode="change",
                delta=0.25,
                all_stats=True,
                silent=True,
                adata=None,
                filter_outlier_cells=False,
                batch_correction=False,
                n_samples_overall=sampling_each,
                batch_size=4096,
            )
        idx1 = adata.obs["perturb_gene"] == g1
        idx2 = adata.obs["perturb_gene"] == g2
        expr1 = model.get_normalized_expression(
            adata=adata,
            indices=idx1,
            library_size=1e4,
            n_samples_overall=sampling_each,
            batch_size=4096,
            return_numpy=True,
        ).mean(axis=0)
        expr2 = model.get_normalized_expression(
            adata=adata,
            indices=idx2,
            library_size=1e4,
            n_samples_overall=sampling_each,
            batch_size=4096,
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
            "seed": int(seed),
            "group1": g1,
            "group2": g2,
            "gene_ids": df["geneID"].values.tolist(),
            "proba_de": df["proba_de"].values.tolist(),
            "bayes_factor": df["bayes_factor"].values.tolist(),
            "scaled_expr_group1": df["scale1"].values.tolist(),
            "scaled_expr_group2": df["scale2"].values.tolist(),
            "normalized_expr_group1": df["expr1"].values.tolist(),
            "normalized_expr_group2": df["expr2"].values.tolist(),
            "lfc_mean": df["lfc_mean"].values.tolist(),
            "lfc_std": df["lfc_std"].values.tolist(),
            "lfc_median": df["lfc_median"].values.tolist(),
            "lfc_min": df["lfc_min"].values.tolist(),
            "lfc_max": df["lfc_max"].values.tolist(),
            "nonzero_proportion1": df["non_zeros_proportion1"].values.tolist(),
            "nonzero_proportion2": df["non_zeros_proportion2"].values.tolist(),
            "is_de_fdr_0.05": df["is_de_fdr_0.05"].values.tolist(),
            "dataset_id": args.dataset,
        }

    clusters = adata.obs["perturb_gene"].unique().tolist()
    non_targeting_clusters = [c for c in clusters if str(c).startswith("non-targeting")]
    other_clusters = [c for c in clusters if not str(c).startswith("non-targeting")]
    pairs = [(g1, g2) for g1 in other_clusters for g2 in non_targeting_clusters]

    sampling_number_each = 1000
    sampling_number_pergene = 50
    seeds = list(range(sampling_number_pergene))
    tasks = [(pair, seed) for pair in pairs for seed in seeds]

    def wrapper(task):
        (g1, g2), seed = task
        if g1 == g2:
            return None
        return compute_pair_de(g1, g2, seed, sampling_number_each)

    with Pool(processes=args.ncore) as pool:
        results = list(tqdm(pool.imap_unordered(wrapper, tasks), total=len(tasks)))

    de = []
    for res in results:
        if res is None:
            continue
        de.append(
            {
                "seed": res["seed"],
                "group1": res["group1"],
                "group2": res["group2"],
                "gene_ids": torch.tensor(res["gene_ids"], dtype=torch.long),
                "proba_de": torch.tensor(res["proba_de"], dtype=torch.float32),
                "bayes_factor": torch.tensor(res["bayes_factor"], dtype=torch.float32),
                "normalized_expr_group1": torch.tensor(res["normalized_expr_group1"], dtype=torch.float32),
                "normalized_expr_group2": torch.tensor(res["normalized_expr_group2"], dtype=torch.float32),
                "lfc_median": torch.tensor(res["lfc_median"], dtype=torch.float32),
                "nonzero_proportion1": torch.tensor(res["nonzero_proportion1"], dtype=torch.float32),
                "nonzero_proportion2": torch.tensor(res["nonzero_proportion2"], dtype=torch.float32),
                "dataset_id": res["dataset_id"],
            }
        )

    nonfiltered_path = os.path.join(args.de_outputdir, "nonfiltered_de.pt")
    torch.save(de, nonfiltered_path)

    def prepare_item(item):
        return {
            "group1": item["group1"],
            "group2": item["group2"],
            "gene_ids": item["gene_ids"].cpu().numpy(),
            "proba_de": item["proba_de"].cpu().numpy(),
            "nonzero_proportion1": item["nonzero_proportion1"].cpu().numpy(),
            "nonzero_proportion2": item["nonzero_proportion2"].cpu().numpy(),
            "lfc_median": item["lfc_median"].cpu().numpy(),
            "normalized_expr_group1": item["normalized_expr_group1"].cpu().numpy(),
            "normalized_expr_group2": item["normalized_expr_group2"].cpu().numpy(),
            "dataset_id": item["dataset_id"],
        }

    def process_de_item(item, nonzero_th=0.3):
        perturbed_gene = item["group1"].split("_")[0]
        perturbed_gene_ens_list = gene_info[gene_info["SYMBOL"] == perturbed_gene]["ENSEMBL"].values
        perturbed_gene_ids = (
            [k for k, v in ens2geneid.items() if v in perturbed_gene_ens_list]
            if len(perturbed_gene_ens_list) > 0
            else []
        )
        if not any(gid in item["gene_ids"] for gid in perturbed_gene_ids):
            return None
        perturbed_gene_ids = [k for k in perturbed_gene_ids if k in item["gene_ids"]]
        perturbed_gene_id = perturbed_gene_ids[0]
        gene_ids = item["gene_ids"]
        logfc = item["lfc_median"].copy()
        proba_de = item["proba_de"]
        group1_expr = item["normalized_expr_group1"]
        group2_expr = item["normalized_expr_group2"]
        nonzero_proportion1 = item["nonzero_proportion1"]
        nonzero_proportion2 = item["nonzero_proportion2"]
        low_nonzero_mask = (nonzero_proportion1 < nonzero_th) | (nonzero_proportion2 < nonzero_th)
        logfc[low_nonzero_mask] = 0.0
        return {
            "dataset_id": item["dataset_id"],
            "pertubed_gene_id": perturbed_gene_id,
            "gene_ids": gene_ids,
            "logfc": logfc,
            "proba_de": proba_de,
            "norm_group1_expr": group1_expr,
            "norm_group2_expr": group2_expr,
            "non_zero_prop1": nonzero_proportion1,
            "non_zero_prop2": nonzero_proportion2,
            "num_genes": len(gene_ids),
        }

    prepared = [prepare_item(item) for item in de]
    results = Parallel(n_jobs=args.ncore)(
        delayed(process_de_item)(item, 0.3) for item in tqdm(prepared, desc="Filtering DE")
    )
    filtered = [r for r in results if r is not None]

    h5_path = os.path.join(args.de_outputdir, "de_masked_high_zero_logfc_to_zero.h5")
    convert_pt_to_hdf5(filtered, h5_path)

    # HVG selection
    ens2geneid_rev = {ens: geneid for geneid, ens in ens2geneid.items()}
    valid_genes = [g for g in adata.var_names if g in ens2geneid_rev.keys()]
    adata_hvg = adata[:, valid_genes].copy()
    if adata_hvg.n_vars <= 5000:
        hvg_geneids = [ens2geneid_rev[ens] for ens in adata_hvg.var_names if ens in ens2geneid_rev]
    else:
        sc.pp.highly_variable_genes(adata_hvg, n_top_genes=5000, flavor="cell_ranger", subset=False)
        adata_hvg = adata_hvg[:, adata_hvg.var["highly_variable"]].copy()
        hvg_geneids = [ens2geneid_rev[ens] for ens in adata_hvg.var_names if ens in ens2geneid_rev]

    hvg_path = os.path.join(args.de_outputdir, "hvg_geneids_5000.pkl")
    with open(hvg_path, "wb") as f:
        import pickle

        pickle.dump(hvg_geneids, f)

    append_filter_log(
        log_path,
        [
            "DE filtering summary",
            f"Total DE pairs: {len(de)}",
            f"Filtered DE profiles: {len(filtered)}",
            "----------------------------------------------------------------------",
        ],
    )


if __name__ == "__main__":
    main()
