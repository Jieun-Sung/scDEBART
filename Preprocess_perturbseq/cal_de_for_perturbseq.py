'''
scrdir="/spstorage/USERS/sung/projects/DEG_Transformer/OSKM_reprogramming/script"
python "${scrdir}/cal_de.py" --dataset liver --ncore 80 --de_outputdir "/spstorage/USERS/sung/projects/DEG_Transformer/OSKM_reprogramming/data/de_output/liver" > cal_de_liver.log 2>&1 &
python "${scrdir}/cal_de.py" --dataset thymus --ncore 80 --de_outputdir "/spstorage/USERS/sung/projects/DEG_Transformer/OSKM_reprogramming/data/de_output/thymus" > cal_de_thymus.log 2>&1 &
python "${scrdir}/cal_de.py" --dataset spleen --ncore 80 --de_outputdir "/spstorage/USERS/sung/projects/DEG_Transformer/OSKM_reprogramming/data/de_output/spleen" > cal_de_spleen.log 2>&1 &
wait

python ${scrdir}/cal_de.py \
--dataset bone_marrow \
--ncore 60 \
--de_outputdir /spstorage/USERS/sung/projects/DEG_Transformer/OSKM_reprogramming/data/de_output/bone_marrow


python ${scrdir}/cal_de.py \
--dataset kidney \
--ncore 60 \
--de_outputdir /spstorage/USERS/sung/projects/DEG_Transformer/OSKM_reprogramming/data/de_output/kidney

'''

import os
import random
import itertools
import torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLAS_NUM_THREADS"] = "1"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import scvi
scvi.settings.dl_num_workers = 0

torch.multiprocessing.set_sharing_strategy('file_system')

import warnings
warnings.filterwarnings("ignore", message="The 'val_dataloader' does not have many workers")
warnings.filterwarnings("ignore", message=r"Category .* has fewer than 3 cells")
warnings.filterwarnings("ignore", message=".*fewer than 3 cells.*")

import argparse
import numpy as np
import pandas as pd
import anndata
import h5py
import scanpy as sc
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed

# ============================================================
# Module-level globals shared with worker processes via fork
# ============================================================
_g_model = None
_g_adata = None
_g_ens2idx_dict = None
_g_gene_info = None
_g_ens2geneid = None
_g_dataset = None
_g_sampling_each = None


def append_filter_log(log_path, lines):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


# ============================================================
# Top-level worker functions (must be picklable for Pool)
# ============================================================
def compute_pair_de(g1, g2, seed):
    scvi.settings.seed = int(seed)
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    with torch.no_grad():
        df = _g_model.differential_expression(
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
            n_samples_overall=_g_sampling_each,
            batch_size=4096,
        )
    idx1 = _g_adata.obs["perturb_gene"] == g1
    idx2 = _g_adata.obs["perturb_gene"] == g2
    expr1 = _g_model.get_normalized_expression(
        adata=_g_adata,
        indices=idx1,
        library_size=1e4,
        n_samples_overall=_g_sampling_each,
        batch_size=4096,
        return_numpy=True,
    ).mean(axis=0)
    expr2 = _g_model.get_normalized_expression(
        adata=_g_adata,
        indices=idx2,
        library_size=1e4,
        n_samples_overall=_g_sampling_each,
        batch_size=4096,
        return_numpy=True,
    ).mean(axis=0)
    s1 = pd.Series(expr1, index=_g_adata.var.index)
    s2 = pd.Series(expr2, index=_g_adata.var.index)
    df["expr1"] = s1.reindex(df.index).to_numpy()
    df["expr2"] = s2.reindex(df.index).to_numpy()
    df = pd.merge(df, _g_adata.var[["feature_id"]], left_index=True, right_index=True)
    df["geneID"] = df["feature_id"].map(_g_ens2idx_dict)
    df = df.dropna(subset=["geneID"])
    return {
        "seed": int(seed),
        "group1": g1,
        "group2": g2,
        "gene_ids": df["geneID"].values.tolist(),
        "proba_de": df["proba_de"].values.tolist(),
        "bayes_factor": df["bayes_factor"].values.tolist(),
        "normalized_expr_group1": df["expr1"].values.tolist(),
        "normalized_expr_group2": df["expr2"].values.tolist(),
        "lfc_median": df["lfc_median"].values.tolist(),
        "nonzero_proportion1": df["non_zeros_proportion1"].values.tolist(),
        "nonzero_proportion2": df["non_zeros_proportion2"].values.tolist(),
        "dataset_id": _g_dataset,
    }


def wrapper(task):
    (g1, g2), seed = task
    if g1 == g2:
        return None
    return compute_pair_de(g1, g2, seed)


# ============================================================
# Post-processing functions (from file 8)
# ============================================================
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
    perturbed_gene_ens_list = _g_gene_info[_g_gene_info["SYMBOL"] == perturbed_gene]["ENSEMBL"].values
    perturbed_gene_ids = (
        [k for k, v in _g_ens2geneid.items() if v in perturbed_gene_ens_list]
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


def process_de_parallel(de, nonzero_th=0.3, n_jobs=None):
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)
    task_args = [(prepare_item(item), nonzero_th) for item in de]
    results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(process_de_item)(*a) for a in tqdm(task_args, desc="Filtering DE")
    )
    processed = [r for r in results if r is not None]
    print(f"Filtered DE results count: {len(processed)}")
    return processed


def convert_pt_to_hdf5(de_res, h5_file_path, target_max_len=20000, batch_size=1000):
    if len(de_res) == 0:
        print("Empty input. Skipping conversion.")
        return
    print("Analyzing lengths...")
    max_len = min(target_max_len, max(len(x["gene_ids"]) for x in de_res))
    n = len(de_res)
    with h5py.File(h5_file_path, "w") as f:
        f.attrs["num_samples"] = n
        f.attrs["description"] = "Concatenated DE profiles (padded to fixed length)"
        f.attrs["padded_length"] = max_len
        f.attrs["pad_values"] = "gene_ids:-1, others:0"
        gene_ids_ds = f.create_dataset("gene_ids", shape=(n, max_len), dtype="int32", chunks=(min(batch_size, n), max_len), compression="gzip", compression_opts=6, shuffle=True)
        logfc_ds = f.create_dataset("logfc", shape=(n, max_len), dtype="float32", chunks=(min(batch_size, n), max_len), compression="gzip", compression_opts=6, shuffle=True)
        proba_ds = f.create_dataset("proba_de", shape=(n, max_len), dtype="float32", chunks=(min(batch_size, n), max_len), compression="gzip", compression_opts=6, shuffle=True)
        g1_ds = f.create_dataset("normalized_expr_group1", shape=(n, max_len), dtype="float32", chunks=(min(batch_size, n), max_len), compression="gzip", compression_opts=6, shuffle=True)
        g2_ds = f.create_dataset("normalized_expr_group2", shape=(n, max_len), dtype="float32", chunks=(min(batch_size, n), max_len), compression="gzip", compression_opts=6, shuffle=True)
        g1_nz = f.create_dataset("nonzero_proportion1", shape=(n, max_len), dtype="float32", chunks=(min(batch_size, n), max_len), compression="gzip", compression_opts=6, shuffle=True)
        g2_nz = f.create_dataset("nonzero_proportion2", shape=(n, max_len), dtype="float32", chunks=(min(batch_size, n), max_len), compression="gzip", compression_opts=6, shuffle=True)
        length_ds = f.create_dataset("valid_lengths", shape=(n,), dtype="int32", chunks=(min(batch_size, n),), compression="gzip", compression_opts=6)
        dsid_ds = f.create_dataset("dataset_id", shape=(n,), dtype=h5py.string_dtype(encoding="utf-8"), compression="gzip", compression_opts=6)
        pert_id_ds = f.create_dataset("pertubed_gene_id", shape=(n,), dtype="int32", compression="gzip", compression_opts=6)
        print("Writing data (padded)...")
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
    print(f"Conversion completed: {h5_file_path}")


def count_genes(item, thresholds=None):
    if thresholds is None:
        thresholds = [np.log2(1.2), np.log2(1.5), np.log2(2)]
    mask = (item["nonzero_proportion1"] > 0.3) & (item["nonzero_proportion2"] > 0.3)
    lfc_abs = item["lfc_median"].abs()
    counts = [int((mask & (lfc_abs > t)).sum().item()) for t in thresholds]
    return {"seed": item["seed"], "group1": item["group1"], "group2": item["group2"], "counts": counts}


def align_genes(d1, d2, vkey1="normalized_expr_group1", vkey2="normalized_expr_group2"):
    g1 = d1["gene_ids"].detach().cpu().numpy()
    g2 = d2["gene_ids"].detach().cpu().numpy()
    map2 = {int(g): i for i, g in enumerate(g2)}
    idx1, idx2 = [], []
    for i, g in enumerate(g1):
        j = map2.get(int(g), None)
        if j is not None:
            idx1.append(i)
            idx2.append(j)
    if len(idx1) < 3:
        return None, None
    v1 = d1[vkey1]
    v2 = d2[vkey2]
    if isinstance(v1, np.ndarray):
        v1 = torch.from_numpy(v1)
    if isinstance(v2, np.ndarray):
        v2 = torch.from_numpy(v2)
    dev = v1.device if torch.is_tensor(v1) else "cpu"
    idx1_t = torch.as_tensor(idx1, dtype=torch.long, device=dev)
    idx2_t = torch.as_tensor(idx2, dtype=torch.long, device=dev)
    if torch.is_tensor(v1) and v1.device != dev:
        v1 = v1.to(dev)
    if torch.is_tensor(v2) and v2.device != dev:
        v2 = v2.to(dev)
    v1 = torch.index_select(v1, 0, idx1_t)
    v2 = torch.index_select(v2, 0, idx2_t)
    return v1, v2


def _safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a = a[m]
    b = b[m]
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def compute_metrics(vec1, vec2):
    vec1 = vec1.detach().float().cpu()
    vec2 = vec2.detach().float().cpu()
    x = vec1.numpy()
    y = vec2.numpy()
    pcc = _safe_corr(x, y)
    rx = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    ry = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort")
    spc = _safe_corr(rx, ry)
    den = (torch.norm(vec1) * torch.norm(vec2)).item()
    cossim = float(torch.dot(vec1, vec2).item() / den) if den != 0 else np.nan
    return pcc, spc, cossim


def compute_top_metrics(vec1, vec2, topn=100):
    vec1 = vec1.detach().float().cpu()
    vec2 = vec2.detach().float().cpu()
    n = min(int(topn), int(vec1.numel()), int(vec2.numel()))
    if n < 1:
        return np.nan, np.nan, np.nan
    idx1_up = torch.argsort(-vec1)[:n]
    idx1_dn = torch.argsort(vec1)[:n]
    idx2_up = torch.argsort(-vec2)[:n]
    idx2_dn = torch.argsort(vec2)[:n]
    union_idx = torch.unique(torch.cat([idx1_up, idx1_dn, idx2_up, idx2_dn]))
    x = vec1[union_idx].numpy()
    y = vec2[union_idx].numpy()
    pcc = _safe_corr(x, y)
    rx = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    ry = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort")
    spc = _safe_corr(rx, ry)
    x_t = vec1[union_idx]
    y_t = vec2[union_idx]
    den = (torch.norm(x_t) * torch.norm(y_t)).item()
    cossim = float(torch.dot(x_t, y_t).item() / den) if den != 0 else np.nan
    return pcc, spc, cossim


def process_pair(i1, i2, topn, de):
    v1, v2 = align_genes(de[i1], de[i2])
    if v1 is None:
        return None
    res_all = compute_metrics(v1, v2)
    res_top = compute_top_metrics(v1, v2, topn)
    return {
        "pair": (i1, i2),
        "pcc_all": res_all[0], "spc_all": res_all[1], "cos_all": res_all[2],
        "pcc_top": res_top[0], "spc_top": res_top[1], "cos_top": res_top[2],
    }


def pair_generator_by_type(de, n_pairs, pair_type):
    group_map = {}
    for i, d in enumerate(de):
        group_map.setdefault(d["group1"], []).append(i)
    nt_idx = [i for i, d in enumerate(de) if d["group2"] == "non-targeting"]
    pert_groups = [g for g in group_map if g != "non-targeting"]
    if pair_type == "ntvsnt":
        if len(nt_idx) < 2:
            return
        for _ in range(n_pairs):
            i1, i2 = random.sample(nt_idx, 2)
            yield (i1, i2)
    elif pair_type == "pertvspert_same":
        all_pairs = []
        for g in pert_groups:
            idxs = group_map[g]
            if len(idxs) < 2:
                continue
            all_pairs.extend(list(itertools.combinations(idxs, 2)))
        random.shuffle(all_pairs)
        for i1, i2 in all_pairs[:n_pairs]:
            yield (i1, i2)
    elif pair_type == "pertvspert_diff":
        all_pert_idx = [i for g in pert_groups for i in group_map[g]]
        if len(all_pert_idx) < 2:
            return
        count = 0
        while count < n_pairs:
            i1, i2 = random.sample(all_pert_idx, 2)
            if de[i1]["group1"] != de[i2]["group1"]:
                yield (i1, i2)
                count += 1


# ============================================================
# Main
# ============================================================

def main():
    global _g_model, _g_adata, _g_ens2idx_dict, _g_gene_info, _g_ens2geneid
    global _g_dataset, _g_sampling_each

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Perturb-seq dataset name.")
    parser.add_argument("--ncore", type=int, required=True, help="Number of cores for DE computation.")
    parser.add_argument("--input_adata", type=str, required=True, help="Input AnnData for DE.")
    parser.add_argument("--de_outputdir", type=str, required=True, help="Output directory for DE results.")
    parser.add_argument("--nonzero_th", type=float, default=0.3, help="Nonzero proportion threshold for masking logFC.")
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

    _g_adata = anndata.read_h5ad(os.path.join(args.de_outputdir, "scvi_model", "adata.h5ad"))
    _g_adata.layers["counts"] = _g_adata.X.copy()
    _g_adata.var["feature_id"] = _g_adata.var_names
    _g_model = scvi.model.SCVI.load(os.path.join(args.de_outputdir, "scvi_model"), adata=_g_adata)

    _g_ens2geneid = pd.read_pickle(os.path.join("/spstorage/USERS/sung/projects/DEG_Transformer/cellxgene", "ens2geneid.pkl"))
    _g_ens2idx_dict = {ens: idx for idx, ens in _g_ens2geneid.items()}
    _g_gene_info = pd.read_csv(os.path.join("/spstorage/USERS/sung/projects/DEG_Transformer/cellxgene", "human_allgene_info.csv"), sep="\t")
    _g_dataset = args.dataset
    _g_sampling_each = 20

    clusters = _g_adata.obs["perturb_gene"].unique().tolist()
    non_targeting_clusters = [c for c in clusters if str(c).startswith("non-targeting")]
    other_clusters = [c for c in clusters if not str(c).startswith("non-targeting")]
    pairs = [(g1, g2) for g1 in other_clusters for g2 in non_targeting_clusters]

    sampling_number_pergene = 10000
    seeds = list(range(sampling_number_pergene))
    tasks = [(pair, seed) for pair in pairs for seed in seeds]

    with Pool(processes=args.ncore) as pool:
        results = list(tqdm(pool.imap_unordered(wrapper, tasks), total=len(tasks)))

    de = []
    for res in results:
        if res is None:
            continue
        de.append({
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
        })

    nonfiltered_path = os.path.join(args.de_outputdir, "nonfiltered_de.pt")
    torch.save(de, nonfiltered_path)

    filtered = process_de_parallel(de, nonzero_th=args.nonzero_th, n_jobs=args.ncore)

    pkl_path = os.path.join(args.de_outputdir, f"{args.dataset}_masked_high_zero_logfc_to_zero_de.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(filtered, f)

    h5_path = os.path.join(args.de_outputdir, "de_masked_high_zero_logfc_to_zero.h5")
    convert_pt_to_hdf5(filtered, h5_path)

    ens2geneid_rev = {ens: geneid for geneid, ens in _g_ens2geneid.items()}
    valid_genes = [g for g in _g_adata.var_names if g in ens2geneid_rev]
    adata_hvg = _g_adata[:, valid_genes].copy()
    all_geneids = [ens2geneid_rev[ens] for ens in adata_hvg.var_names if ens in ens2geneid_rev]
    
    with open(os.path.join(args.de_outputdir, f"all_geneids_{len(all_geneids)}.pkl"), "wb") as f:
        pickle.dump(all_geneids, f)

    if adata_hvg.n_vars <= 5000:
        hvg_geneids = [ens2geneid_rev[ens] for ens in adata_hvg.var_names if ens in ens2geneid_rev]
    else:
        sc.pp.highly_variable_genes(adata_hvg, n_top_genes=5000, flavor="cell_ranger", subset=False)
        adata_hvg = adata_hvg[:, adata_hvg.var["highly_variable"]].copy()
        hvg_geneids = [ens2geneid_rev[ens] for ens in adata_hvg.var_names if ens in ens2geneid_rev]

    with open(os.path.join(args.de_outputdir, "hvg_geneids_5000.pkl"), "wb") as f:
        pickle.dump(hvg_geneids, f)

    append_filter_log(log_path, [
        "DE filtering summary",
        f"Total DE pairs: {len(de)}",
        f"Filtered DE profiles: {len(filtered)}",
        "----------------------------------------------------------------------",
    ])


if __name__ == "__main__":
    main()
