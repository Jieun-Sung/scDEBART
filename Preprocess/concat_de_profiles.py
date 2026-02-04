import argparse
import os
import time
from multiprocessing import Pool

import h5py
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Concatenate per-dataset DE profiles into a single HDF5 file."
    )
    parser.add_argument("--datadir", type=str, default="./cellxgene/processed_de_per_dataset")
    parser.add_argument("--outputdir", type=str, default="./cellxgene/concat_de")
    parser.add_argument("--topk", type=int, default=1024)
    parser.add_argument("--min_genes", type=int, default=10)
    parser.add_argument("--proba_de_th", type=float, default=0.8)
    parser.add_argument("--nonzero_th", type=float, default=0.3)
    parser.add_argument("--fc_th", type=float, default=1.5)
    parser.add_argument("--ncore", type=int, default=40)
    return parser.parse_args()


def _get_key(item, *keys):
    for key in keys:
        if key in item:
            return item[key]
    raise KeyError(f"None of the keys {keys} were found in the DE record.")


def process_file(path, topk, min_genes, proba_de_th, nonzero_th, fc_th):
    try:
        de = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"Failed to load {path}: {exc}")
        return []

    processed = []
    for item in de:
        proba_de = _get_key(item, "proba_de")
        nonzero1 = _get_key(item, "nonzero_proportion1")
        nonzero2 = _get_key(item, "nonzero_proportion2")
        logfc = _get_key(item, "lfC_median", "logfc", "lfc_median")

        mask = (
            (proba_de >= proba_de_th)
            & (nonzero1 >= nonzero_th)
            & (nonzero2 >= nonzero_th)
            & (logfc.abs() >= np.log2(fc_th))
        )
        if mask.sum() < min_genes:
            continue

        filtered_logfc = logfc[mask]
        topn = min(len(filtered_logfc), topk)
        topk_idx = torch.topk(filtered_logfc.abs(), topn).indices

        gene_ids = _get_key(item, "gene_ids")[mask][topk_idx].cpu().numpy()
        logfc_vals = logfc[mask][topk_idx].cpu().numpy()
        proba_vals = proba_de[mask][topk_idx].cpu().numpy()
        group1_expr = _get_key(item, "normalized_expr_group1")[mask][topk_idx].cpu().numpy()
        group2_expr = _get_key(item, "normalized_expr_group2")[mask][topk_idx].cpu().numpy()

        processed.append(
            {
                "dataset_id": item.get("dataset_id"),
                "group1": item.get("group1"),
                "group2": item.get("group2"),
                "gene_ids": gene_ids,
                "logfc": logfc_vals,
                "proba_de": proba_vals,
                "norm_group1_expr": group1_expr,
                "norm_group2_expr": group2_expr,
                "num_genes": len(gene_ids),
            }
        )
    return processed


def convert_pt_to_hdf5(de_res, h5_file_path, target_max_len=1024, batch_size=1000):
    if len(de_res) == 0:
        print("Empty input. Skipping conversion.")
        return

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
            "norm_group1_expr",
            shape=(n, max_len),
            dtype="float32",
            chunks=(min(batch_size, n), max_len),
            compression="gzip",
            compression_opts=6,
            shuffle=True,
        )
        g2_ds = f.create_dataset(
            "norm_group2_expr",
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

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            bsize = end - start
            gene_ids_batch = np.full((bsize, max_len), -1, dtype="int32")
            logfc_batch = np.zeros((bsize, max_len), dtype="float32")
            proba_batch = np.zeros((bsize, max_len), dtype="float32")
            g1_batch = np.zeros((bsize, max_len), dtype="float32")
            g2_batch = np.zeros((bsize, max_len), dtype="float32")
            lengths = np.zeros((bsize,), dtype="int32")
            dsids = []

            for bi, idx in enumerate(range(start, end)):
                rec = de_res[idx]
                L = min(len(rec["gene_ids"]), max_len)
                lengths[bi] = L
                dsids.append(rec["dataset_id"])
                gene_ids_batch[bi, :L] = rec["gene_ids"][:L].astype("int32")
                logfc_batch[bi, :L] = rec["logfc"][:L].astype("float32")
                proba_batch[bi, :L] = rec["proba_de"][:L].astype("float32")
                g1_batch[bi, :L] = rec["norm_group1_expr"][:L].astype("float32")
                g2_batch[bi, :L] = rec["norm_group2_expr"][:L].astype("float32")

            gene_ids_ds[start:end] = gene_ids_batch
            logfc_ds[start:end] = logfc_batch
            proba_ds[start:end] = proba_batch
            g1_ds[start:end] = g1_batch
            g2_ds[start:end] = g2_batch
            length_ds[start:end] = lengths
            dsid_ds[start:end] = dsids


def main():
    args = parse_args()
    datadir = os.path.abspath(args.datadir)
    outputdir = os.path.abspath(args.outputdir)
    os.makedirs(outputdir, exist_ok=True)

    de_profiles = [
        os.path.join(datadir, f)
        for f in os.listdir(datadir)
        if f.endswith(".pt") and f.startswith("processed_de_")
    ]
    if not de_profiles:
        raise FileNotFoundError(f"No processed DE files found in {datadir}")

    start = time.time()
    if args.ncore > 1:
        with Pool(processes=args.ncore) as pool:
            results = pool.starmap(
                process_file,
                [
                    (path, args.topk, args.min_genes, args.proba_de_th, args.nonzero_th, args.fc_th)
                    for path in de_profiles
                ],
            )
    else:
        results = [
            process_file(
                path, args.topk, args.min_genes, args.proba_de_th, args.nonzero_th, args.fc_th
            )
            for path in de_profiles
        ]

    flat_list = [item for sublist in results for item in sublist]

    output_path = os.path.join(outputdir, "all_processed_de_profiles_compressed.h5")
    print(f"Writing HDF5 to {output_path}")
    convert_pt_to_hdf5(flat_list, output_path, target_max_len=args.topk)
    end = time.time()
    print(f"Processing completed in {int(end - start)} seconds.")


if __name__ == "__main__":
    main()
