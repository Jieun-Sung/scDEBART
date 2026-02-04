import argparse
import concurrent.futures
import datetime
import glob
import os
import time

import numpy as np
import pandas as pd
import cellxgene_census


def find_cell_metadata(datadir):
    pattern = os.path.join(datadir, "census_obs_cellmetadata_onlyhuman_*.pkl")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def download_adata(dataset_id, census, out_dir, log_file):
    start = time.time()
    adata = cellxgene_census.get_anndata(
        census,
        organism="Homo sapiens",
        obs_value_filter=f"dataset_id == '{dataset_id}'",
    )
    nonzero_mask = np.array(adata.X.sum(axis=0)).flatten() > 0
    adata_f = adata[:, nonzero_mask]
    dataset_dir = os.path.join(out_dir, dataset_id)
    os.makedirs(dataset_dir, exist_ok=True)
    file_path = os.path.join(dataset_dir, "adata.h5ad")
    adata_f.write(file_path)
    end = time.time()
    duration_time = round(end - start, 2)
    file_size = round(os.path.getsize(file_path) / (1024 * 1024), 2)
    download_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"{dataset_id}\t{adata_f.shape[0]}\t{adata_f.shape[1]}\t"
        f"{duration_time}\t{download_date}\t{file_size}\n"
    )
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(log_entry)


def download_adata_wrapper(dataset_id, census, out_dir, log_file, failed_log_file):
    try:
        download_adata(dataset_id, census, out_dir, log_file)
    except Exception:
        with open(failed_log_file, "a", encoding="utf-8") as log:
            log.write(f"{dataset_id}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download cellxgene AnnData per dataset using the census API."
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="./cellxgene",
        help="Directory to store downloaded datasets and metadata.",
    )
    args = parser.parse_args()

    datadir = os.path.abspath(args.datadir)
    os.makedirs(datadir, exist_ok=True)

    census = cellxgene_census.open_soma()
    census_datasets = census["census_info"]["datasets"].read().concat().to_pandas()
    metadata_path = find_cell_metadata(datadir)
    if metadata_path:
        cell_metadata = pd.read_pickle(metadata_path)
        unique_datasetid = cell_metadata.dataset_id.unique()
    else:
        print(
            "No local census cell metadata file found; defaulting to all census datasets."
        )
        unique_datasetid = census_datasets.dataset_id.unique()

    excluded_ids = set(unique_datasetid) - set(census_datasets.dataset_id)
    if excluded_ids:
        print(
            f"{excluded_ids} dataset id are not found in census datasets. Will be excluded from download."
        )

    log_file = os.path.join(datadir, "raw_anndata_download_log.txt")
    failed_log_file = os.path.join(datadir, "raw_anndata_download_failed_log.txt")

    if os.path.exists(log_file):
        os.remove(log_file)

    log_entry = (
        "datasetID\tobs_rownumber\tvar_rownumber\t"
        "download_time(sec)\tdownload_date\tadata_size(MB)\n"
    )
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(log_entry)

    valid_dataset_ids = [
        dataset_id for dataset_id in unique_datasetid if dataset_id not in excluded_ids
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
        for dataset_id in valid_dataset_ids:
            executor.submit(
                download_adata_wrapper,
                dataset_id,
                census,
                datadir,
                log_file,
                failed_log_file,
            )


if __name__ == "__main__":
    main()
