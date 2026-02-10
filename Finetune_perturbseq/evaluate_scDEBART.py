import os
import argparse
import pandas as pd
import numpy as np
import pickle
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def match_gene_ids_to_pred_logfc(pred_logfc, gene_ids):
    aligned = {}
    all_gene_ids = np.unique(np.concatenate([np.array(v).astype(str) for v in gene_ids.values()]))
    all_gene_ids.sort()
    for pid in tqdm(pred_logfc, desc="Matching gene IDs"):
        order = np.array(gene_ids[pid]).astype(str)
        arr = np.asarray(pred_logfc[pid], dtype=object)
        if isinstance(pred_logfc[pid], list):
            for i, a in enumerate(pred_logfc[pid]):
                vec = np.asarray(a)
                orderi = order[i]
                if len(vec) != len(orderi):
                    continue
                sidx = np.argsort(orderi)
                sorted_gene_ids = orderi[sidx]
                sorted_pred = vec[sidx]
                full = np.full(len(all_gene_ids), np.nan)
                pos = np.searchsorted(all_gene_ids, sorted_gene_ids)
                full[pos] = sorted_pred
                aligned[f"{pid}_{i+1}"] = full
        else:
            vec = arr
            if len(vec) != len(order):
                continue
            sidx = np.argsort(order)
            sorted_gene_ids = order[sidx]
            sorted_pred = vec[sidx]
            full = np.full(len(all_gene_ids), np.nan)
            pos = np.searchsorted(all_gene_ids, sorted_gene_ids)
            full[pos] = sorted_pred
            aligned[str(pid)] = full
    df = pd.DataFrame.from_dict(aligned, orient='index', columns=all_gene_ids)
    return df

def compute_rowwise_corr_top_withoutzero(df_pred, df_true, topn=200, only_true_top=True):
    results = []
    for idx in df_pred.index:
        pred = df_pred.loc[idx].values
        true = df_true.loc[idx].values
        if np.all(np.isnan(pred)) or np.all(np.isnan(true)):
            continue
        mask = (~np.isnan(pred)) & (~np.isnan(true)) & (true != 0.0)
        pred, true = pred[mask], true[mask]
        if len(pred) < 3:
            continue
        pcc_all, _ = pearsonr(pred, true)
        spc_all, _ = spearmanr(pred, true)
        cossim_all = cosine_similarity(pred.reshape(1, -1), true.reshape(1, -1))[0][0]
        true_top_idx = np.argsort(-np.abs(true))[:topn]
        pred_top_idx = np.argsort(-np.abs(pred))[:topn]
        union_idx = true_top_idx if only_true_top else np.union1d(true_top_idx, pred_top_idx)
        if len(union_idx) > 3:
            pcc_top, _ = pearsonr(pred[union_idx], true[union_idx])
            spc_top, _ = spearmanr(pred[union_idx], true[union_idx])
            cossim_top = cosine_similarity(pred[union_idx].reshape(1, -1), true[union_idx].reshape(1, -1))[0][0]
        else:
            pcc_top, spc_top, cossim_top = np.nan, np.nan, np.nan
        results.append([idx, pcc_all, spc_all, cossim_all, pcc_top, spc_top, cossim_top])
    return pd.DataFrame(results, columns=["perturb_gene", "pcc_all", "spc_all", "cossim_all", "pcc_top", "spc_top", "cossim_top"])

def compute_rowwise_mse_top_withoutzero(df_pred, df_true, topn=200, only_true_top=True):
    results = []
    for idx in df_pred.index:
        pred = df_pred.loc[idx].values
        true = df_true.loc[idx].values
        if np.all(np.isnan(pred)) or np.all(np.isnan(true)):
            continue
        mask = (~np.isnan(pred)) & (~np.isnan(true)) & (true != 0.0)
        pred, true = pred[mask], true[mask]
        if len(pred) < 3:
            continue
        mse_all = np.mean((pred - true) ** 2)
        true_top_idx = np.argsort(-np.abs(true))[:topn]
        pred_top_idx = np.argsort(-np.abs(pred))[:topn]
        union_idx = true_top_idx if only_true_top else np.union1d(true_top_idx, pred_top_idx)
        if len(union_idx) > 3:
            mse_top = np.mean((pred[union_idx] - true[union_idx]) ** 2)
        else:
            mse_top = np.nan
        results.append([idx, mse_all, mse_top])
    return pd.DataFrame(results, columns=["perturb_gene", "mse_all", "mse_top"])

def compute_enrichment(query, ref, space, pseudocount=0.5):
    overlap = len(query.intersection(ref)) + pseudocount
    expected = (len(query) * len(ref) / max(len(space), 1)) + pseudocount
    return overlap / expected

def compute_rowwise_ef_top(df_pred, df_true, topn=100, pseudocount=0.5):
    cols = sorted(set(df_pred.columns).intersection(set(df_true.columns)))
    df_pred = df_pred[cols]
    df_true = df_true[cols]
    results = []
    for idx in df_pred.index:
        pv = df_pred.loc[idx].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        tv = df_true.loc[idx].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        common_index = pv.index.intersection(tv.index)
        if len(common_index) == 0:
            continue
        pv = pv.loc[common_index]
        tv = tv.loc[common_index]
        nz_mask = tv != 0
        if nz_mask.sum() == 0:
            continue
        tv_nz = tv[nz_mask]
        space = set(pv.index.tolist())
        k = min(topn, len(space))
        if k == 0:
            continue
        query = set(pv.abs().nlargest(k).index.tolist())
        ref = set(tv_nz.abs().nlargest(k).index.tolist())
        ef = compute_enrichment(query, ref, space, pseudocount=pseudocount)
        results.append([idx, ef, len(query.intersection(ref)), len(query), len(ref), len(space)])
    return pd.DataFrame(results, columns=["row_id", "ef", "overlap", "|query|", "|ref|", "|space|"]).set_index("row_id")

def compute_sign_metrics_multiclass(df_pred, df_true, fc_list=[1.2]):
    results = []
    logs = [np.log2(fc) for fc in fc_list]
    for idx in df_pred.index:
        pred = df_pred.loc[idx].values
        true = df_true.loc[idx].values
        if np.all(np.isnan(pred)) or np.all(np.isnan(true)):
            continue
        mask = (~np.isnan(pred)) & (~np.isnan(true))
        pred, true = pred[mask], true[mask]
        if len(pred) < 3:
            continue
        for fc, th in zip(fc_list, logs):
            y_true = np.where(true >= th, 1, np.where(true <= -th, -1, 0))
            y_pred = np.where(pred >= th, 1, np.where(pred <= -th, -1, 0))
            TP_up = int(((y_true == 1) & (y_pred == 1)).sum())
            FP_up = int(((y_true != 1) & (y_pred == 1)).sum())
            FN_up = int(((y_true == 1) & (y_pred != 1)).sum())
            TN_up = len(y_true) - TP_up - FP_up - FN_up
            TP_down = int(((y_true == -1) & (y_pred == -1)).sum())
            FP_down = int(((y_true != -1) & (y_pred == -1)).sum())
            FN_down = int(((y_true == -1) & (y_pred != -1)).sum())
            TN_down = len(y_true) - TP_down - FP_down - FN_down
            TP_none = int(((y_true == 0) & (y_pred == 0)).sum())
            FP_none = int(((y_true != 0) & (y_pred == 0)).sum())
            FN_none = int(((y_true == 0) & (y_pred != 0)).sum())
            TN_none = len(y_true) - TP_none - FP_none - FN_none
            results.append([idx, fc, TP_up, FP_up, FN_up, TN_up, TP_down, FP_down, FN_down, TN_down, TP_none, FP_none, FN_none, TN_none, len(y_true)])
    return pd.DataFrame(results, columns=["perturb_gene", "fc_cut", "TP_up", "FP_up", "FN_up", "TN_up", "TP_down", "FP_down", "FN_down", "TN_down", "TP_none", "FP_none", "FN_none", "TN_none", "n_genes"])

def main():
    parser = argparse.ArgumentParser(description='Evaluate scDEBART predictions')
    parser.add_argument('--pred_npz', type=str, required=True, help='Path to scDEBART prediction npz file')
    parser.add_argument('--real_lfc_pkl', type=str, required=True, help='Path to real LFC pkl file (lfc_median_df_withens_mean)')
    parser.add_argument('--outputdir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--gene_id_dict_pkl', type=str, default=None, help='Path to gene_id_dict pkl file (optional)')
    args = parser.parse_args()
    os.makedirs(args.outputdir, exist_ok=True)
    print("Loading scDEBART predictions...")
    data = np.load(args.pred_npz, allow_pickle=True)
    scdebart_pred_logfc = data['pred_logfc'].item()
    scdebart_true_logfc = data['true_logfc'].item()
    scdebart_gene_ids = data['gene_ids'].item()
    print("Processing prediction data...")
    scdebart_pred_logfc_df = match_gene_ids_to_pred_logfc(scdebart_pred_logfc, scdebart_gene_ids)
    scdebart_true_logfc_df = match_gene_ids_to_pred_logfc(scdebart_true_logfc, scdebart_gene_ids)
    if args.gene_id_dict_pkl:
        with open(args.gene_id_dict_pkl, 'rb') as f:
            gene_id_dict = pickle.load(f)
        scdebart_pred_logfc_df['perturb_gene_ens'] = scdebart_pred_logfc_df.index.to_series().apply(
            lambda x: gene_id_dict.get(int(x.split('_')[0]), x.split('_')[0])
        )
        scdebart_true_logfc_df['perturb_gene_ens'] = scdebart_true_logfc_df.index.to_series().apply(
            lambda x: gene_id_dict.get(int(x.split('_')[0]), x.split('_')[0])
        )
        num_cols = [c for c in scdebart_true_logfc_df.columns if c not in ['perturb_gene_ens'] and pd.api.types.is_numeric_dtype(scdebart_true_logfc_df[c])]
        scdebart_true_logfc_df_mean = scdebart_true_logfc_df.groupby('perturb_gene_ens')[num_cols].mean()
        scdebart_pred_logfc_df_mean = scdebart_pred_logfc_df.groupby('perturb_gene_ens')[num_cols].mean()
        scdebart_true_logfc_df_mean = scdebart_true_logfc_df_mean.rename(columns=lambda x: gene_id_dict.get(int(x), x))
        scdebart_pred_logfc_df_mean = scdebart_pred_logfc_df_mean.rename(columns=lambda x: gene_id_dict.get(int(x), x))
    else:
        scdebart_pred_logfc_df_mean = scdebart_pred_logfc_df
        scdebart_true_logfc_df_mean = scdebart_true_logfc_df
    print("Loading real LFC data...")
    lfc_median_df_withens_mean = pd.read_pickle(args.real_lfc_pkl)
    print("Finding common genes...")
    gene_cols = [c for c in lfc_median_df_withens_mean.columns if c.startswith("ENSG")]
    common_gspace = sorted(list(set(scdebart_pred_logfc_df_mean.columns).intersection(set(gene_cols))))
    common_pert_genes = sorted(list(set(scdebart_pred_logfc_df_mean.index.astype(str)).intersection(set(lfc_median_df_withens_mean.index.astype(str)))))
    print(f'Number of common genes in gene space: {len(common_gspace)}')
    print(f'Number of common perturbation genes: {len(common_pert_genes)}')
    real_lfc_median_df_common = lfc_median_df_withens_mean.loc[common_pert_genes, common_gspace]
    scdebart_pred_logfc_common = scdebart_pred_logfc_df_mean.loc[common_pert_genes, common_gspace]
    scdebart_true_logfc_common = scdebart_true_logfc_df_mean.loc[common_pert_genes, common_gspace]
    topn_list = [ 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    print("\nCalculating correlation metrics...")
    dfs_corr = []
    dfs_mse = []
    for t in tqdm(topn_list, desc="TopN Correlation and MSE"):
        df_corr = compute_rowwise_corr_top_withoutzero(scdebart_pred_logfc_common, real_lfc_median_df_common, topn=t, only_true_top=True)
        df_corr["topn"] = t
        dfs_corr.append(df_corr)
        df_mse = compute_rowwise_mse_top_withoutzero(scdebart_pred_logfc_common, real_lfc_median_df_common, topn=t, only_true_top=True)
        df_mse["topn"] = t
        dfs_mse.append(df_mse)
    corr_df = pd.concat(dfs_corr, ignore_index=True)
    mse_df = pd.concat(dfs_mse, ignore_index=True)
    corr_df.to_csv(os.path.join(args.outputdir, "scDEBART_correlation_results_topn.csv"), index=False)
    mse_df.to_csv(os.path.join(args.outputdir, "scDEBART_mse_results_topn.csv"), index=False)
    print(f"Saved: scDEBART_correlation_results_topn.csv")
    print(f"Saved: scDEBART_mse_results_topn.csv")
    print("\nCalculating Enrichment Factor (EF)...")
    efs = []
    for t in tqdm(topn_list, desc="EF topn"):
        df = compute_rowwise_ef_top(scdebart_pred_logfc_common, real_lfc_median_df_common, topn=t, pseudocount=0.05).reset_index()
        df["topn"] = t
        efs.append(df[["row_id", "ef", "topn"]])
    all_ef_df = pd.concat(efs, ignore_index=True)
    all_ef_df.to_csv(os.path.join(args.outputdir, "scDEBART_EF_results_topn.csv"), index=False)
    print(f"Saved: scDEBART_EF_results_topn.csv")
    print("\nCalculating Sign Metrics (F1, Accuracy)...")
    df_acc = compute_sign_metrics_multiclass(scdebart_pred_logfc_common, real_lfc_median_df_common)
    df_acc["precision_up"] = df_acc["TP_up"] / (df_acc["TP_up"] + df_acc["FP_up"] + 1e-6)
    df_acc["recall_up"] = df_acc["TP_up"] / (df_acc["TP_up"] + df_acc["FN_up"] + 1e-6)
    df_acc["f1_up"] = 2 * (df_acc["precision_up"] * df_acc["recall_up"]) / (df_acc["precision_up"] + df_acc["recall_up"] + 1e-6)
    df_acc["precision_down"] = df_acc["TP_down"] / (df_acc["TP_down"] + df_acc["FP_down"] + 1e-6)
    df_acc["recall_down"] = df_acc["TP_down"] / (df_acc["TP_down"] + df_acc["FN_down"] + 1e-6)
    df_acc["f1_down"] = 2 * (df_acc["precision_down"] * df_acc["recall_down"]) / (df_acc["precision_down"] + df_acc["recall_down"] + 1e-6)
    df_acc["acc_up"] = (df_acc["TP_up"] + df_acc["TN_up"]) / (df_acc["TP_up"] + df_acc["TN_up"] + df_acc["FP_up"] + df_acc["FN_up"] + 1e-6)
    df_acc["acc_down"] = (df_acc["TP_down"] + df_acc["TN_down"]) / (df_acc["TP_down"] + df_acc["TN_down"] + df_acc["FP_down"] + df_acc["FN_down"] + 1e-6)
    df_acc["acc_mean"] = (df_acc["acc_up"] + df_acc["acc_down"]) / 2
    df_acc["f1_mean"] = (df_acc["f1_up"] + df_acc["f1_down"]) / 2
    df_acc.to_csv(os.path.join(args.outputdir, "scDEBART_sign_metrics.csv"), index=False)
    
    print(f"Saved: scDEBART_sign_metrics.csv")
    print("\n=== Summary Statistics ===")
    summary_corr = corr_df.groupby("topn")[["pcc_top", "spc_top", "cossim_top"]].mean()
    summary_mse = mse_df.groupby("topn")[["mse_top"]].mean()
    summary_ef = all_ef_df.groupby("topn")[["ef"]].mean()
    summary_sign = df_acc.groupby("fc_cut")[["f1_mean", "acc_mean"]].mean()
    
    print("\nCorrelation (mean across perturbations):")
    print(summary_corr)
    print("\nMSE (mean across perturbations):")
    print(summary_mse)
    print("\nEnrichment Factor (mean across perturbations):")
    print(summary_ef)
    print("\nSign Metrics (mean across perturbations):")
    print(summary_sign)
    
    summary_corr.to_csv(os.path.join(args.outputdir, "scDEBART_summary_correlation.csv"))
    summary_mse.to_csv(os.path.join(args.outputdir, "scDEBART_summary_mse.csv"))
    summary_ef.to_csv(os.path.join(args.outputdir, "scDEBART_summary_ef.csv"))
    summary_sign.to_csv(os.path.join(args.outputdir, "scDEBART_summary_sign_metrics.csv"))
    print("\n✓ All evaluation metrics calculated and saved successfully!")

if __name__ == "__main__":
    main()
