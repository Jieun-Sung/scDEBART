import os
import argparse
import random
import pickle
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def append_filter_log(log_path, lines):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")

def load_knowledge_data(data_dir):
    with open(os.path.join(data_dir, "ens2geneid.pkl"), "rb") as f:
        gene_dict = pickle.load(f)
    ens2id = {v: k for k, v in gene_dict.items()}
    with open(os.path.join(data_dir, "BioBERT_gene_embedding_dict.pkl"), "rb") as f:
        biobert_raw = pickle.load(f)
    reactome_vec = pd.read_csv(
        os.path.join(data_dir, "c2_all_2024_1_membership_ens2pathway.tsv"),
        sep="\t",
        index_col=0,
    )
    reactome_vec = reactome_vec[reactome_vec.index.isin(gene_dict.values())]
    reactome_raw = reactome_vec.apply(lambda x: x.values.tolist(), axis=1).to_dict()
    biobert_processed = {
        ens2id[k]: torch.tensor(v, dtype=torch.bfloat16)
        for k, v in biobert_raw.items()
        if k in ens2id
    }
    reactome_processed = {
        ens2id[k]: torch.tensor(v, dtype=torch.bfloat16)
        for k, v in reactome_raw.items()
        if k in ens2id
    }
    return gene_dict, biobert_processed, reactome_processed

class FlashAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=4096, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.d_head = d_model // nhead
        assert self.d_head * nhead == self.d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        bsz, seq_len, dim = src.shape
        q = self.q_proj(src).view(bsz, seq_len, self.nhead, self.d_head).transpose(1, 2)
        k = self.k_proj(src).view(bsz, seq_len, self.nhead, self.d_head).transpose(1, 2)
        v = self.v_proj(src).view(bsz, seq_len, self.nhead, self.d_head).transpose(1, 2)
        attn_mask = None
        if src_key_padding_mask is not None:
            attn_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(bsz, self.nhead, seq_len, seq_len)
            attn_mask = attn_mask.masked_fill(attn_mask, float("-inf"))
        if src_mask is not None:
            if attn_mask is None:
                attn_mask = src_mask
            else:
                attn_mask = attn_mask + src_mask
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        attn_output = self.out_proj(attn_output)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        ffn_output = self.ffn(src)
        src = src + self.dropout(ffn_output)
        src = self.norm2(src)
        return src

class FlashAttentionDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=4096, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.d_head = d_model // nhead
        assert self.d_head * nhead == self.d_model
        self.self_attn_q = nn.Linear(d_model, d_model)
        self.self_attn_k = nn.Linear(d_model, d_model)
        self.self_attn_v = nn.Linear(d_model, d_model)
        self.self_attn_out = nn.Linear(d_model, d_model)
        self.cross_attn_q = nn.Linear(d_model, d_model)
        self.cross_attn_k = nn.Linear(d_model, d_model)
        self.cross_attn_v = nn.Linear(d_model, d_model)
        self.cross_attn_out = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        bsz, seq_len, dim = tgt.shape
        q = self.self_attn_q(tgt).view(bsz, seq_len, self.nhead, self.d_head).transpose(1, 2)
        k = self.self_attn_k(tgt).view(bsz, seq_len, self.nhead, self.d_head).transpose(1, 2)
        v = self.self_attn_v(tgt).view(bsz, seq_len, self.nhead, self.d_head).transpose(1, 2)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=tgt.device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(bsz, self.nhead, -1, -1)
        attn_mask = causal_mask
        if tgt_key_padding_mask is not None:
            pad_mask = tgt_key_padding_mask.unsqueeze(1).unsqueeze(2)
            pad_mask = pad_mask.expand(bsz, self.nhead, seq_len, seq_len)
            attn_mask = attn_mask | pad_mask
        attn_mask = attn_mask.masked_fill(attn_mask, float("-inf"))
        self_attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        self_attn_output = self_attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        self_attn_output = self.self_attn_out(self_attn_output)
        tgt = tgt + self.dropout(self_attn_output)
        tgt = self.norm1(tgt)
        if memory is not None:
            b_mem, l_mem, _ = memory.shape
            q_cross = self.cross_attn_q(tgt).view(bsz, seq_len, self.nhead, self.d_head).transpose(1, 2)
            k_cross = self.cross_attn_k(memory).view(b_mem, l_mem, self.nhead, self.d_head).transpose(1, 2)
            v_cross = self.cross_attn_v(memory).view(b_mem, l_mem, self.nhead, self.d_head).transpose(1, 2)
            cross_attn_mask = None
            if memory_key_padding_mask is not None:
                cross_attn_mask = memory_key_padding_mask.unsqueeze(1).unsqueeze(2)
                cross_attn_mask = cross_attn_mask.expand(bsz, self.nhead, seq_len, l_mem)
                cross_attn_mask = cross_attn_mask.masked_fill(cross_attn_mask, float("-inf"))
            cross_attn_output = F.scaled_dot_product_attention(
                q_cross,
                k_cross,
                v_cross,
                attn_mask=cross_attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
            cross_attn_output = cross_attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
            cross_attn_output = self.cross_attn_out(cross_attn_output)
            tgt = tgt + self.dropout(cross_attn_output)
            tgt = self.norm2(tgt)
        ffn_output = self.ffn(tgt)
        tgt = tgt + self.dropout(ffn_output)
        tgt = self.norm3(tgt)
        return tgt

class PlainMLPHead(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.GELU(),
            nn.Linear(512, 1),
        )
    def forward(self, x):
        return self.mlp(x).squeeze(-1)

class EmbeddingReconstructionHead(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.reconstruction_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
    def forward(self, x):
        return self.reconstruction_head(x)

class DEGTransformer(nn.Module):
    def __init__(self, args, reactome_dict, biobert_dict):
        super().__init__()
        embedding_dim = args.embedding_dim
        self.reactome_linear = nn.Linear(len(next(iter(reactome_dict.values()))), embedding_dim)
        self.knowledge_proj = nn.Linear(embedding_dim * 2, embedding_dim)
        self.logfc_embedding = nn.Linear(1, embedding_dim)
        self.expr_embedding = nn.Linear(1, embedding_dim)
        self.proj = nn.Linear(embedding_dim * 3, embedding_dim)
        self.encoder_layers = nn.ModuleList(
            [
                FlashAttentionEncoderLayer(
                    d_model=embedding_dim,
                    nhead=args.nhead,
                    dim_feedforward=embedding_dim * 4,
                    dropout=args.dropout,
                )
                for _ in range(args.num_encoder_layer)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                FlashAttentionDecoderLayer(
                    d_model=embedding_dim,
                    nhead=args.nhead,
                    dim_feedforward=embedding_dim * 4,
                    dropout=args.dropout,
                )
                for _ in range(args.num_decoder_layer)
            ]
        )
        self.logfc_head = PlainMLPHead(embedding_dim)
        self.embedding_reconstruction_head = EmbeddingReconstructionHead(embedding_dim)
        num_total_genes = args.num_genes
        reactome_dim = len(next(iter(reactome_dict.values())))
        biobert_dim = len(next(iter(biobert_dict.values())))
        reactome_embedding_matrix = torch.zeros((num_total_genes, reactome_dim), dtype=torch.bfloat16, device=args.device)
        biobert_embedding_matrix = torch.zeros((num_total_genes, biobert_dim), dtype=torch.bfloat16, device=args.device)
        for gid, emb in reactome_dict.items():
            reactome_embedding_matrix[gid] = emb.to(args.device)
        for gid, emb in biobert_dict.items():
            biobert_embedding_matrix[gid] = emb.to(args.device)
        self.register_buffer("reactome_embedding_matrix", reactome_embedding_matrix)
        self.register_buffer("biobert_embedding_matrix", biobert_embedding_matrix)

class LearnableLossWeights(nn.Module):
    def __init__(self, init_alpha=1.0, init_beta=0.5, init_gamma=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(init_beta, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))
    def forward(self):
        return torch.sigmoid(self.alpha), torch.sigmoid(self.beta), torch.sigmoid(self.gamma)

class CombinedModel(nn.Module):
    def __init__(self, base_model, lossw_module):
        super().__init__()
        self.base = base_model
        self.lossw = lossw_module
    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

def compute_weighted_mse_loss(pred, target, weight_factor=0.5):
    weights = 1.0 + weight_factor * torch.abs(target)
    weighted_se = weights * (pred - target) ** 2
    return weighted_se.mean()

def compute_differentiable_correlation(pred, target):
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    cov = (pred_centered * target_centered).mean()
    pred_std = torch.sqrt((pred_centered**2).mean() + 1e-8)
    target_std = torch.sqrt((target_centered**2).mean() + 1e-8)
    return cov / (pred_std * target_std + 1e-8)

def compute_topk_correlation_differentiable(pred, target, k_ratio=0.2):
    num_samples = len(pred)
    k = max(1, int(num_samples * k_ratio))
    abs_target = torch.abs(target)
    _, topk_indices = torch.topk(abs_target, k)
    pred_topk = pred[topk_indices]
    target_topk = target[topk_indices]
    pred_mean = pred_topk.mean()
    target_mean = target_topk.mean()
    pred_centered = pred_topk - pred_mean
    target_centered = target_topk - target_mean
    cov = (pred_centered * target_centered).mean()
    pred_std = torch.sqrt((pred_centered**2).mean() + 1e-8)
    target_std = torch.sqrt((target_centered**2).mean() + 1e-8)
    return cov / (pred_std * target_std + 1e-8)

def compute_combined_loss(pred, target, loss_weights, weight_factor=0.5, topk_ratio=0.2):
    weighted_mse = compute_weighted_mse_loss(pred, target, weight_factor=weight_factor)
    global_corr = compute_differentiable_correlation(pred, target)
    topk_corr = compute_topk_correlation_differentiable(pred, target, k_ratio=topk_ratio)
    alpha, beta, gamma = loss_weights()
    combined_corr = beta * global_corr + (1 - beta) * topk_corr
    corr_loss = 1.0 - combined_corr
    total_loss = alpha * weighted_mse + gamma * corr_loss
    return total_loss, weighted_mse, global_corr, topk_corr, alpha, beta, gamma

class PerturbationDataset(Dataset):
    def __init__(self, h5_file_path, allowed_gene_ids, target_gene_ids):
        self.h5_file_path = h5_file_path
        self.allowed_gene_ids = set(allowed_gene_ids)
        self.target_gene_ids = set(target_gene_ids)
        self.file = None
        with h5py.File(self.h5_file_path, "r") as f:
            self.num_samples = f.attrs["num_samples"]
            self.idx_to_pert = f["pertubed_gene_id"][:]
        self.selected_indices = [i for i, gid in enumerate(self.idx_to_pert) if gid in self.allowed_gene_ids]
    def _ensure_file(self):
        if self.file is None:
            self.file = h5py.File(self.h5_file_path, "r", libver="latest", swmr=True)
    def __len__(self):
        return len(self.selected_indices)
    def __getitem__(self, idx):
        self._ensure_file()
        actual_idx = self.selected_indices[idx]
        gene_ids = torch.from_numpy(self.file["gene_ids"][actual_idx]).long()
        target_logfc = torch.from_numpy(self.file["logfc"][actual_idx]).bfloat16()
        ctrl_exp_np = np.log1p(self.file["normalized_expr_group2"][actual_idx])
        ctrl_exp = torch.from_numpy(ctrl_exp_np).bfloat16()
        pertubed_gene_id = torch.tensor(self.file["pertubed_gene_id"][actual_idx], dtype=torch.long)
        mask = torch.tensor([gid.item() in self.target_gene_ids for gid in gene_ids], dtype=torch.bool)
        gene_ids = gene_ids[mask]
        target_logfc = target_logfc[mask]
        ctrl_exp = ctrl_exp[mask]
        init_pert_logfc = torch.zeros_like(target_logfc)
        return {
            "gene_ids": gene_ids,
            "ctrl_exp": ctrl_exp,
            "init_pert_logfc": init_pert_logfc,
            "target_logfc": target_logfc,
            "pertubed_gene_id": pertubed_gene_id,
        }
    def __del__(self):
        if self.file is not None:
            self.file.close()
            self.file = None

def custom_collate_fn(batch, max_len, pad_token_id):
    padded_batch = []
    for sample in batch:
        gene_ids = sample["gene_ids"]
        ctrl_exp = sample["ctrl_exp"]
        init_pert_logfc = sample["init_pert_logfc"]
        target_logfc = sample["target_logfc"]
        pertubed_gene_id = sample["pertubed_gene_id"]
        order = torch.argsort(gene_ids)
        gene_ids = gene_ids[order]
        ctrl_exp = ctrl_exp[order]
        init_pert_logfc = init_pert_logfc[order]
        target_logfc = target_logfc[order]
        current_len = len(gene_ids)
        if current_len < max_len:
            pad_length = max_len - current_len
            gene_ids = F.pad(gene_ids, (0, pad_length), value=pad_token_id)
            ctrl_exp = F.pad(ctrl_exp, (0, pad_length), value=0.0)
            init_pert_logfc = F.pad(init_pert_logfc, (0, pad_length), value=0.0)
            target_logfc = F.pad(target_logfc, (0, pad_length), value=0.0)
        padded_batch.append(
            {
                "gene_ids": gene_ids,
                "ctrl_exp": ctrl_exp,
                "init_pert_logfc": init_pert_logfc,
                "target_logfc": target_logfc,
                "pertubed_gene_id": pertubed_gene_id,
            }
        )
    return torch.utils.data.default_collate(padded_batch)

def load_pretrained_model(model_path, args, reactome_dict, biobert_dict):
    model = DEGTransformer(args, reactome_dict, biobert_dict)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    return model

def save_tuned_model(model_engine, save_dir, epoch):
    if model_engine.local_rank != 0:
        return
    os.makedirs(save_dir, exist_ok=True)
    base_state = model_engine.module.base.state_dict()
    save_path = os.path.join(save_dir, f"tuned_model_epoch_{epoch}.pt")
    torch.save(base_state, save_path)

def main():
    parser = argparse.ArgumentParser()
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--input_adata", type=str, required=True)
    parser.add_argument("--de_result_dir", type=str, required=True)
    parser.add_argument("--pretrained_model_dir", type=str, required=True)
    parser.add_argument("--outputdir", type=str, required=True)
    parser.add_argument("--pert_type", type=str, required=True)
    parser.add_argument("--train_val_test_dict_path", type=str, required=True)
    parser.add_argument("--num_HVG", type=int, default = 5000)
    parser.add_argument("--num_epochs", type = int, default = 50)
    parser.add_argument("--patience", type = int, default = 5)

    args = parser.parse_args()
    if not getattr(args, "deepspeed_config", None):
        raise ValueError("--deepspeed_config is required")
    pert_type = args.pert_type.upper()
    if pert_type == "INH":
        pert_value = -10.0
    elif pert_type == "OE":
        pert_value = 10.0
    else:
        raise ValueError("pert_type must be INH or OE")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        args.device = f"cuda:{args.local_rank}"
    else:
        args.device = "cpu"
    deepspeed.init_distributed()
    datadir = "./data"
    gene_id_dict, biobert_dict, reactome_dict = load_knowledge_data(datadir)
    ens2geneid = {v: k for k, v in gene_id_dict.items()}
    max_gene_id = len(np.array(list(gene_id_dict.values())))
    args.pad_token_id = max_gene_id + 1
    args.mask_token_id = max_gene_id + 2
    args.num_genes = max_gene_id + 3
    os.makedirs(args.outputdir, exist_ok=True)
    model_weights_dir = os.path.join(args.outputdir, "model_weights")
    os.makedirs(model_weights_dir, exist_ok=True)
    dataset_root = os.path.abspath(os.path.join(args.outputdir, os.pardir))
    log_path = os.path.join(dataset_root, "filtering_metadata.txt")
    append_filter_log(
        log_path,
        [
            "Finetune configuration",
            f"dataset: {args.dataset}",
            f"input_adata: {args.input_adata}",
            f"de_result_dir: {args.de_result_dir}",
            f"pretrained_model_dir: {args.pretrained_model_dir}",
            f"outputdir: {args.outputdir}",
            f"pert_type: {pert_type}",
            f"train_val_test_dict_path: {args.train_val_test_dict_path}",
            "----------------------------------------------------------------------",
        ],
    )
    h5_data_path = os.path.join(args.de_result_dir, "de_masked_high_zero_logfc_to_zero.h5")
    hvg_path = os.path.join(args.de_result_dir, f"hvg_geneids_{args.num_HVG}.pkl")
    with open(hvg_path, "rb") as f:
        hvg_geneids = pickle.load(f)
    with open(args.train_val_test_dict_path, "rb") as f:
        train_val_test_dict = pickle.load(f)
    train_ids_ens = train_val_test_dict['train']
    val_ids_ens = train_val_test_dict['val']
    test_ids_ens = train_val_test_dict['test']
    train_ids = [ens2geneid[g] for g in train_ids_ens if g in ens2geneid]
    val_ids = [ens2geneid[g] for g in val_ids_ens if g in ens2geneid]
    test_ids = [ens2geneid[g] for g in test_ids_ens if g in ens2geneid]
    pt_files = [f for f in os.listdir(args.pretrained_model_dir) if f.startswith("best_model_weights_epoch_")]
    if len(pt_files) == 0:
        pt_files = [f for f in os.listdir(args.pretrained_model_dir) if f.startswith("model_weights_epoch_")]
    if len(pt_files) == 0:
        raise FileNotFoundError("No pretrained weights found in pretrained_model_dir")
    pt_files.sort()
    best_model_path = os.path.join(args.pretrained_model_dir, pt_files[-1])
    pretrained_model = load_pretrained_model(best_model_path, args, reactome_dict, biobert_dict)
    loss_weights = LearnableLossWeights(init_alpha=1.0, init_beta=0.5, init_gamma=0.5)
    combined_model = CombinedModel(pretrained_model, loss_weights)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=combined_model, model_parameters=combined_model.parameters()
    )
    train_dataset = PerturbationDataset(h5_data_path, train_ids, target_gene_ids=hvg_geneids)
    val_dataset = PerturbationDataset(h5_data_path, val_ids, target_gene_ids=hvg_geneids)
    test_dataset = PerturbationDataset(h5_data_path, test_ids, target_gene_ids=hvg_geneids)
    collate_fn = lambda batch: custom_collate_fn(batch, max_len=len(hvg_geneids), pad_token_id=args.pad_token_id)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, sampler=test_sampler, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    loss_log_path = os.path.join(args.outputdir, "tuning_loss.txt")
    if args.local_rank == 0:
        with open(loss_log_path, "w") as f_log:
            f_log.write("Epoch\tTrain MSE\tTrain Global Corr\tTrain TopK Corr\tVal MSE\tVal Global Corr\tVal TopK Corr\n")
    best_val_loss = float("inf")
    no_improve_count = 0
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        model_engine.train()
        epoch_train_mse = 0.0
        epoch_train_global_corr = 0.0
        epoch_train_topk_corr = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", disable=(args.local_rank != 0)):
            gene_ids = batch["gene_ids"].to(args.device)
            ctrl_exp = batch["ctrl_exp"].to(args.device, dtype=torch.bfloat16)
            target_logfc = batch["target_logfc"].to(args.device, dtype=torch.bfloat16)
            pertubed_gene_ids = batch["pertubed_gene_id"].to(args.device)
            key_padding_mask = gene_ids == args.pad_token_id
            reactome_emb_data = model_engine.module.base.reactome_embedding_matrix[gene_ids]
            biobert_emb_data = model_engine.module.base.biobert_embedding_matrix[gene_ids]
            reactome_emb = model_engine.module.base.reactome_linear(reactome_emb_data)
            biobert_emb = biobert_emb_data
            knowledge_emb = torch.cat([reactome_emb, biobert_emb], dim=-1)
            knowledge_emb = model_engine.module.base.knowledge_proj(knowledge_emb)
            predicted_init_logfc = torch.zeros_like(target_logfc)
            for bi in range(len(pertubed_gene_ids)):
                pert_gene_id = pertubed_gene_ids[bi]
                pert_gene_mask = gene_ids[bi] == pert_gene_id
                if pert_gene_mask.any():
                    pert_gene_pos = torch.where(pert_gene_mask)[0]
                    predicted_init_logfc[bi, pert_gene_pos] = pert_value
            init_pert_emb = model_engine.module.base.logfc_embedding(predicted_init_logfc.unsqueeze(-1))
            expr_emb = model_engine.module.base.expr_embedding(ctrl_exp.unsqueeze(-1))
            combined_embeddings = torch.cat([knowledge_emb, init_pert_emb, expr_emb], dim=-1)
            combined_embeddings = model_engine.module.base.proj(combined_embeddings)
            encoder_output = combined_embeddings
            for enc in model_engine.module.base.encoder_layers:
                encoder_output = enc(encoder_output, src_key_padding_mask=key_padding_mask)
            decoder_output = combined_embeddings
            for dec in model_engine.module.base.decoder_layers:
                decoder_output = dec(
                    tgt=decoder_output,
                    memory=encoder_output,
                    tgt_key_padding_mask=key_padding_mask,
                    memory_key_padding_mask=key_padding_mask,
                )
            logfc_pred = model_engine.module.base.logfc_head(decoder_output)
            valid_mask = gene_ids != args.pad_token_id
            pred_masked = logfc_pred[valid_mask]
            target_masked = target_logfc[valid_mask]
            loss, mse_loss, global_corr, topk_corr, _, _, _ = compute_combined_loss(
                pred_masked, target_masked, model_engine.module.lossw
            )
            model_engine.backward(loss)
            model_engine.step()
            epoch_train_mse += mse_loss.item()
            epoch_train_global_corr += global_corr
            epoch_train_topk_corr += topk_corr
        avg_train_mse = epoch_train_mse / len(train_loader)
        avg_train_global_corr = epoch_train_global_corr / len(train_loader)
        avg_train_topk_corr = epoch_train_topk_corr / len(train_loader)
        model_engine.eval()
        val_mse_sum = 0.0
        val_global_corr_sum = 0.0
        val_topk_corr_sum = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", disable=(args.local_rank != 0)):
                gene_ids = batch["gene_ids"].to(args.device)
                ctrl_exp = batch["ctrl_exp"].to(args.device, dtype=torch.bfloat16)
                target_logfc = batch["target_logfc"].to(args.device, dtype=torch.bfloat16)
                pertubed_gene_ids = batch["pertubed_gene_id"].to(args.device)
                key_padding_mask = gene_ids == args.pad_token_id
                reactome_emb_data = model_engine.module.base.reactome_embedding_matrix[gene_ids]
                biobert_emb_data = model_engine.module.base.biobert_embedding_matrix[gene_ids]
                reactome_emb = model_engine.module.base.reactome_linear(reactome_emb_data)
                biobert_emb = biobert_emb_data
                knowledge_emb = torch.cat([reactome_emb, biobert_emb], dim=-1)
                knowledge_emb = model_engine.module.base.knowledge_proj(knowledge_emb)
                predicted_init_logfc = torch.zeros_like(target_logfc)
                for bi in range(len(pertubed_gene_ids)):
                    pert_gene_id = pertubed_gene_ids[bi]
                    pert_gene_mask = gene_ids[bi] == pert_gene_id
                    if pert_gene_mask.any():
                        pert_gene_pos = torch.where(pert_gene_mask)[0]
                        predicted_init_logfc[bi, pert_gene_pos] = pert_value
                init_pert_emb = model_engine.module.base.logfc_embedding(predicted_init_logfc.unsqueeze(-1))
                expr_emb = model_engine.module.base.expr_embedding(ctrl_exp.unsqueeze(-1))
                combined_embeddings = torch.cat([knowledge_emb, init_pert_emb, expr_emb], dim=-1)
                combined_embeddings = model_engine.module.base.proj(combined_embeddings)
                encoder_output = combined_embeddings
                for enc in model_engine.module.base.encoder_layers:
                    encoder_output = enc(encoder_output, src_key_padding_mask=key_padding_mask)
                decoder_output = combined_embeddings
                for dec in model_engine.module.base.decoder_layers:
                    decoder_output = dec(
                        tgt=decoder_output,
                        memory=encoder_output,
                        tgt_key_padding_mask=key_padding_mask,
                        memory_key_padding_mask=key_padding_mask,
                    )
                logfc_pred = model_engine.module.base.logfc_head(decoder_output)
                valid_mask = gene_ids != args.pad_token_id
                pred_masked = logfc_pred[valid_mask]
                target_masked = target_logfc[valid_mask]
                _, mse_loss, global_corr, topk_corr, _, _, _ = compute_combined_loss(
                    pred_masked, target_masked, model_engine.module.lossw
                )
                val_mse_sum += mse_loss.item()
                val_global_corr_sum += global_corr
                val_topk_corr_sum += topk_corr
        avg_val_mse = val_mse_sum / len(val_loader)
        avg_val_global_corr = val_global_corr_sum / len(val_loader)
        avg_val_topk_corr = val_topk_corr_sum / len(val_loader)
        if args.local_rank == 0:
            with open(loss_log_path, "a") as f_log:
                f_log.write(
                    f"{epoch+1}\t{avg_train_mse:.6f}\t{avg_train_global_corr:.4f}\t{avg_train_topk_corr:.4f}\t"
                    f"{avg_val_mse:.6f}\t{avg_val_global_corr:.4f}\t{avg_val_topk_corr:.4f}\n"
                )
        did_improve = avg_val_mse < best_val_loss
        if did_improve:
            best_val_loss = avg_val_mse
            no_improve_count = 0
            save_tuned_model(model_engine, model_weights_dir, epoch)
        else:
            no_improve_count += 1
        if no_improve_count >= args.patience:
            break
    model_engine.eval()
    pred_logfc_dict = {}
    true_logfc_dict = {}
    true_ctrl_exp_dict = {}
    gene_ids_dict = {}
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", disable=(args.local_rank != 0)):
            gene_ids = batch["gene_ids"].to(args.device)
            ctrl_exp = batch["ctrl_exp"].to(args.device, dtype=torch.bfloat16)
            target_logfc = batch["target_logfc"].to(args.device, dtype=torch.bfloat16)
            pertubed_gene_ids = batch["pertubed_gene_id"].to(args.device)
            key_padding_mask = gene_ids == args.pad_token_id
            reactome_emb_data = model_engine.module.base.reactome_embedding_matrix[gene_ids]
            biobert_emb_data = model_engine.module.base.biobert_embedding_matrix[gene_ids]
            reactome_emb = model_engine.module.base.reactome_linear(reactome_emb_data)
            biobert_emb = biobert_emb_data
            knowledge_emb = torch.cat([reactome_emb, biobert_emb], dim=-1)
            knowledge_emb = model_engine.module.base.knowledge_proj(knowledge_emb)
            predicted_init_logfc = torch.zeros_like(target_logfc)
            for bi in range(len(pertubed_gene_ids)):
                pert_gene_id = pertubed_gene_ids[bi]
                pert_gene_mask = gene_ids[bi] == pert_gene_id
                if pert_gene_mask.any():
                    pert_gene_pos = torch.where(pert_gene_mask)[0]
                    predicted_init_logfc[bi, pert_gene_pos] = pert_value
            init_pert_emb = model_engine.module.base.logfc_embedding(predicted_init_logfc.unsqueeze(-1))
            expr_emb = model_engine.module.base.expr_embedding(ctrl_exp.unsqueeze(-1))
            combined_embeddings = torch.cat([knowledge_emb, init_pert_emb, expr_emb], dim=-1)
            combined_embeddings = model_engine.module.base.proj(combined_embeddings)
            encoder_output = combined_embeddings
            for enc in model_engine.module.base.encoder_layers:
                encoder_output = enc(encoder_output, src_key_padding_mask=key_padding_mask)
            decoder_output = combined_embeddings
            for dec in model_engine.module.base.decoder_layers:
                decoder_output = dec(
                    tgt=decoder_output,
                    memory=encoder_output,
                    tgt_key_padding_mask=key_padding_mask,
                    memory_key_padding_mask=key_padding_mask,
                )
            logfc_pred = model_engine.module.base.logfc_head(decoder_output)
            for i in range(len(pertubed_gene_ids)):
                pert_gene_id = pertubed_gene_ids[i].item()
                valid_mask_np = (gene_ids[i] != args.pad_token_id).cpu().numpy()
                valid_genes = gene_ids[i].cpu().numpy()[valid_mask_np]
                pred_valid = logfc_pred[i].detach().float().cpu().numpy()[valid_mask_np]
                target_valid = target_logfc[i].detach().float().cpu().numpy()[valid_mask_np]
                ctrl_valid = ctrl_exp[i].detach().float().cpu().numpy()[valid_mask_np]
                pred_logfc_dict.setdefault(pert_gene_id, []).append(pred_valid.astype(np.float16))
                true_logfc_dict.setdefault(pert_gene_id, []).append(target_valid.astype(np.float16))
                true_ctrl_exp_dict.setdefault(pert_gene_id, []).append(ctrl_valid.astype(np.float16))
                gene_ids_dict.setdefault(pert_gene_id, []).append(valid_genes)
    if args.local_rank == 0:
        pred_path = os.path.join(dataset_root, f"{args.dataset}_test_gene_perturb_predictions.npz")
        np.savez_compressed(
            pred_path,
            pred_logfc=pred_logfc_dict,
            true_logfc=true_logfc_dict,
            true_ctrl_exp=true_ctrl_exp_dict,
            gene_ids=gene_ids_dict,
        )

if __name__ == "__main__":
    main()
