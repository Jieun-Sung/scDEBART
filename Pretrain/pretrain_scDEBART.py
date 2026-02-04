import pandas as pd
import numpy as np
import re, os, sys, argparse
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata.utils")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)
import anndata
import time
from tqdm import tqdm
import json
import networkx as nx
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import umap
import math
import datetime
import matplotlib.pyplot as plt
import pickle
import shutil
from matplotlib.ticker import FuncFormatter
import random
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import issparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.functional import scaled_dot_product_attention
from transformers import get_cosine_schedule_with_warmup
import deepspeed
from torch.utils.data.distributed import DistributedSampler
import h5py
import gc

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#============================================================
# LOAD DATA
#============================================================

datadir = './data'
gene_id_dict = None
biobert_dict = None
reactome_dict = None

def load_knowledge_data(data_dir):
    with open(os.path.join(data_dir, 'ens2geneid.pkl'), 'rb') as f:
        gene_dict = pickle.load(f)

    ens2id = {v: k for k, v in gene_dict.items()}

    with open(os.path.join(data_dir, 'BioBERT_gene_embedding_dict.pkl'), 'rb') as f:
        biobert_raw = pickle.load(f)

    reactome_vec = pd.read_csv(
        os.path.join(data_dir, 'c2_all_2024_1_membership_ens2pathway.tsv'),
        sep='\t',
        index_col=0
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

#============================================================
# Functions and Classes
#============================================================

def check_overflow(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"Overflow detected in {name}: NaN={torch.isnan(tensor).sum()}, Inf={torch.isinf(tensor).sum()}")
        return True
    return False

class MemoryMappedDataset(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.file = None
        self.num_samples = None
        with h5py.File(hdf5_path, 'r') as f:
            self.num_samples = f.attrs['num_samples']

    def _ensure_file(self):
        if self.file is None:
            self.file = h5py.File(
                self.hdf5_path,
                'r',
                libver='latest',
                swmr=True
            )

    def __getitem__(self, idx):
        self._ensure_file()
        idx = int(idx)
        try:
            gene_ids = torch.from_numpy(self.file['gene_ids'][idx]).long()
            logfc = torch.from_numpy(self.file['logfc'][idx]).bfloat16()
            ctrl_exp_np = np.log1p(self.file['norm_group2_expr'][idx])
            ctrl_exp = torch.from_numpy(ctrl_exp_np).bfloat16()            
            return {'gene_ids': gene_ids,
                   'logfc': logfc,
                   'ctrl_exp': ctrl_exp}
        except Exception as e:
            print(f"Error reading index {idx}: {e}")
            raise

    def __len__(self):
        return self.num_samples

    def __del__(self):
        if self.file is not None:
            self.file.close()
            self.file = None

def custom_collate_fn(batch):
    max_len = 1024  
    pad_token_id = len(gene_id_dict) + 1
    
    padded_batch = []
    for sample in batch:
        gene_ids = sample['gene_ids']
        logfc = sample['logfc'] 
        ctrl_exp = sample['ctrl_exp']
        
        current_len = len(gene_ids)
        if current_len < max_len:
            pad_length = max_len - current_len
            gene_ids = F.pad(gene_ids, (0, pad_length), value=pad_token_id)
            logfc = F.pad(logfc, (0, pad_length), value=0.0)
            ctrl_exp = F.pad(ctrl_exp, (0, pad_length), value=0.0)
        
        padded_batch.append({
            'gene_ids': gene_ids,
            'logfc': logfc,
            'ctrl_exp': ctrl_exp
        })
    
    return torch.utils.data.default_collate(padded_batch)


class FlashAttentionEncoderLayer(nn.Module):
    """BART-style Bidirectional Encoder Layer"""
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
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        B, L, D = src.shape
        
        # Multi-head self-attention (bidirectional)
        q = self.q_proj(src).view(B, L, self.nhead, self.d_head).transpose(1, 2)
        k = self.k_proj(src).view(B, L, self.nhead, self.d_head).transpose(1, 2)
        v = self.v_proj(src).view(B, L, self.nhead, self.d_head).transpose(1, 2)
        
        attn_mask = None
        if src_key_padding_mask is not None:
            attn_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(B, self.nhead, L, L)
            attn_mask = attn_mask.masked_fill(attn_mask, float('-inf'))
        
        if src_mask is not None:
            if attn_mask is None:
                attn_mask = src_mask
            else:
                attn_mask = attn_mask + src_mask
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False  # Bidirectional
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        attn_output = self.out_proj(attn_output)
        
        # Residual connection + LayerNorm
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        
        # Feed-forward
        ffn_output = self.ffn(src)
        src = src + self.dropout(ffn_output)
        src = self.norm2(src)
        
        return src

class FlashAttentionDecoderLayer(nn.Module):
    """BART-style Autoregressive Decoder Layer"""
    def __init__(self, d_model, nhead, dim_feedforward=4096, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.d_head = d_model // nhead
        assert self.d_head * nhead == self.d_model
        
        # Masked self-attention
        self.self_attn_q = nn.Linear(d_model, d_model)
        self.self_attn_k = nn.Linear(d_model, d_model)
        self.self_attn_v = nn.Linear(d_model, d_model)
        self.self_attn_out = nn.Linear(d_model, d_model)
        
        # Cross-attention (encoder-decoder)
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
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        B, L, D = tgt.shape
        
        # Masked self-attention (autoregressive)
        q = self.self_attn_q(tgt).view(B, L, self.nhead, self.d_head).transpose(1, 2)
        k = self.self_attn_k(tgt).view(B, L, self.nhead, self.d_head).transpose(1, 2)
        v = self.self_attn_v(tgt).view(B, L, self.nhead, self.d_head).transpose(1, 2)
        
        # Causal mask for autoregressive behavior
        causal_mask = torch.triu(torch.ones(L, L, device=tgt.device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, self.nhead, -1, -1)
        
        attn_mask = causal_mask
        if tgt_key_padding_mask is not None:
            pad_mask = tgt_key_padding_mask.unsqueeze(1).unsqueeze(2)
            pad_mask = pad_mask.expand(B, self.nhead, L, L)
            attn_mask = attn_mask | pad_mask
        
        attn_mask = attn_mask.masked_fill(attn_mask, float('-inf'))
        
        self_attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )
        
        self_attn_output = self_attn_output.transpose(1, 2).contiguous().view(B, L, D)
        self_attn_output = self.self_attn_out(self_attn_output)
        
        tgt = tgt + self.dropout(self_attn_output)
        tgt = self.norm1(tgt)
        
        # Cross-attention with encoder output
        if memory is not None:
            B_mem, L_mem, _ = memory.shape
            
            q_cross = self.cross_attn_q(tgt).view(B, L, self.nhead, self.d_head).transpose(1, 2)
            k_cross = self.cross_attn_k(memory).view(B_mem, L_mem, self.nhead, self.d_head).transpose(1, 2)
            v_cross = self.cross_attn_v(memory).view(B_mem, L_mem, self.nhead, self.d_head).transpose(1, 2)
            
            cross_attn_mask = None
            if memory_key_padding_mask is not None:
                cross_attn_mask = memory_key_padding_mask.unsqueeze(1).unsqueeze(2)
                cross_attn_mask = cross_attn_mask.expand(B, self.nhead, L, L_mem)
                cross_attn_mask = cross_attn_mask.masked_fill(cross_attn_mask, float('-inf'))
            
            cross_attn_output = F.scaled_dot_product_attention(
                q_cross, k_cross, v_cross,
                attn_mask=cross_attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
            
            cross_attn_output = cross_attn_output.transpose(1, 2).contiguous().view(B, L, D)
            cross_attn_output = self.cross_attn_out(cross_attn_output)
            
            tgt = tgt + self.dropout(cross_attn_output)
            tgt = self.norm2(tgt)
        
        # Feed-forward
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
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)

class EmbeddingReconstructionHead(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.reconstruction_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

    def forward(self, x):
        return self.reconstruction_head(x)

class DEGTransformer(nn.Module):
    def __init__(self, args, reactome_dict, biobert_dict):
        super().__init__()
        self.args = args
        embedding_dim = args.embedding_dim
        
        # Remove gene_embedding, use only reactome + biobert
        # self.gene_embedding = nn.Embedding(args.num_genes, embedding_dim, padding_idx=args.pad_token_id)  # REMOVED
        
        # Knowledge embeddings only (reactome + biobert)
        self.reactome_linear = nn.Linear(len(next(iter(reactome_dict.values()))), embedding_dim)
        self.knowledge_proj = nn.Linear(embedding_dim * 2, embedding_dim)  # Changed from 3 to 2
        
        # Value embeddings
        self.logfc_embedding = nn.Linear(1, embedding_dim)
        self.expr_embedding = nn.Linear(1, embedding_dim)
        
        # Final projection (knowledge + logfc + expr)
        self.proj = nn.Linear(embedding_dim * 3, embedding_dim)
        
        # BART-style Encoder (Bidirectional)
        self.encoder_layers = nn.ModuleList([
            FlashAttentionEncoderLayer(
                d_model=embedding_dim,
                nhead=args.nhead,
                dim_feedforward=embedding_dim * 4,
                dropout=args.dropout
            ) for _ in range(args.num_encoder_layer)
        ])
        
        # BART-style Decoder (Autoregressive)
        self.decoder_layers = nn.ModuleList([
            FlashAttentionDecoderLayer(
                d_model=embedding_dim,
                nhead=args.nhead,
                dim_feedforward=embedding_dim * 4,
                dropout=args.dropout
            ) for _ in range(args.num_decoder_layer)
        ])
        
        # Output heads
        self.logfc_head = PlainMLPHead(embedding_dim)
        self.embedding_reconstruction_head = EmbeddingReconstructionHead(embedding_dim)
        
        # Loss functions
        self.logfc_loss = nn.MSELoss()
        self.embedding_loss = nn.MSELoss()
        
        self.current_step = 0
        
        # Initialize embedding matrices
        num_total_genes = args.num_genes
        reactome_dim = len(next(iter(reactome_dict.values())))
        biobert_dim = len(next(iter(biobert_dict.values())))
        
        reactome_embedding_matrix = torch.zeros((num_total_genes, reactome_dim), dtype=torch.bfloat16, device=args.device)
        biobert_embedding_matrix = torch.zeros((num_total_genes, biobert_dim), dtype=torch.bfloat16, device=args.device)
        
        for gid, emb in reactome_dict.items():
            reactome_embedding_matrix[gid] = emb.to(args.device)
        for gid, emb in biobert_dict.items():
            biobert_embedding_matrix[gid] = emb.to(args.device)
        
        self.register_buffer('reactome_embedding_matrix', reactome_embedding_matrix)
        self.register_buffer('biobert_embedding_matrix', biobert_embedding_matrix)

    def forward(self, gene_ids, logfc, expr, mask_gene, original_logfc=None, step_count=None):
        gene_ids = gene_ids.long()
        
        # Create embeddings (reactome + biobert only, no gene_embedding)
        reactome_emb_data = self.reactome_embedding_matrix[gene_ids]
        biobert_emb_data = self.biobert_embedding_matrix[gene_ids]
        
        reactome_emb = self.reactome_linear(reactome_emb_data)
        biobert_emb = biobert_emb_data
        
        # Combine knowledge embeddings (reactome + biobert only)
        knowledge_emb = torch.cat([reactome_emb, biobert_emb], dim=-1)
        knowledge_emb = self.knowledge_proj(knowledge_emb)
        
        # Value embeddings
        logfc_emb = self.logfc_embedding(logfc.unsqueeze(-1))
        expr_emb = self.expr_embedding(expr.unsqueeze(-1))
        
        # Combine all embeddings
        original_embeddings = torch.cat([knowledge_emb, logfc_emb, expr_emb], dim=-1)
        original_embeddings = self.proj(original_embeddings)
        
        # Create noisy embeddings for masked positions (logFC part set to 0)
        noisy_embeddings = original_embeddings.clone()
        
        # Mask logFC embedding part only for selected positions
        noisy_embeddings = original_embeddings.clone()
        zero_logfc_emb = torch.zeros_like(logfc_emb)
        masked_logfc_emb = torch.where(mask_gene.unsqueeze(-1), zero_logfc_emb, logfc_emb)
        masked_combined = torch.cat([knowledge_emb, masked_logfc_emb, expr_emb], dim=-1)
        masked_projected = self.proj(masked_combined)
        noisy_embeddings[mask_gene] = masked_projected[mask_gene]

        # Attention masks
        key_padding_mask = (gene_ids == self.args.pad_token_id)
        
        # BART Encoder (Bidirectional)
        encoder_output = noisy_embeddings
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, src_key_padding_mask=key_padding_mask)
        
        # BART Decoder (Autoregressive)  
        decoder_output = noisy_embeddings  # Start with noisy embeddings
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(
                tgt=decoder_output,
                memory=encoder_output,
                tgt_key_padding_mask=key_padding_mask,
                memory_key_padding_mask=key_padding_mask
            )
        
        # Predictions
        reconstructed_embeddings = self.embedding_reconstruction_head(decoder_output)
        logfc_pred = self.logfc_head(reconstructed_embeddings)
        
        # Loss calculation
        non_pad_and_masked_indices = (gene_ids != self.args.pad_token_id) & mask_gene
        
        if non_pad_and_masked_indices.sum() > 0:
            # LogFC reconstruction loss
            masked_logfc_pred = logfc_pred[non_pad_and_masked_indices]
            # masked_logfc_labels = logfc[non_pad_and_masked_indices]

            if original_logfc is not None:
                masked_logfc_labels = original_logfc[non_pad_and_masked_indices]  
            else:
                masked_logfc_labels = logfc[non_pad_and_masked_indices]  

            logfc_loss = self.logfc_loss(masked_logfc_pred, masked_logfc_labels)
            
            # Embedding reconstruction loss
            masked_reconstructed_emb = reconstructed_embeddings[non_pad_and_masked_indices]
            masked_original_emb = original_embeddings[non_pad_and_masked_indices]
            embedding_recon_loss = self.embedding_loss(masked_reconstructed_emb, masked_original_emb)
            
            # Combined loss (weighted)
            total_loss = 0.8 * logfc_loss + 0.2 * embedding_recon_loss
        else:
            total_loss = torch.tensor(0.0, device=logfc_pred.device)
            masked_logfc_pred = torch.tensor([], device=logfc_pred.device)
            masked_logfc_labels = torch.tensor([], device=logfc.device)
        
        if step_count is not None:
            self.current_step = step_count
        elif self.training:
            self.current_step += 1
        
        return (total_loss, logfc_loss, embedding_recon_loss, masked_logfc_pred, masked_logfc_labels)

def bart_style_mask(gene_ids, logfc, mask_prob=0.15, pad_token_id=1, epoch=None, step=None):
    """Modified to only mask logFC values (set to 0)"""
    if logfc.ndim == 1:
        logfc = logfc.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, L = logfc.shape
    
    if epoch is not None and step is not None:
        seed = (epoch * 10000 + step) % (2**32)
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)
    else:
        generator = None
    
    pad_mask = (gene_ids == pad_token_id)
    zero_logfc_mask = (logfc == 0)
    valid_positions = ~(pad_mask | zero_logfc_mask)
    
    # mask only in valid_positions
    mask_all = torch.zeros_like(valid_positions, dtype=torch.bool)
    
    for b in range(B):
        valid_indices = torch.where(valid_positions[b])[0]
        if len(valid_indices) > 0:
            n_mask = int(len(valid_indices) * mask_prob)
            if n_mask > 0:
                if generator is not None:
                    perm = torch.randperm(len(valid_indices), generator=generator)
                else:
                    perm = torch.randperm(len(valid_indices))
                mask_indices = valid_indices[perm[:n_mask]]
                mask_all[b, mask_indices] = True

    # Only mask logFC values by setting them to 0
    bart_masked = logfc.clone()
    bart_masked[mask_all] = 0.0
    
    if squeeze_output:
        bart_masked = bart_masked.squeeze(0)
        mask_all = mask_all.squeeze(0)
        pad_mask = pad_mask.squeeze(0)
    
    return bart_masked, mask_all, pad_mask

# Rest of the functions remain the same
def safe_save_model_weights(model_engine, save_dir, epoch):
    if model_engine.local_rank != 0:
        return True
    
    for name, param in model_engine.module.named_parameters():
        param.data.clamp_(-60000, 60000)
        if not torch.isfinite(param).all():
            print(f"[Rank 0] Warning: Non-finite values in parameter {name} AFTER CLAMPING. Skipping model save.")
            return False
    
    try:
        print(f"[Rank 0] Preparing to save model weights for epoch {epoch}...")
        torch.cuda.empty_cache()
        output_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_engine.module.state_dict())
        save_path = os.path.join(save_dir, f"model_weights_epoch_{epoch}.pt")
        torch.save(output_state_dict, save_path)
        print(f"[Rank 0] Model weights for epoch {epoch} saved successfully to {save_path}")
        return True
    except Exception as e:
        print(f"[Rank 0] Failed to save model weights at epoch {epoch}: {e}")
        if "out of memory" in str(e).lower():
            print("[Rank 0] This is likely a CPU Out-of-Memory error during state_dict consolidation.")
            print("[Rank 0] The system does not have enough RAM to gather all model parameters on rank 0.")
        return False

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

#============================================================
# MAIN
#============================================================

def main():
    parser = argparse.ArgumentParser()
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument(
        '--input_train_h5_path',
        type=str,
        required=True,
        help='Path to concatenated DE profile H5 file for pretraining.'
    )
    parser.add_argument(
        '--outputdir',
        type=str,
        required=True,
        help='Output directory for checkpoints and logs.'
    )
    parser.add_argument('--input_deg_len', type=int, default=1024)
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--nhead', type=int, default=12)
    parser.add_argument('--num_encoder_layer', type=int, default=8)
    parser.add_argument('--num_decoder_layer', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=90)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    
    args = parser.parse_args()

    global gene_id_dict, biobert_dict, reactome_dict
    gene_id_dict, biobert_dict, reactome_dict = load_knowledge_data(datadir)
    
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    
    args.device = f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu'
    
    train_dataset = MemoryMappedDataset(args.input_train_h5_path)
    
    # 98:1:1 split for train:val:test
    full_indices = range(len(train_dataset))
    train_val_indices, test_indices = train_test_split(full_indices, test_size=0.02, random_state=42)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.010204, random_state=42)
    
    outputdir = args.outputdir
    os.makedirs(outputdir, exist_ok=True)
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)
    test_subset = torch.utils.data.Subset(train_dataset, test_indices)
    
    max_gene_id = len(np.array(list(gene_id_dict.values())))
    args.pad_token_id = max_gene_id + 1
    args.mask_token_id = max_gene_id + 2
    args.num_genes = max_gene_id + 3
    
    model = DEGTransformer(args, reactome_dict, biobert_dict)
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
    )
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_subset, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=6,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=custom_collate_fn
    )
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_subset, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=6,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=custom_collate_fn
    )
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_subset, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        pin_memory=True,
        num_workers=6,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=custom_collate_fn
    )
    
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * args.num_epochs
    
    train_losses = []
    train_logfc_losses = []
    train_embed_losses = [] 
    val_losses = []
    val_logfc_losses = []
    val_embed_losses = []
    best_val_loss = float('inf')
    
    if args.local_rank == 0:
        loss_file = os.path.join(outputdir, 'loss.txt')
        with open(loss_file, 'w') as f:
            f.write('Training started at ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
            f.write('------------------------------------------------------------\n')
            f.write('Time\tEpoch\tTrainLoss\tTrainMSELoss\tTrainEmbedLoss\tValLoss\tValMSELoss\tValEmbedLoss\tValCorr\tValSpCorr\n')
    
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        stop_training = torch.tensor(0, device=args.device)
        
        train_sampler.set_epoch(epoch)
        model_engine.train()
        epoch_train_loss = 0
        epoch_logfc_loss = 0
        epoch_embed_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}", disable=(args.local_rank != 0))
        
        for batch_idx, batch in enumerate(progress_bar):
            gene_ids = batch['gene_ids'].to(args.device)
            logfc = batch['logfc'].to(args.device).bfloat16()
            ctrl_expr = batch['ctrl_exp'].to(args.device).bfloat16()
            
            logfc_masked, mask_gene, pad_mask_returned = bart_style_mask(
                gene_ids, logfc, mask_prob=0.15, 
                pad_token_id=args.pad_token_id, epoch=epoch, step=batch_idx
            )
            
            current_step = epoch * len(train_dataloader) + batch_idx
            
            loss, logfc_loss, embedding_recon_loss, _, _ = model_engine(
                gene_ids, logfc_masked, ctrl_expr,
                mask_gene, 
                step_count=current_step,
                original_logfc=logfc
            )
            
            model_engine.backward(loss)
            model_engine.step()
            
            epoch_train_loss += loss.item()
            epoch_logfc_loss += logfc_loss.item()
            epoch_embed_loss += embedding_recon_loss.item()
        
        num_batches = len(train_dataloader)
        train_losses.append(epoch_train_loss / num_batches)
        train_logfc_losses.append(epoch_logfc_loss / num_batches)
        train_embed_losses.append(epoch_embed_loss / num_batches)
        
        clear_memory()
        
        # Validation
        model_engine.eval()
        all_val_logfc_preds, all_val_logfc_labels = [], []
        all_masked_counts = []
        val_loss_sum = 0
        val_logfc_sum = 0
        val_embed_sum = 0
        val_sampler.set_epoch(epoch)
        
        with torch.no_grad():
            for val_batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Validating", disable=(args.local_rank != 0))):
                gene_ids = batch['gene_ids'].to(args.device)
                logfc = batch['logfc'].to(args.device).bfloat16()
                ctrl_expr = batch['ctrl_exp'].to(args.device).bfloat16()
                
                logfc_masked, val_loss_mask_gene, _ = bart_style_mask(
                    gene_ids, logfc, mask_prob=0.15,
                    pad_token_id=args.pad_token_id,
                    epoch=epoch, step=val_batch_idx
                )
                
                loss, logfc_loss, embedding_recon_loss, logfc_pred, logfc_labels = model_engine(
                    gene_ids, logfc_masked, 
                    ctrl_expr, 
                    val_loss_mask_gene,
                    original_logfc=logfc
                )
                
                logfc_pred_cpu = logfc_pred.detach().cpu()
                logfc_labels_cpu = logfc_labels.detach().cpu()
                
                all_val_logfc_preds.append(logfc_pred_cpu)
                all_val_logfc_labels.append(logfc_labels_cpu)
                all_masked_counts.append(val_loss_mask_gene.sum().item())

                val_loss_sum += loss.item()
                val_logfc_sum += logfc_loss.item()
                val_embed_sum += embedding_recon_loss.item()
        
        num_val_batches = len(val_dataloader)
        avg_val_loss = val_loss_sum / num_val_batches
        val_losses.append(avg_val_loss)
        val_logfc_losses.append(val_logfc_sum / num_val_batches)
        val_embed_losses.append(val_embed_sum / num_val_batches)
        
        clear_memory()
        
        if args.local_rank == 0:
            total_masked = sum(all_masked_counts)
            print("Total masked positions in val:", total_masked)

            all_val_logfc_preds = torch.cat(all_val_logfc_preds, dim=0).float().numpy()
            all_val_logfc_labels = torch.cat(all_val_logfc_labels, dim=0).float().numpy()
            
            preds = all_val_logfc_preds.ravel()  
            labels = all_val_logfc_labels.ravel() 
            
            print("Unique preds:", np.unique(preds)[:10], " ...", len(np.unique(preds)))
            print("Unique labels:", np.unique(labels)[:10], " ...", len(np.unique(labels)))
            print("Non-zero preds count:", (preds != 0).sum(), "/", preds.size)  

            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            val_corr = pearsonr(preds, labels)[0] if len(preds) > 1 else 0
            val_spcorr = spearmanr(preds, labels)[0] if len(preds) > 1 else 0

            with open(loss_file, 'a') as f:
                f.write(f'{current_time}\t{epoch+1}\t{train_losses[-1]:.4f}\t{train_logfc_losses[-1]:.4f}\t{train_embed_losses[-1]:.4f}\t{avg_val_loss:.4f}\t{val_logfc_losses[-1]:.4f}\t{val_embed_losses[-1]:.4f}\t{val_corr:.4f}\t{val_spcorr:.4f}\n')
            
            safe_save_model_weights(model_engine, outputdir, epoch)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
                if patience_counter > args.patience:
                    print(f'Early stopping triggered at epoch {epoch+1}')
                    stop_training.fill_(1)
        
        torch.distributed.all_reduce(stop_training, op=torch.distributed.ReduceOp.MAX)
        
        if stop_training.item() == 1:
            if args.local_rank != 0:
                print(f'[Rank {args.local_rank}] Received early stopping signal. Exiting training loop.')
            break
    
    print(f"[Rank {args.local_rank}] Training loop finished. Waiting at barrier.")
    torch.distributed.barrier()
    print(f"[Rank {args.local_rank}] Passed barrier. Proceeding to final cleanup.")
    
    # Test phase after training
    model_engine.eval()
    all_test_logfc_preds, all_test_logfc_labels = [], []
    test_loss_sum = 0
    test_logfc_loss_sum =0
    test_embed_loss_sum = 0
    
    with torch.no_grad():
        for test_batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Testing", disable=(args.local_rank != 0))):
            gene_ids = batch['gene_ids'].to(args.device)
            logfc = batch['logfc'].to(args.device).bfloat16()
            ctrl_expr = batch['ctrl_exp'].to(args.device).bfloat16()
            
            logfc_masked, test_mask_gene, _ = bart_style_mask(
                gene_ids, logfc, mask_prob=0.15,
                pad_token_id=args.pad_token_id,
                epoch=0, step=test_batch_idx  # epoch=0 for test
            )
            
            loss, logfc_loss, embedding_recon_loss, logfc_pred, logfc_labels = model_engine(
                gene_ids, logfc_masked, ctrl_expr, test_mask_gene,
                original_logfc=logfc)
            
            logfc_pred_cpu = logfc_pred.detach().cpu()
            logfc_labels_cpu = logfc_labels.detach().cpu()
            
            all_test_logfc_preds.append(logfc_pred_cpu)
            all_test_logfc_labels.append(logfc_labels_cpu)
            
            test_loss_sum += loss.item()
            test_logfc_loss_sum += logfc_loss.item()
            test_embed_loss_sum += embedding_recon_loss.item()
    
    num_test_batches = len(test_dataloader)
    avg_test_loss = test_loss_sum / num_test_batches
    avg_test_logfc_loss = test_logfc_loss_sum / num_test_batches
    avg_test_embed_loss = test_embed_loss_sum / num_test_batches
    
    clear_memory()
    
    if args.local_rank == 0:
        all_test_logfc_preds = torch.cat(all_test_logfc_preds, dim=0).float().numpy()
        all_test_logfc_labels = torch.cat(all_test_logfc_labels, dim=0).float().numpy()
        
        test_corr = pearsonr(all_test_logfc_preds, all_test_logfc_labels)[0] if len(all_test_logfc_preds) > 1 else 0
        test_spcorr = spearmanr(all_test_logfc_preds, all_test_logfc_labels)[0] if len(all_test_logfc_preds) > 1 else 0
        
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(loss_file, 'a') as f:
            f.write(f'{current_time}\tTest\t-\t{avg_test_loss:.4f}\t{avg_test_logfc_loss:.4f}\t{avg_test_embed_loss:.4f}\t{test_corr:.4f}\t{test_spcorr:.4f}\n')

        best_epoch = None
        best_val = None
        try:
            with open(loss_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('Training started') or line.startswith('---') or line.startswith('Time\t'):
                        continue
                    parts = line.split('\t')
                    if len(parts) < 6:
                        continue
                    epoch_str = parts[1]
                    if not epoch_str.isdigit():
                        continue
                    val_loss_str = parts[5]
                    try:
                        val_loss = float(val_loss_str)
                    except ValueError:
                        continue
                    epoch_num = int(epoch_str) - 1
                    if best_val is None or val_loss < best_val:
                        best_val = val_loss
                        best_epoch = epoch_num
        except FileNotFoundError:
            best_epoch = None

        if best_epoch is not None:
            src_path = os.path.join(outputdir, f"model_weights_epoch_{best_epoch}.pt")
            dst_path = os.path.join(outputdir, f"best_model_weights_epoch_{best_epoch}.pt")
            if os.path.exists(src_path):
                shutil.copyfile(src_path, dst_path)


if __name__ == "__main__":
    main()
