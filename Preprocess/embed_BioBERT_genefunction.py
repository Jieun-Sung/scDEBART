import argparse
import os
import pickle
import re

import numpy as np
import requests
import torch
from transformers import AutoModel, AutoTokenizer
import mygene


def get_gene_description_from_ensembl(ensembl_id):
    url = f"https://rest.ensembl.org/lookup/id/{ensembl_id}"
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code == 200:
        data = response.json()
        description = data.get("description", "")
        display_name = data.get("display_name", "")
        return f"{display_name}: {description}"
    return f"Gene ID: {ensembl_id}"


def get_gene_summary(ensembl_id):
    mg = mygene.MyGeneInfo()
    try:
        gene_info = mg.getgene(ensembl_id, fields="name,symbol,summary")
    except Exception as exc:
        return f"API Error for {ensembl_id}: {exc}"
    if not gene_info:
        return get_gene_description_from_ensembl(ensembl_id)
    name = gene_info.get("name", "N/A")
    symbol = gene_info.get("symbol", "N/A")
    summary = gene_info.get("summary", "No summary available.")
    summary = re.sub(r"\s*\[provided by RefSeq.*?\]\.*", "", summary)
    return f"Gene: {name} (Symbol: {symbol}, ID: {ensembl_id}). [Summary]: {summary}"


def batch_gene_embedding(gene_descriptions, model, tokenizer, device, batch_size=32):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(gene_descriptions), batch_size):
            batch = gene_descriptions[i : i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
    return np.concatenate(embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Generate BioBERT embeddings for Ensembl gene descriptions."
    )
    parser.add_argument("--datadir", type=str, default="./data")
    parser.add_argument("--outputdir", type=str, default="./data")
    args = parser.parse_args()

    datadir = os.path.abspath(args.datadir)
    outputdir = os.path.abspath(args.outputdir)
    os.makedirs(outputdir, exist_ok=True)

    ens2geneid_path = os.path.join(datadir, "ens2geneid.pkl")
    if not os.path.exists(ens2geneid_path):
        raise FileNotFoundError(f"Missing ens2geneid.pkl at {ens2geneid_path}")

    with open(ens2geneid_path, "rb") as f:
        gene_id_dict = pickle.load(f)

    genel = list(gene_id_dict.values())

    model_name = "dmis-lab/biobert-base-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    gene_descriptions = [get_gene_summary(gid) for gid in genel]
    embeddings = batch_gene_embedding(
        gene_descriptions, model, tokenizer, device=device, batch_size=32
    )

    embedding_dict = {gene: emb for gene, emb in zip(genel, embeddings)}

    out_path = os.path.join(outputdir, "BioBERT_gene_embedding_dict.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(embedding_dict, f)


if __name__ == "__main__":
    main()
