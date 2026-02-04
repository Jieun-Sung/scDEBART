<p align="center">
  <img src="final_scheme.png" width="900" alt="scDEBART overview"/>
</p>

**scDEBART** is a **Transformer-based** model that predicts **perturbation-induced gene expression** from single-cell transcriptomic data.  
It is **pre-trained at scale** on **basal expression** and **log fold-change (logFC)** signals to learn gene dependencies and generalize across cell types and experimental conditions.

## What scDEBART does
- **Input:** basal (control-like) expression + perturbation information (e.g., INH/ACT)  
- **Learning signal:** large-scale **logFC** profiles (pretraining) + task-specific fine-tuning (e.g., Perturb-seq)  
- **Output:** predicted post-perturbation expression (or differential response)

> 📌 Figure: overall architecture and training pipeline (pretrain on cellxgene → fine-tune on Perturb-seq).
