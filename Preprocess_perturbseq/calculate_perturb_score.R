suppressMessages({
  Sys.setenv(OMP_NUM_THREADS = 1)
  Sys.setenv(OPENBLAS_NUM_THREADS = 1)
  Sys.setenv(MKL_NUM_THREADS = 1)
  Sys.setenv(NUMEXPR_NUM_THREADS = 1)
  library(Seurat)
  library(scMAGeCK)
  library(Matrix)
  library(dplyr)
  library(parallel)
  library(data.table)
})

parse_args <- function(args) {
  get_val <- function(flag, default = NULL) {
    idx <- which(args == flag)
    if (length(idx) == 0) return(default)
    if (idx == length(args)) return(default)
    return(args[idx + 1])
  }
  list(
    dataset = get_val("--dataset"),
    ncore = as.integer(get_val("--ncore", "1")),
    inputdir = get_val("--inputdir"),
    outputdir = get_val("--outputdir"),
    keep_per_gene = toupper(get_val("--keep_per_gene", "FALSE"))
  )
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
if (is.null(args$dataset) || is.null(args$inputdir) || is.null(args$outputdir)) {
  cat("Usage: Rscript calculate_perturb_score.R --dataset <NAME> --ncore <N> --inputdir <EXPORT_DIR> --outputdir <PS_SCORE_DIR> [--keep_per_gene TRUE|FALSE]\n")
  quit(status = 1)
}

dataset_id <- args$dataset
ncore <- args$ncore
inputdir <- args$inputdir
outputdir <- args$outputdir

per_gene_dir <- file.path(outputdir, "per_gene_result")
if (!dir.exists(per_gene_dir)) {
  dir.create(per_gene_dir, recursive = TRUE)
}

expr <- readMM(file.path(inputdir, "matrix.mtx"))
meta <- read.csv(file.path(inputdir, "metadata.csv"), row.names = 1)
features <- read.csv(file.path(inputdir, "features.csv"), row.names = 1)

expr <- t(expr)
rownames(expr) <- rownames(features)
colnames(expr) <- rownames(meta)

all_pert_genes <- setdiff(unique(meta$perturb_gene), "non-targeting")

res_list <- mclapply(seq_along(all_pert_genes), mc.cores = ncore, mc.preschedule = FALSE, function(i) {
  batch_gene <- all_pert_genes[i]
  message(sprintf("[%d/%d] Running scMAGeCK for %s", i, length(all_pert_genes), batch_gene))

  ctrl_cells <- rownames(meta)[meta$perturb_gene == "non-targeting"]
  pert_cells <- rownames(meta)[meta$perturb_gene %in% batch_gene]
  sel_cells <- c(pert_cells, ctrl_cells)

  expr_f <- expr[, sel_cells]
  meta_f <- meta[sel_cells, , drop = FALSE]

  seurat_obj <- CreateSeuratObject(counts = expr_f, meta.data = meta_f)
  all_levels <- unique(c(as.character(batch_gene), "non-targeting"))
  seurat_obj$perturb_gene <- factor(seurat_obj$perturb_gene, levels = all_levels)
  seurat_obj@active.ident <- seurat_obj$perturb_gene

  bc_frame <- data.frame(
    cell = rownames(meta_f),
    barcode = meta_f$sg_guide,
    sgrna = meta_f$sg_guide,
    gene = ifelse(meta_f$perturb_gene %in% batch_gene, batch_gene, "non-targeting"),
    read_count = 1,
    umi_count = meta_f$UMI_count_sum
  )

  seurat_obj <- NormalizeData(seurat_obj)
  seurat_obj <- FindVariableFeatures(seurat_obj)
  seurat_obj <- ScaleData(seurat_obj)

  target_gene <- setdiff(batch_gene %>% unique, "non-targeting")
  control_label <- "non-targeting"

  outputdir_final <- file.path(per_gene_dir, batch_gene)
  if (!dir.exists(outputdir_final)) {
    dir.create(outputdir_final, recursive = TRUE)
  }

  res <- scmageck_eff_estimate(
    rds_object = seurat_obj,
    bc_frame = bc_frame,
    perturb_gene = target_gene,
    non_target_ctrl = control_label,
    outputdir = outputdir_final,
    ncores = 1
  )

  eff_mat <- res$eff_matrix %>% as.data.frame()
  saveRDS(eff_mat, file.path(outputdir_final, "effect_matrix.rds"))
  saveRDS(res, file.path(outputdir_final, "scMAGeCK_result.rds"))

  rm(seurat_obj, expr_f, meta_f, bc_frame, eff_mat, res)
  gc()

  return(TRUE)
})

files <- list.files(per_gene_dir, full.names = TRUE)
ps_res <- mclapply(files, function(f) {
  ll <- readRDS(paste0(f, "/effect_matrix.rds"))
  colnames(ll) <- "PS"
  ll$perturb_gene <- basename(f)
  meta <- read.csv(file.path(inputdir, "metadata.csv"), row.names = 1)
  control_cells <- meta %>% filter(perturb_gene == "non-targeting") %>% rownames()
  pert_ll <- ll[!rownames(ll) %in% control_cells, ]
  return(pert_ll)
}, mc.cores = ncore)

ps_res <- ps_res[sapply(ps_res, function(x) !inherits(x, "try-error") && nrow(x) > 0)]
ps_res <- do.call(rbind, ps_res)

write.table(
  ps_res,
  file.path(outputdir, "perturb_score_per_cell_per_gene.csv"),
  row.names = TRUE,
  col.names = TRUE,
  sep = ",",
  quote = FALSE
)

if (args$keep_per_gene != "TRUE") {
  unlink(per_gene_dir, recursive = TRUE, force = TRUE)
}
