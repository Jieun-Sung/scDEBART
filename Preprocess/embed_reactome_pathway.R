#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(GSEABase)
  library(data.table)
})

option_list <- list(
  make_option(c("--datadir"), type = "character", default = "./data",
              help = "Directory containing pathway and gene mapping files."),
  make_option(c("--outputdir"), type = "character", default = "./data",
              help = "Directory to write c2_all_2024_1_membership_ens2pathway.tsv")
)

opt <- parse_args(OptionParser(option_list = option_list))

datadir <- normalizePath(opt$datadir, winslash = "/", mustWork = FALSE)
outputdir <- normalizePath(opt$outputdir, winslash = "/", mustWork = FALSE)
if (!dir.exists(outputdir)) {
  dir.create(outputdir, recursive = TRUE)
}

gmt_candidates <- c(
  file.path(datadir, "c2.all.v2024.1.Hs.entrez.gmt"),
  file.path(datadir, "c2.all.v2024.1.Hs.entrez.gmt.gz")
)
gmt_path <- gmt_candidates[file.exists(gmt_candidates)][1]
if (is.na(gmt_path)) {
  stop("Missing GMT file. Expected c2.all.v2024.1.Hs.entrez.gmt in --datadir.")
}

pcg_candidates <- c(
  file.path(datadir, "human_pcg_info.tsv"),
  file.path(datadir, "human_pcg_info.csv"),
  file.path(datadir, "human_pcg_info.txt")
)
pcg_path <- pcg_candidates[file.exists(pcg_candidates)][1]
if (is.na(pcg_path)) {
  stop("Missing protein-coding gene file. Expected human_pcg_info.tsv in --datadir.")
}

pcg <- if (grepl("\\.csv$", pcg_path)) {
  fread(pcg_path)
} else {
  fread(pcg_path, sep = "\t")
}

if (!all(c("ENTREZID", "ENSEMBL") %in% colnames(pcg))) {
  stop("pcg file must contain ENTREZID and ENSEMBL columns.")
}

message("Loading pathways from GMT...")
c2_path <- getGmt(gmt_path)
c2_path <- setNames(lapply(c2_path, geneIds), sapply(c2_path, setName))
c2_path <- lapply(c2_path, function(x) pcg[pcg$ENTREZID %in% as.integer(x), ]$ENSEMBL)

all_genes <- unique(unlist(c2_path))
message("Building membership matrix...")
mat_list <- lapply(c2_path, function(genes) as.integer(all_genes %in% genes))
mat <- do.call(cbind, mat_list)
rownames(mat) <- all_genes
colnames(mat) <- names(c2_path)
df <- as.data.frame(mat)

out_path <- file.path(outputdir, "c2_all_2024_1_membership_ens2pathway.tsv")
fwrite(df, out_path, sep = "\t", quote = FALSE, row.names = TRUE)
message(paste("Wrote", out_path))
