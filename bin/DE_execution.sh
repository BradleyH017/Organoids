#! bin/bash
# Execution of DE

#module load HGI/common/nextflow/23.04.3
export CUR_DIR="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Organoids"
export REPO_DIR="../sc_ti_atlas" # Location of the clone of git@github.com:andersonlab/sc_ti_atlas.git
export SC_TI_OUTDIR="$CUR_DIR/results"
mkdir -p "${SC_TI_OUTDIR}"


conda activate sc_ti_dge
nextflow run \
    "${REPO_DIR}/sc_nf_diffexpression/main.nf" \
     -profile "lsf" \
     --file_anndata "${CUR_DIR}/results/objects/adata_filtered_batched_umapd.h5ad" \
     --output_dir "${SC_TI_OUTDIR}" \
     -params-file "${CUR_DIR}/configs/dge_config.yml" \
     -resume