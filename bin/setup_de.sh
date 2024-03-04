# From https://github.com/andersonlab/sc_nf_diffexpression/blob/main/env/README.md
# Forked and cloned git

# The repo directory.
REPO_MODULE="/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/sc_nf_diffexpression"

# Install environment using Mamba. Replace 'mamba' with 'conda' if mamba not available.
mamba env create --name sc_diff_expr --file code ${REPO_MODULE}/env/environment.yml

# Activate the new Conda environment.
conda activate sc_diff_expr

# To update environment file:
#conda env export --no-builds | grep -v prefix | grep -v name > environment.yml