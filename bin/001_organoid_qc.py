#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2024-02-28'
__version__ = '0.0.1'

####################################################################
########## Stand alone script to QC the organoids ##################
####################################################################
# QC params are the same as biopsies (min UMI, min Gene)

# Do this in bbknn environment
# Change dir
import os
os.chdir("/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/scripts/scRNAseq/Organoids")

# Import libraries
import anndata as ad
import scanpy as sc
import pandas as pandas
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mp
from matplotlib import pyplot as plt
import kneed as kd
import bbknn as bbknn
import argparse

seed_value = 0
# 0. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)

def parse_options():    
    parser = argparse.ArgumentParser(
        description="""
            Compute optimum NN from the desired matrix with bbknn
            """
    )
    
    parser.add_argument(
        '-h5', '--h5ad_file',
        action='store',
        dest='h5ad_file',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-fkp', '--filter_keras_probability',
        action='store',
        dest='filter_keras_probability',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-feo', '--filt_epi_only',
        action='store',
        dest='filt_epi_only',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-cc', '--category_col',
        action='store',
        dest='category_col',
        required=True,
        help=''
    )
    
    parser.add_argument(
        '-nvg', '--n_variable_genes',
        action='store',
        dest='n_variable_genes',
        required=True,
        help=''
    )
        
    parser.add_argument(
        '-rpg', '--remove_problem_genes',
        action='store',
        dest='remove_problem_genes',
        required=True,
        help=''
    )

    parser.add_argument(
        '-ct', '--condition_test',
        action='store',
        dest='condition_test',
        required=True,
        help=''
    )
    
    return parser.parse_args()


def main():
    inherited_options = parse_options()
    fpath = inherited_options.h5ad_file
    filter_keras_probability = float(inherited_options.filter_keras_probability)
    filt_epi_only = inherited_options.filt_epi_only
    category_col = inherited_options.category_col
    n_variable_genes = int(inherited_options.n_variable_genes)
    remove_problem_genes = inherited_options.remove_problem_genes
    condition_test = inherited_options.condition_test

    # testing
    #fpath = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/marcus_analysis/organoid_analysis/yascp_analysis/results/celltype/keras_celltype/full/full___cellbender_fpr0pt1-scrublet-ti_freeze003_prediction-predictions.h5ad"
    #filter_keras_probability = 0.5 # If no filter, set to 0
    #filt_epi_only = "yes"
    #category_col = "sum_category__machine"
    #n_variable_genes = "2000"
    #remove_problem_genes = "yes"
    #condition_test = "IFNg" # Can be "IFNg,TNFa" for both, or either on their own
    #python bin/001_organoid_qc.py \
    #   --h5ad "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/marcus_analysis/organoid_analysis/yascp_analysis/results/celltype/keras_celltype/full/full___cellbender_fpr0pt1-scrublet-ti_freeze003_prediction-predictions.h5ad" \
    #   --filter_keras_probability 0.5 \
    #   --filt_epi_only "yes" \
    #   --category_col "sum_category__machine" \
    #   --n_variable_genes 2000 \
    #   --remove_problem_genes "yes" \
    #   --condition_test "IFNg"

    # Load in the data
    adata = sc.read_h5ad(fpath)

    # Append the category information
    clean_cluster_conv = "/lustre/scratch126/humgen/projects/sc-eqtl-ibd/analysis/bradley_analysis/cluster_annotations/data-clean_annotation.csv"
    annot = pd.read_csv(clean_cluster_conv)
    annot.rename(columns={"label__machine": "predicted_celltype"}, inplace=True)
    adata.obs = adata.obs.reset_index()
    adata.obs = adata.obs.merge(annot, how="left", on="predicted_celltype")
    adata.obs.set_index("index", inplace=True)
        
    # Compute the category confidence based on the sum of probabilities within a category
    print("Summing category probabilities")
    cats = np.unique(annot['category__machine'])
    for c in cats:
        print(c)
        labels = np.unique(annot[annot['category__machine'] == c]['predicted_celltype'])
        adata.obs['sum_' + c] = adata.obs['probability__' + labels].sum(axis=1)

    # Find the maximum category based on these sum values
    adata.obs['sum_category__machine'] = adata.obs['sum_'+cats].idxmax(axis=1) 

    # Add the max probability of these summed categories
    adata.obs['sum_category__machine_probability'] = adata.obs['sum_'+cats].max(axis=1)

    # Also add the organoid metadata
    meta = pd.read_csv("data/sample_to_individual_mapping.txt", sep = " ")
    meta = meta.assign(donor_vcf_ids=meta['donor_vcf_ids'].str.split(',')).explode('donor_vcf_ids')
    meta['sequence_number'] = meta.groupby('experiment_id').cumcount()
    meta['experiment_id'] = meta['experiment_id'] + "__donor" + meta['sequence_number'].astype(str)
    # Adjust one
    meta['experiment_id'] = meta['experiment_id'].replace("6123STDY12147213__donor0", "6123STDY12147213__donor")
    adata.obs['experiment_id'].isin(meta['experiment_id']).all()
    # Merge
    meta = meta[["experiment_id", "donor_vcf_ids"]]
    adata.obs = adata.obs.reset_index()
    adata.obs = adata.obs.merge(meta, how="left", on="experiment_id")
    adata.obs.set_index("index", inplace=True)
    # Add the disease stimulation status
    # From organoid meta: https://docs.google.com/spreadsheets/d/1dLsk5WUawRJVLVm2MDfpFlCSjcVAZDuI5QlZI0xX0tc/edit#gid=0
    # Samples treated are: 
    ifn_treated = ["6123STDY12147213", "6123STDY12147214", "6123STDY12147215", "6123STDY12472441", "6123STDY12472442", "6123STDY12472443"]
    tnf_treated = ["6123STDY12472437", "6123STDY12472438"]
    adata.obs['condition'] = "control"
    adata.obs.loc[adata.obs['convoluted_samplename'].isin(ifn_treated), "condition"] = "IFNg"
    adata.obs.loc[adata.obs['convoluted_samplename'].isin(tnf_treated), "condition"] = "TNFa"

    # Map back to disease status
    # This is taken straight from the gut metadata
    cd_samps = ["3981576772874", "3981576796726", "3981576869727", "F2", "SC0010", "SC0011"]
    adata.obs['disease_status'] = "healthy"
    adata.obs.loc[adata.obs['donor_vcf_ids'].isin(cd_samps), "disease_status"] = "cd"

    # Filter for condition if desired
    condition_keep = condition_test.split(",")
    condition_keep.append("control")
    adata = adata[adata.obs['condition'].isin(condition_keep)]

    # Save the outdir
    qc_path="results/QC"
    if not os.path.exists(qc_path):
        os.makedirs(qc_path)

    # Filter cells 
    sc.pp.filter_cells(adata, min_genes=100) # Same as the biopsies

    # Plot the distribution of probabilities across categories
    if category_col == "sum_category__machine":
        probability_col = "sum_category__machine_probability"
    else:
        probability_col = "predicted_celltype_probability"

    cats = np.unique(adata.obs[category_col])
    for c in cats:
        data = adata.obs[adata.obs[category_col] == c][probability_col]
        sns.distplot(data, hist=False, rug=True, label=c)

    plt.legend()
    plt.xlabel('keras probability')
    plt.title(f"Absolute cut off (black): {filter_keras_probability}")
    plt.axvline(x = filter_keras_probability, color = 'black', linestyle = '--', alpha = 0.5)
    plt.savefig(qc_path + '/probability_per_category', bbox_inches='tight')
    plt.clf()

    # Subset for cells with probability
    adata = adata[adata.obs[probability_col] > filter_keras_probability]

    # Interesting that there are cells strongly annotated for non-epithelial cells. 
    # Plot the proportion of categories per sample
    samp_data = np.unique(adata.obs.experiment_id, return_counts=True)
    cells_sample = pd.DataFrame({'sample': samp_data[0], 'Ncells':samp_data[1]})

    cells_sample_proportions = pd.DataFrame(columns=[cats], index=cells_sample['sample'])
    cells_sample_proportions = cells_sample_proportions.assign(Total=0)
    for s in cells_sample['sample']:
        use = adata.obs[adata.obs.experiment_id == s]
        cells_sample_proportions.loc[cells_sample_proportions.index == s, 'Total'] = use.shape[0]
        for c in cats:
            prop = use[use[category_col] == c].shape[0]/use.shape[0]
            cells_sample_proportions.loc[cells_sample_proportions.index == s, c] = prop

    cells_sample_proportions = cells_sample_proportions.drop('Total', axis=1)

    plt.figure(figsize=(16, 12))
    fig,ax = plt.subplots(figsize=(16,12))
    cells_sample_proportions.plot(
        kind = 'barh',
        stacked = True,
        title = 'Stacked Bar Graph',
        mark_right = True)
    plt.legend(cats, bbox_to_anchor=(1.0, 1.0))
    plt.suptitle('')
    plt.xlabel('Proportion of cell category/sample')
    plt.ylabel('')
    plt.savefig(f"{qc_path}/prop_cats_per_sample.png", bbox_inches='tight')
    plt.clf()

    # Divide this by disease status
    statuses = np.unique(adata.obs["disease_status"])
    for s in statuses:
        status_samps = np.unique(adata.obs[adata.obs['disease_status'] == s]['experiment_id'])
        temp = cells_sample_proportions[cells_sample_proportions.index.isin(status_samps)]
        plt.figure(figsize=(16, 12))
        fig,ax = plt.subplots(figsize=(16,12))
        temp.plot(
            kind = 'barh',
            stacked = True,
            title = 'Stacked Bar Graph',
            mark_right = True)
        plt.legend(cats, bbox_to_anchor=(1.0, 1.0))
        plt.suptitle('')
        plt.xlabel('Proportion of cell category/sample - ' + s)
        plt.ylabel('')
        plt.savefig(f"{qc_path}/prop_cats_per_sample_{s}.png", bbox_inches='tight')
        plt.clf()

    # Also divide by condition
    for s in condition_keep:
        status_samps = np.unique(adata.obs[adata.obs['condition'] == s]['experiment_id'])
        temp = cells_sample_proportions[cells_sample_proportions.index.isin(status_samps)]
        plt.figure(figsize=(16, 12))
        fig,ax = plt.subplots(figsize=(16,12))
        temp.plot(
            kind = 'barh',
            stacked = True,
            title = 'Stacked Bar Graph',
            mark_right = True)
        plt.legend(cats, bbox_to_anchor=(1.0, 1.0))
        plt.suptitle('')
        plt.xlabel('Proportion of cell category/sample - ' + s)
        plt.ylabel('')
        plt.savefig(f"{qc_path}/prop_cats_per_sample_{s}.png", bbox_inches='tight')
        plt.clf()

    # Filter for cells annotated as epithelial cells if desirved, using the desired specification of category
    if filt_epi_only == "yes":
        epithelial_cats = ["Enterocyte", "Secretory", "Stem_cells"]
        if category_col == "sum_category__machine":
            epithelial_cats = ['sum_' + element for element in epithelial_cats]
        
        adata = adata[adata.obs[category_col].isin(epithelial_cats)]


    # Look at cells per sample and depth 
    mean_df = adata.obs.groupby('experiment_id')[['n_genes_by_counts', 'total_counts']].mean().reset_index()
    count_df = adata.obs.groupby('experiment_id').size().reset_index(name='n_cells')
    result_df = pd.merge(mean_df, count_df, on='experiment_id')
    status_df = adata.obs[["experiment_id", "disease_status"]].reset_index()
    status_df = status_df[["experiment_id", "disease_status"]].drop_duplicates()
    result_df = result_df.merge(status_df, on='experiment_id')
    condition_df = adata.obs[["experiment_id", "condition"]].reset_index()
    condition_df = condition_df[["experiment_id", "condition"]].drop_duplicates()
    result_df = result_df.merge(condition_df, on='experiment_id')
    print(result_df.head())
    color_dict = {'cd': 'orange', 'healthy': 'darkblue'}
    colors = [color_dict[status] for status in result_df['disease_status']]


    plt.figure(figsize=(8, 6))
    plt.scatter(result_df["total_counts"], result_df["n_genes_by_counts"],  s=result_df["n_cells"], alpha=0.6, c=colors)
    plt.xlabel('Mean counts / cell')
    plt.ylabel('Mean genes detected / cell')
    sizes_legend = [min(result_df["n_cells"]), np.median(result_df["n_cells"]), max(result_df["n_cells"])]  # Define sizes for legend
    sizes_labels = [f"min: {min(result_df['n_cells'])}", f"median: {np.median(result_df['n_cells'])}", f"max: {max(result_df['n_cells'])}"]  # Labels for legend
    legend_handles1 = [plt.scatter([], [], s=size, alpha=0.5, label=label) for size, label in zip(sizes_legend, sizes_labels)]
    legend_handles2 = [plt.Line2D([0], [0], marker='o', color='w', markersize=10, label=status, markerfacecolor=color) 
                    for status, color in color_dict.items()]
    all_handles = legend_handles1 + legend_handles2
    all_labels = sizes_labels + list(color_dict.keys())
    plt.legend(handles=all_handles, labels=all_labels, title='Legend', loc='center left', bbox_to_anchor=(1, 0.5))

    # Label points with n_genes_by_counts < 25000
    for i, row in result_df[result_df['total_counts'] < 25000].iterrows():
        plt.text(row['total_counts'], row['n_genes_by_counts'], str(row['experiment_id']))

    plt.savefig(f"{qc_path}/sample_mean_counts_ngenes_all_status.png", bbox_inches='tight')
    plt.clf()

    # Also colour by condition
    conditon_dict = {'IFNg': 'pink', 'control': 'green', 'TNFa': 'red'}
    colors_condition = [conditon_dict[condition] for condition in result_df['condition']]
    plt.figure(figsize=(8, 6))
    plt.scatter(result_df["total_counts"], result_df["n_genes_by_counts"],  s=result_df["n_cells"], alpha=0.6, c=colors_condition)
    plt.xlabel('Mean counts / cell')
    plt.ylabel('Mean genes detected / cell')
    sizes_legend = [min(result_df["n_cells"]), np.median(result_df["n_cells"]), max(result_df["n_cells"])]  # Define sizes for legend
    sizes_labels = [f"min: {min(result_df['n_cells'])}", f"median: {np.median(result_df['n_cells'])}", f"max: {max(result_df['n_cells'])}"]  # Labels for legend
    legend_handles1 = [plt.scatter([], [], s=size, alpha=0.5, label=label) for size, label in zip(sizes_legend, sizes_labels)]
    legend_handles2 = [plt.Line2D([0], [0], marker='o', color='w', markersize=10, label=condition, markerfacecolor=color) 
                    for condition, color in conditon_dict.items()]
    all_handles = legend_handles1 + legend_handles2
    all_labels = sizes_labels + list(conditon_dict.keys())
    plt.legend(handles=all_handles, labels=all_labels, title='Legend', loc='center left', bbox_to_anchor=(1, 0.5))


    # Label points with n_genes_by_counts < 25000
    for i, row in result_df[result_df['total_counts'] < 25000].iterrows():
        plt.text(row['total_counts'], row['n_genes_by_counts'], str(row['experiment_id']))

    plt.savefig(f"{qc_path}/sample_mean_counts_ngenes_all_condition.png", bbox_inches='tight')
    plt.clf()

    # Save a table of cell abundances in final shape
    cell_counts = adata.obs.groupby('predicted_celltype').size().reset_index(name='n_cells')
    label_cat = adata.obs.reset_index()[["predicted_celltype", "category__machine"]].drop_duplicates()
    cell_counts = cell_counts.merge(label_cat, on="predicted_celltype")
    cell_counts = cell_counts.sort_values(by='n_cells', ascending=False)
    cell_counts.to_csv(f"{qc_path}/ncells_after_filt.csv")

    # How do other parameters look at this threshold
    other_filters = ["pct_counts_gene_group__mito_transcript", "log1p_n_genes_by_counts", "log1p_total_counts"]
    for f in other_filters:
        for c in cats:
            data = adata.obs.loc[adata.obs[category_col] == c,f]
            sns.distplot(data, hist=False, rug=True, label=c)
        plt.legend()
        plt.xlabel(f)
        plt.title(f"{f} - probability > {filter_keras_probability}")
        plt.savefig(f"{qc_path}/{f}_per_category_keras_filt.png", bbox_inches='tight')
        plt.clf()

    # Plot the number of cells per sample after applying or not appplying these filters
    samp_data = np.unique(adata.obs.experiment_id, return_counts=True)
    cells_sample = pd.DataFrame({'sample': samp_data[0], 'Ncells':samp_data[1]})
    sns.distplot(cells_sample['Ncells'], hist=True, rug=True, kde=False)
    plt.xlabel('Cells per sample')
    plt.title(f"Number of cells per sample - probability > {filter_keras_probability}")
    plt.savefig(f"{qc_path}/Ncells_per_sample_keras_filt.png", bbox_inches='tight')

    # Filter genes
    sc.pp.filter_genes(adata, min_cells=5)

    # Keep a copy of the raw counts
    adata.layers['counts'] = adata.X.copy()
    print("Copied counts")

    # Calulate the CP10K expression
    sc.pp.normalize_total(adata, target_sum=1e4)
    print("CP10K")

    # Now normalise the data to identify highly variable genes (Same as in Tobi and Monika's paper)
    sc.pp.log1p(adata)
    print("log1p")

    # Store this
    adata.layers['log1p_cp10k'] = adata.X.copy()

    # Find highly variable
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=int(n_variable_genes))
    print("Found highly variable")

    # Check for intersection of IG, MT and RP genes in the HVGs
    print("IG")
    print(np.unique(adata.var[adata.var.gene_symbols.str.contains("IG[HKL]V|IG[KL]J|IG[KL]C|IGH[ADEGM]")].highly_variable, return_counts=True))
    print("MT")
    print(np.unique(adata.var[adata.var.gene_symbols.str.contains("^MT-")].highly_variable, return_counts=True))
    print("RP")
    print(np.unique(adata.var[adata.var.gene_symbols.str.contains("^RP")].highly_variable, return_counts=True))


    if remove_problem_genes == "yes":
        condition = adata.var['gene_symbols'].str.contains('IG[HKL]V|IG[KL]J|IG[KL]C|IGH[ADEGM]|^MT-|^RP', regex=True)
        adata.var.loc[condition, 'highly_variable'] = False

    # Scale
    sc.pp.scale(adata, max_value=10)
    print("Scaled")

    # Print the final shape of the high QC data
    print(f"The final shape of the high QC data is: {adata.shape}")

    # Run PCA
    sc.tl.pca(adata, svd_solver='arpack')

    # PCA path
    pca_path="results/PCA"
    if not os.path.exists(pca_path):
        os.makedirs(pca_path)

    sc.settings.figdir=pca_path

    # Plot PCA
    colby=["disease_status", "experiment_id", "category__machine", category_col, "predicted_celltype", "predicted_celltype_probability", "pct_counts_gene_group__mito_transcript", "log1p_n_genes_by_counts", "log1p_total_counts", "sum_category__machine"]
    for c in colby:
        sc.pl.pca(adata, color=c, save="_" + c + ".png")

    # PLot Elbow plot
    sc.pl.pca_variance_ratio(adata, log=True, save=True, n_pcs = 50)

    #  Determine the optimimum number of PCs
    # Extract PCs
    pca_variance=pd.DataFrame({'x':list(range(1, 51, 1)), 'y':adata.uns['pca']['variance']})
    # Identify 'knee'
    knee=kd.KneeLocator(x=list(range(1, 51, 1)), y=adata.uns['pca']['variance'], curve="convex", direction = "decreasing")
    knee_point = knee.knee
    elbow_point = knee.elbow
    print('Knee: ', knee_point) 
    print('Elbow: ', elbow_point)
    # Use the 'knee' + 5 additional PCs
    nPCs = knee_point + 5
    print("The number of PCs used for this analysis is {}".format(nPCs))

    # batch correct with bbkkn
    bbknn.bbknn(adata, batch_key='experiment_id', n_pcs=nPCs, use_rep="X_pca", neighbors_within_batch=1)

    # Compute UMAP
    sc.tl.umap(adata)

    # Plot UMAP
    umap_path="results/UMAP"
    if not os.path.exists(umap_path):
        os.makedirs(umap_path)

    sc.settings.figdir=umap_path

    for c in colby:
        sc.pl.umap(adata, color = c, save="_" +c + ".png")

    # Save the object
    obj_path="results/objects"
    if not os.path.exists(obj_path):
        os.makedirs(obj_path)

    adata.write(f"{obj_path}/adata_filtered_batched_umapd.h5ad")

    # Plot MHC Class 1 gene expression
    mhc1_genes = ["HLA-A", "HLA-B", "HLA-C", "HLA-E", "HLA-F", "B2M", "TAP1", "TAPBP", "CANX", "CALR", "PSME2", "PSMB8", "PSMB9",
                "PSMA5", "PSME1", "PSMA3", "PSMD8", "PSMA7", "PSMD1", "PSMB6", "PSMB3", "PSMB1", "PDIA3", "HSPA5", "CTSS", "SEC61G",
                "CYBA", "SEC61B", "RNF213", "HMGB1", "RPS27A", "VAMP8", "UBC", "SKP1", "RNF7", "ANAPC11"]

    print("Missing genes")
    np.setdiff1d(mhc1_genes, adata.var['gene_symbols'])

    # For each category, plot a violin plot of these genes across IFNg treatment
    cats = np.unique(adata.obs[category_col])

    violin_path = pca_path="results/violin"
    if not os.path.exists(violin_path):
        os.makedirs(violin_path)

    sc.settings.figdir=violin_path
    for c in cats:
        print(f"~~~~~~ {c} ~~~~~~")
        temp = adata[adata.obs[category_col] == c]
        for g in mhc1_genes:
            print(g)
            ens = adata.var[adata.var['gene_symbols'] == g].index[0]
            sc.pl.violin(temp, keys=ens, layer="counts", groupby='condition', save=f"_{c}_{g}_across_condition.png")


# Execute
if __name__ == '__main__':
    main()