import textwrap
import itertools as it


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches


def color_pallet(n):
    """construct a color pallet"""
    if(n > 12):
        colors_base = plt.cm.tab20c((np.arange(20)).astype(int))
    else:
        colors_base = plt.cm.Paired((np.arange(12)).astype(int))
    colors = [x[1] for x in zip(np.arange(n), it.cycle(colors_base))]
    return(colors)


def color_pallet_with_gray(n):
    """construct a color pallet with gray at the end"""
    return(np.vstack([color_pallet(n - 1), np.array([0, 0, 0, .5])]))


def plot_scatter(plot_d, ignore = None, save=None):
    """scatter plot given a dictionary object"""
    fig = plt.figure(figsize=(6,6))
    gs = gridspec.GridSpec(1, 1)
    fig_axs = [fig.add_subplot(sp) for sp in gs]
    if 'color' in plot_d:
        fig_axs[0].scatter(plot_d['x'], plot_d['y'], color=plot_d['color'])
    else:
        fig_axs[0].scatter(plot_d['x'], plot_d['y'])
    if ignore is None:
        ignore = set([])
    if(('title' in plot_d) and ('title' not in ignore)):
        fig_axs[0].set_title(plot_d['title'])
    if(('xlabel' in plot_d) and ('xlabel' not in ignore)):
        fig_axs[0].set_xlabel(plot_d['xlabel'])
    if(('ylabel' in plot_d) and ('ylabel' not in ignore)):
        fig_axs[0].set_ylabel(plot_d['ylabel'])
    if(('xticklabels' in plot_d) and ('xticklabels' not in ignore)):
        fig_axs[0].set_xticklabels(
            [textwrap.fill(x, 50) for x in plot_d['xticklabels']],
            rotation=270+45, rotation_mode="anchor", ha='left',
            minor=False
        )        
    gs.tight_layout(fig, rect=[0, 0, 1, 1]) 
    if save is not None:
        fig.savefig(save, bbox_inches="tight", pad_inches=0.0)


def compute_factor(eigen_vec, eigen_values):
    return np.dot(eigen_vec, np.diag(eigen_values))


def compute_contribution(factor):
    return (factor ** 2) / (np.sum(factor ** 2, axis = 0).reshape((1, factor.shape[1])))


def compute_cos(factor):
    return (factor ** 2) / (np.sum(factor ** 2, axis = 1).reshape((factor.shape[0], 1)))

def build_from_guide(degas_npz_file_path, w_xl, w_lt, out_prefix=None):
    if out_prefix is None:
        print("specify an out prefix")
        return
    degas_raw = np.load(degas_npz_file_path, allow_pickle=True)
    US = degas_raw['eigen_phe']*degas_raw['eigen_v']
    V = degas_raw['eigen_var']
    data_from_file = (US @ V.T)

    beta_l = data_from_file @ w_lt.T
    lambda_l = (w_xl.T @ data_from_file).T
    
    contribution_var = compute_contribution(beta_l)
    contribution_phe = compute_contribution(lambda_l)
    
    cos_var = compute_cos(beta_l)
    cos_phe = compute_cos(lambda_l)
    
    out_path = f'{out_prefix}_decomp_bl_ll.npz'
    np.savez(out_path, contribution_var=contribution_var, contribution_phe=contribution_phe, factor_var=beta_l, factor_phe=lambda_l,label_var=degas_all['label_var'], label_phe=degas_all['label_phe'], cos_var=cos_var, cos_phe=cos_phe, label_phe_code=degas_all['label_phe_code'], label_gene=degas_all['label_gene'], label_phe_stackedbar=degas_all['label_phe_stackedbar'], contribution_gene=degas_all['contribution_gene'])
    
    return out_path


def compute_contribution_gene(
    label_var, contribution_var, var_gene_file
):
    #variant_df = pd.read_csv(variant_tsv, sep='\t', compression='gzip')
    #var2gene_dict = dict(zip(variant_df['label_var'], variant_df['label_gene'])) 
    #var2gene_dict = dict()
    var2gene_dict = dict()
    with open(var_gene_file, 'r') as f:
        for l in f.read().split('\n'): 
            #print(l.split('\t')[1])
            var2gene_dict[l.split('\t')[0]] = l.split('\t')[1]
            

    contribution_var_df = pd.DataFrame(contribution_var)    
    contribution_var_df['gene'] = [var2gene_dict[x] for x in label_var]
    contribution_gene_df = contribution_var_df.groupby('gene').sum()
    
    return contribution_gene_df.to_numpy(), np.array(contribution_gene_df.index)


def compute_contribution_cyto(
    label_var, contribution_var, var_gene_file
):
    #variant_df = pd.read_csv(variant_tsv, sep='\t', compression='gzip')
    #var2gene_dict = dict(zip(variant_df['label_var'], variant_df['label_gene'])) 
    #var2gene_dict = dict()
    var2cyto_dict = dict()
    with open(var_gene_file, 'r') as f:
        for l in f.read().split('\n'): 
            #print(l.split('\t')[1])
            var2cyto_dict[l.split('\t')[0]] = l.split('\t')[2]
            

    contribution_var_df = pd.DataFrame(contribution_var)    
    contribution_var_df['cyto'] = [var2cyto_dict[x] for x in label_var]

    contribution_cyto_df = contribution_var_df.groupby('cyto').sum()
    
    return contribution_cyto_df.to_numpy(), np.array(contribution_cyto_df.index)

def compute_gene_contrib_to_cyto(
    label_var, contribution_var, var_gene_file
):
    print(len(label_var))
    print(contribution_var.shape)
    #variant_df = pd.read_csv(variant_tsv, sep='\t', compression='gzip')
    #var2gene_dict = dict(zip(variant_df['label_var'], variant_df['label_gene'])) 
    var2gene_dict = dict()
    var2cyto_dict = dict()
    gene2cyto_dict = dict()
    with open(var_gene_file, 'r') as f:
        for l in f.read().split('\n'): 
            try:
                var2gene_dict[l.split('\t')[0]] = l.split('\t')[1]
                var2cyto_dict[l.split('\t')[0]] = l.split('\t')[2] 
                gene2cyto_dict[l.split('\t')[1]] = l.split('\t')[2]   
            except:
                pass
                #print(l)
    # with open('./v2gdict.txt', 'w+') as f:
    #     f.write(str(var2gene_dict))
    contribution_var_df = pd.DataFrame(contribution_var)    
    #contribution_var_df['cyto'] = [var2cyto_dict[x] for x in label_var]
    contribution_var_df['gene'] = [var2gene_dict[x] for x in label_var]

    print(contribution_var_df)
    contribution_cyto_df = contribution_var_df.groupby('gene').sum()
    
    return contribution_cyto_df.to_numpy(), np.array(contribution_cyto_df.index), gene2cyto_dict