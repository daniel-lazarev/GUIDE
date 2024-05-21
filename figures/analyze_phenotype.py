from __future__ import print_function

_README_ = '''
-------------------------------------------------------------------------
Post_process_step1.py

Given a npz file, generate plots and GREAT query bed file

npz_file, out_dir_root, phe_list

Usage: python post_process_step1.py -d ../private_data/results/dev_PTVsNonMHC_z_nonCenter_p0001_100PCs.npz -p ../public_data/phenotype_of_interest.lst

Author: Yosuke Tanigawa (ytanigaw@stanford.edu)
Date: 2018/01/22 (update on 2018/5/18)
-------------------------------------------------------------------------
'''

import os, logging, collections, itertools
import numpy as np
import pandas as pd
import argparse
import matplotlib
from misc import plot_scatter
from logging.config import dictConfig
from logging import getLogger

dictConfig(dict(
    version = 1,
    formatters = {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    handlers = {
        'h': {'class': 'logging.StreamHandler','formatter': 'f',
              'level': logging.DEBUG}},
    root = {'handlers': ['h'], 'level': logging.DEBUG,},
))

matplotlib.rc('font',**{'size':16, 'family':'sans-serif','sans-serif':['HelveticaNeue', 'Helvetica']})

logger = getLogger('post_process')
from decomposition import decomposition

def analyze_phenotype(d, out_dir, phe, num_top_components=5, is_guide=None):
    if is_guide is None:
        print('Last argument, "is_guide" must be either True or False. If input is GUIDE result, it is true. Otherwise, False.')
        return
    if is_guide:
        algo_str = 'guide'
    else:
        algo_str = 'degas'
    def get_safe_phe_name(phe):
        return ''.join([c if (c.isalnum() or c in ['_', '.'] ) else '_' for c in phe])    
    def cos2score_tbl(d, phe):
        pcs, score = d.get_topk_pcs_for_phe_with_scores_by_label(phe)
        return pd.DataFrame(collections.OrderedDict([
            ('#Rank', np.arange(d.d['n_PCs']) + 1),
            ('PC_(zero_based)', pcs),
            ('squared_cosine_score', score)
        ]))
    def cont_plot_filename(phe_dir, rank, pc, target):
        return os.path.join(
            phe_dir, 'contribution', 
            '{:02d}_PC{:02d}_{}'.format(rank, pc, target)
        )    
    
    logger.info(phe)
    phe_dir=os.path.join(out_dir, 'phenotypes', algo_str, get_safe_phe_name(phe))
    if(not os.path.exists(os.path.join(phe_dir, 'contribution'))):
        os.makedirs(os.path.join(phe_dir, 'contribution'))
        
    cos2score_tbl=cos2score_tbl(d, phe)
    cos2score_tbl.to_csv(
        os.path.join(phe_dir, 'squared_cosine_scores.tsv'),
        index=False, sep='\t'
    
    )

    plot_scatter(
        d.plot_data_cos_phe_by_label(phe),
        save=os.path.join(phe_dir, 'squared_cosine_scores.pdf')
    )
    
    for rank, pc in enumerate(d.get_topk_pcs_for_phe_by_label(phe, topk=num_top_components)):
        try:
            decomposition.plot_contribution_legend_gene(
                d, pc, topk=20, save_exts=['pdf'],
                save=cont_plot_filename(phe_dir, rank, pc, 'gene'),                
            )
        except Exception as e:  
            logger.warning('Gene contribution score plot failed! Rank = {}, PC = {}'.format(rank, pc))
            logger.warning(e)
        try:            
            decomposition.plot_contribution_legend_phe(
                d, pc, topk=20, save_exts=['pdf'],
                save=cont_plot_filename(phe_dir, rank, pc, 'phe')
            )
        except Exception as e:  
            logger.warning('Phenotype contribution score plot failed! Rank = {}, PC = {}'.format(rank, pc))
            logger.warning(e)
        matplotlib.pyplot.close('all')

