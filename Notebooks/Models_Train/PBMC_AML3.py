
import numpy as np
from numpy import intersect1d, setdiff1d, quantile, unique, asarray, zeros
from matplotlib import pyplot
import matplotlib.pyplot as plt
import os
from copy import deepcopy


from time import time

from math import ceil
from scipy.stats import spearmanr, gamma, poisson
import scipy.sparse as sp
from scipy.io import mmread
from scipy.stats import pearsonr

from anndata import AnnData, read_h5ad
import anndata as ad
import scanpy as sc
from scanpy import read
import seaborn as sns
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset
from torch import tensor
from torch.cuda import is_available


import importlib

import sys
import umap
from fuzzywuzzy import process

from anndata import AnnData, read_h5ad
import anndata as ad
import scanpy as sc
from scanpy import read
import seaborn as sns
import pandas as pd
import importlib

import numpy as np
from numpy import intersect1d, setdiff1d, quantile, unique, asarray, zeros
from matplotlib import pyplot
import matplotlib.pyplot as plt

import sys
import umap
from fuzzywuzzy import process



# Add the src directory to the Python path
sys.path.append(os.path.abspath('/home/nandivada.s/genoTopheno/src'))

# Import the sciPENN_API module


import sciPENN.sciPENN_API
import sciPENN.Preprocessing

# Reload the module to ensure the latest changes are reflected
# importlib.reload(SCIPENN.sciPENN_API)

from sciPENN.sciPENN_API import sciPENN_API
from sciPENN.Preprocessing import preprocess
from sciPENN.Utils import build_dir
from sciPENN.Preprocessing import preprocess
from sciPENN.Data_Infrastructure.DataLoader_Constructor import build_dataloaders
from sciPENN.Network.Model import sciPENN_Model
from sciPENN.Network.Losses import cross_entropy, mse_quantile, no_loss




""" PBMC Read in Raw Data"""
adata_gene = sc.read("/home/nandivada.s/R/x86_64-pc-linux-gnu-library/pbmc_gene.h5ad")
adata_protein = sc.read("/home/nandivada.s/R/x86_64-pc-linux-gnu-library/pbmc_protein.h5ad")


pbmc_aml3 = sc.read("/home/nandivada.s/genoTopheno/Datasets/GSM7510833_AML3_expression_counts.csv.gz").T
adata_gene_test = pbmc_aml3
# Define the required parameters
gene_trainsets =  [adata_gene] # Your gene training datasets
protein_trainsets = [adata_protein]  # Your protein training datasets
gene_test = adata_gene_test  # Optional gene test datasets
gene_list = []  # Your list of genes
select_hvg = True
train_batchkeys = ['donor']
test_batchkey = None
type_key = 'celltype.l2'
cell_normalize = True
log_normalize = True
gene_normalize = True
min_cells = 3
min_genes = 20
batch_size = 128
val_split = 0.1
use_gpu = True

# Create an instance of the sciPENN_API class
sciPENN_instance = sciPENN_API(
    gene_trainsets =  [adata_gene], # Your gene training datasets
    protein_trainsets = [adata_protein],  # Your protein training datasets
    gene_test = adata_gene_test, # Optional gene test datasets
    gene_list = [],  # Your list of genes
    select_hvg = True,
    train_batchkeys = ['donor'],
    test_batchkey = None,
    type_key= 'celltype.l2', 
    cell_normalize = True,
    log_normalize = True,
    gene_normalize = True,
    min_cells = 3,
    min_genes = 20,
    batch_size = 128,
    val_split = 0.1,
    use_gpu = True
)

sciPENN_instance.train(quantiles = [0.1, 0.25, 0.75, 0.9], n_epochs = 10000, ES_max = 12, decay_max = 6, 
             decay_step = 0.1, lr = 10**(-3), weights_dir = "pbmc_to_AML3", load = True)

imputed_AML2_PBMC_protein = sciPENN_instance.predict()
imputed_AML2_PBMC_protein.write("imputed_PBMC_Protein_AML3_Patient.h5ad")