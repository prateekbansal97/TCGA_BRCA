import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from utils.preprocessing import load_rnaseq, load_clinical_phenotype, generate_feature_and_labels
from models.mlp import MLP
from sklearn.model_selection import train_test_split

#Input data files
rna_data_path = './data/RNAseq_HiSeqV2.tsv'
clinical_data_path = './data/Clinical_Phenotype.tsv'

rna_data = load_rnaseq(rna_data_path)
clinical_data = load_clinical_phenotype(clinical_data_path)

features, labels = generate_feature_and_labels(rna_data, clinical_data)
