# This script is exploring the properties of the chemical descriptors within assay datasets


import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from rdkit import DataStructs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import plotly.graph_objects as go
import itertools

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

from itertools import combinations
import scipy.stats


#Load the stats dataset and the toxcast data
df_stats = pd.read_csv('data/assay_stats.csv')
df_toxcast = pd.read_csv('data/Toxcast_aggregated.csv')

#Create a dataframe contianing the smiles and chemical structures
df_chemical_descriptor = pd.DataFrame(df_toxcast.iloc[:,df_toxcast.shape[1]-1])

#Using RDkit to obtain mols and molecular weights for the molecules
mols = [Chem.MolFromSmiles(smi) for smi in df_chemical_descriptor['standardised_smiles']]
weights = [Descriptors.ExactMolWt(mol) for mol in mols]

#Using pandas to create a df with smiles and respective molecular weight
df_chemical_descriptor['weights']=pd.DataFrame(weights)

#Saving data
df_chemical_descriptor.to_csv('data/chemical_descriptor.csv')


# Creating plot of the distribution of molecular weights
print(df_chemical_descriptor['weights'].mean())
print(df_chemical_descriptor['weights'].median())
plt.hist(df_chemical_descriptor['weights'], bins=30, label = 'Compounds', edgecolor = '#1f77b4')
plt.title('Molecular Weights of Compounds in Toxcast')
plt.xlabel('Molecular Weight')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('plots/weights.pdf')


