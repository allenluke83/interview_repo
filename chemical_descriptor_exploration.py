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
from itertools import combinations

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



# Taking concepts from information theory to produce comparisons

# This function takes two assays and calculates a contingency table and then uses this to calculate 
# the joint entropy and the mutual information.

def metric_of_two(assay1,assay2):
    #get contigengency table that describes actives in assay1 and assay2
    #considers only substances measured in both assays
    #return 2x2 contingengy table: (0,0)/a: A+B+, (0,1)/b: A+B-, (1,0)/c: A-B+, (1,1)/d: A-B-
    
    a = 0
    b = 0
    c = 0
    d = 0
    
    for ass1,ass2 in zip(assay1,assay2):
        
        if np.isnan(ass1) or np.isnan(ass2): #consider only cases where compound was measured in both assyas
            continue
        
        if ass1 == 1 and ass2 == 1:
            a+=1
        
        elif ass1 == 1 and ass2 == 0:
            b+=1
        
        elif ass1 == 0 and ass2 == 1:
            c+=1
        
        elif ass1 == 0 and ass2 == 0:
            d+=1
    
    #prob active in assay1
    p1 = (a+b)/(a+b+c+d)
    #entropy assay1
    h1 = -p1*np.log2(p1) - (1-p1)*np.log2(1-p1)
    
    #prob active in assay2
    p2 = (a+c)/(a+b+c+d)
    #entropy assay2
    h2 = -p2*np.log2(p2) - (1-p2)*np.log2(1-p2)
    
    #prob each table field
    p3 = a/(a+b+c+d)
    p4 = b/(a+b+c+d)
    p5 = c/(a+b+c+d)
    p6 = d/(a+b+c+d)
    
    #joint entropy
    h12 = -p3*np.log2(p3) - p4*np.log2(p4) - p5*np.log2(p5) - p6*np.log2(p6)
    
    #mutual information
    mi = h1 + h2 - h12
    
    return(h12,mi)




def info_table(df, extra = False):



    header_list = list(df.columns)
    del header_list[-1] #delete the smiles heading
    if extra == True:
        del header_list[-1]
    else: pass
    header_combinations = []

    for combo in combinations(header_list, 2):  # 2 for pairs, 3 for triplets, etc
        header_combinations.append(combo)

    header_combinations = np.array(header_combinations)



    information_table = np.zeros([len(header_combinations), 2])

    for i in range(len(header_combinations)): #iterate over combinations

        try:
            information_table[i,0], information_table[i,1] = metric_of_two(df[header_combinations[i,0]],df[header_combinations[i,1]])
        except ZeroDivisionError:
            information_table[i,0], information_table[i,1] = np.nan, np.nan
            print('error')





    df_relations = pd.DataFrame({'assay1' : header_combinations[:,0],
                               'assay2': header_combinations[:,1],
                               'joint_entropy' : information_table[:,0],
                               'mutual_information' : information_table[:,1]})

    return(df_relations)



# Creating assay_info table for toxcast dataframe
df_relations = info_table(df_toxcast)

# Doing the same ofor the toher datasets

#Read in the data
df_ames = pd.read_csv('data/Ames_aggregated.csv')
df_tox21 = pd.read_csv('data/Tox21_aggregated.csv')

#Create the relationship tables
ames_relations = info_table(df_ames, True)
tox21_relations = info_table(df_tox21, True)


#Save for later use
df_relations.to_csv('data/assay_information.csv')
ames_relations.to_csv('data/ames_assay_information.csv')
tox21_relations.to_csv('data/tox21_assay_information.csv')