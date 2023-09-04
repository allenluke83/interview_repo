#Import Modules
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


#Importing data prepared earlier
df_train = pd.read_csv('data/df_train.csv') 
df_test = pd.read_csv('data/df_test.csv')

#Delete the indexing columns (a result of importing the data)
del df_train['Unnamed: 0'] 
del df_test['Unnamed: 0']

#Create headers list
headers = df_train.columns.tolist()
del headers[0]

#get array of all fingerprints

mols = [Chem.MolFromSmiles(smi) for smi in df_train['standardised_smiles']]
fps = [AllChem.GetMorganFingerprintAsBitVect(mol,radius=2,nBits=2048) for mol in mols]

fingerprint = np.empty([df_train.shape[0],2048])
for i,fp in enumerate(fps):
    for j,bit in enumerate(fp):
        fingerprint[i,j] = bit
    
fingerprint.shape

indexes = df_train['OT_AR_ARSRC1_0960'].dropna().index.values.tolist()

indexes

X_train = []

for i in indexes:
    X_train.append(fingerprint[i])
    
X_train = np.array(X_train)

X_train
#This then loops through all the assays using the above code

values = []
roc = []
comparison_table = []

for i in headers:
    

    #X_Train
    indexes = df_train[i].dropna().index.values.tolist()
    X_train = []


    for j in indexes:
        X_train.append(fingerprint[j])
    X_train = np.array(X_train)
    

    #X_test
    indexes = df_test[i].dropna().index.values.tolist()
    X_test = []


    for j in indexes:
        X_test.append(fingerprint[j])
    X_test = np.array(X_test)
    
    #y_train
    y_train = df_train[i].dropna()
    y_train = np.array(y_train)


    #y_test
    y_test = df_test[i].dropna()
    y_test = np.array(y_test)
    
    #Build Model
    clf = RandomForestClassifier(random_state=58)
    clf.fit(X_train,y_train)


    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    print(matthews_corrcoef(y_test,y_pred))
    values.append(matthews_corrcoef(y_test,y_pred))
    roc.append(roc_auc_score(y_test,y_pred_proba[:,1]))
    comparison_table.append([i, matthews_corrcoef(y_test,y_pred)])

#Saving data
comparison_table = np.array(comparison_table)
np.savetxt('data/ST_comp.csv', comparison_table, delimiter = ',', fmt = '%s')

#Ordering mcc values and creating list of numbers for the graphs
assays = []
for i in range(len(values)): 
    assays.append(i)

print(assays) 
values.sort(reverse = True)
roc.sort(reverse = True)

plt.clf()
plt.plot(assays, values, label = 'Single Task Model')
plt.title('MCC by Assay')
plt.xlabel('Assays')
plt.ylabel('Matthews correlation coefficient (MCC)')
plt.grid(axis = 'y')
plt.ylim(-0.05, 0.8)
plt.xlim(-1,100)
plt.legend()
plt.savefig('plots/MCC_ST.pdf')

plt.clf()
plt.plot(assays, roc, label = 'Single Task Model')
plt.title('ROC-AUC for Assays')
plt.xlabel('Assays')
plt.ylabel('ROC-AUC Values')
plt.grid(axis = 'y')
plt.ylim(0.3, 1)
plt.legend()
plt.xlim(-1,100)
plt.savefig('plots/ROC_ST.pdf')


# Addition of GHOST sorting method

#This is the same code as above with the introduction of GHOST optimisation
values_GHOST = []
roc_GHOST = [] 
comparison_table_GHOST = []


#GHOST Stuff ******************************************
from GHOST import ghostml
def probs_to_binary_with_threshold(probs,threshold):
    y_pred = np.array([1 if i >threshold else 0 for i in probs])
    return(y_pred)
#*******************************************************


thresholds = np.round(np.arange(0.05,0.55,0.05),2)

for i in headers:
    #X_Train
    indexes = df_train[i].dropna().index.values.tolist() 
    X_train = []
    for j in indexes: X_train.append(fingerprint[j])
    X_train = np.array(X_train)
    #X_test
    indexes = df_test[i].dropna().index.values.tolist() 
    X_test = []
    for j in indexes: X_test.append(fingerprint[j])
    X_test = np.array(X_test)
    #y_train
    y_train = df_train[i].dropna() 
    y_train = np.array(y_train)
    #y_test
    y_test = df_test[i].dropna() 
    y_test = np.array(y_test)
    #Build Model
    clf = RandomForestClassifier(random_state=58) 
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test) 
    y_pred_proba = clf.predict_proba(X_test)
    #GHOST stuff
    y_test_probs = clf.predict_proba(X_test)[:,1]
    y_train_probs = clf.predict_proba(X_train)[:,1]
    threshold_opt = ghostml.optimise_threshold_from_predictions(y_train, y_train_probs, thresholds, ThOpt_metrics = 'ROC') 
    y_pred_opt = probs_to_binary_with_threshold(y_test_probs,threshold_opt)
    print(matthews_corrcoef(y_test,y_pred_opt))
    values_GHOST.append(matthews_corrcoef(y_test,y_pred_opt))
    roc_GHOST.append(roc_auc_score(y_test,y_pred_proba[:,1]))
    comparison_table_GHOST.append([i, matthews_corrcoef(y_test,y_pred_opt)])



#Again, sorting the MCC values based on size
assays = []
for i in range(len(values_GHOST)): 
    assays.append(i)
values_GHOST.sort(reverse = True) 
#roc.sort(reverse = True)


#Plotting a comparison
plt.plot(assays, values_GHOST, label = 'Single Task Model with GHOST') 
plt.plot(assays, values, label = 'Single Task Model')
plt.title('MCC by Assay')
plt.xlabel('Assays')
plt.ylabel('Matthews correlation coefficient (MCC), \n optimised using GHOST') 
plt.grid(axis = 'y')
plt.ylim(-0.05, 0.8)
plt.xlim(-1,100)
plt.legend() 
plt.savefig('data/comp_ST.pdf')


#PLotting ROC-AUC values
plt.plot(assays, roc, label = 'Single Task Model') 
plt.title('ROC-AUC for Assays') 
plt.xlabel('Assays')
plt.ylabel('ROC-AUC Values')
plt.grid(axis = 'y')
plt.ylim(0.3, 1)
plt.legend()
plt.xlim(-1,100) 
plt.savefig('data/ROC_ST.pdf')