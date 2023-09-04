#Import modules
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

#Import Data
df_ToxCast = pd.read_csv('data/ToxCast_aggregated.csv') 
df_train = pd.read_csv('data/df_train.csv')
df_test = pd.read_csv('data/df_test.csv')

#Delete the indexing columns (a result of importing the data)
del df_train['Unnamed: 0'] 
del df_test['Unnamed: 0']

#Creating dictionary with locations of values etc. This is handy for step 2 when position of things becomes much more importan t
headers = df_train.columns.tolist()
del headers[0]
dictionary = dict()

#get array of all fingerprints
mols = [Chem.MolFromSmiles(smi) for smi in df_train['standardised_smiles']]
fps = [AllChem.GetMorganFingerprintAsBitVect(mol,radius=2,nBits=2048) for mol in mols]
fingerprint = np.empty([df_train.shape[0],2048]) 

for i,fp in enumerate(fps):
    for j,bit in enumerate(fp): 
        fingerprint[i,j] = bit

#Looping through the headers
for j in headers: 
    count = -1
    indexes_train = []
    #For train df
    for i in df_train[j]:
        count += 1
        if i == np.nan: #ignore if empty
            continue
        if i == 0 or i == 1:
            indexes_train.append(count) #If active/inactive note position
    count = -1

    #Do same for test df
    indexes_test = [] 

    for i in df_test[j]:
        count += 1
        if i == np.nan:
            continue

        if i == 0 or i == 1:
            indexes_test.append(count)

    dictionary[j] = dict(test = indexes_test, train = indexes_train)

#Step ONE. PREDICT FOR ALL MISSING LABELS

# This essentially uses same as standard QSAR and returns filled matrix #Create headers list
headers = df_train.columns.tolist() 
del headers[0]
count = -1

#Table that will be populated with predictions
table = np.zeros((7787,100)) 
for j in headers:
    count += 1
    indexes = df_train[j].dropna().index.values.tolist()
    indexes
    X_train = []
    for i in indexes: 
        X_train.append(fingerprint[i])

    X_train = np.array(X_train)
    y_train = df_train[j].dropna() 
    y_train = np.array(y_train)

    #data to be predicted
    indexes = df_train[j].drop(dictionary[j]['train']).index.values.tolist()
    X_test= []
    for i in indexes: 
        X_test.append(fingerprint[i])
    
    X_test = np.array(X_test)

    #creating the model
    clf = RandomForestClassifier(random_state=58)
    clf.fit(X_train,y_train)

    #predicting the data
    y_pred = clf.predict(X_test)

    #bringing all together in the correct order
    train_zip = zip(df_train[j].dropna().index.values.tolist(),y_train) 
    train_zip=list(train_zip)
    pred_zip = zip(df_train[j].drop(dictionary[j]['train']).index.values.tolist(),y_pred) 
    pred_zip=list(pred_zip)

    #combine predictions to original test data
    pred_zip.extend(train_zip)

    #convert to array
    pred_zip=np.array(pred_zip)

    #sort by indexes (first column)
    pred_zip_sorted = pred_zip[pred_zip[:, 0].argsort()] 
    table[:,count] = pred_zip_sorted[:,1]
    print(count) #just to track progress in loop



#STEP 2.
#This creates a big array with concatenated chemical features and assays in
step2_array = np.concatenate((fingerprint,table), axis=1)
values = []
roc = [] 
comparison_table = []
count = 2048 #This allows one of the loops to begin on the assay data - skipping over the chemical descriptor data 
for i in headers:
    #This gets all the rows where data is found in the test molecule
    train = step2_array[dictionary[i]['train'],:]

    #Now removing the test assay, leaving just the training molecules
    X_train = np.delete(train, count, 1) 
    y_train = train[:,count]

    #Same as above with test data
    test = step2_array[dictionary[i]['test'],:] 
    X_test = np.delete(test, count, 1)
    #y_test
    y_test = df_test[i].dropna()
    y_test = np.array(y_test)

    clf = RandomForestClassifier(random_state=58) 
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    values.append(matthews_corrcoef(y_test,y_pred))
    count +=1
    #print(count - 2048) #purely for progress throught the loops

    roc.append(roc_auc_score(y_test,y_pred_proba[:,1]))

    comparison_table.append([i, matthews_corrcoef(y_test,y_pred)])


assays = []
for i in range(len(values)):
    assays.append(i)


#Saving data to make a compariosn with other models later
comparison_table = np.array(comparison_table)
np.savetxt('data/FN_comp.csv', comparison_table, delimiter = ',', fmt = '%s')

#Plotting roc
plt.plot(assays, roc, label = 'Feature Net') 
plt.title('ROC-AUC for Assays')
plt.xlabel('Assays') 
plt.ylabel('ROC-AUC Values') 
plt.grid(axis = 'y') 
plt.ylim(0.3, 1) 
plt.xlim(-1,100) 
plt.legend()
np.median(roc) 
plt.savefig('data/ROC_FN.pdf')

#Single MCC graph for FN

values.sort(reverse = True) 
roc.sort(reverse = True)
plt.plot(assays, values, label = 'Feature Net') 
plt.title('MCC for Assays')
plt.xlabel('Assays')
plt.ylabel('Matthews correlation coefficient (MCC)')
plt.grid(axis = 'y')
plt.ylim(-0.05, 1)
plt.xlim(-1,100)
plt.legend() 
plt.savefig('data/MCC_FN.pdf')

