# This script arranges sampled data into testing and training data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#Load the stats dataframe in order to apply appropriate filters based on sparsity etc
df_stats = pd.read_csv('data/assay_stats.csv')

# Load toxcast data
df_toxcast = pd.read_csv('data/Toxcast_aggregated.csv')

#Filter assays based on onumber of active compounds, and data sparsity
df_filtered_assays = df_stats.loc[(df_stats['active_compounds'] >= 60) & (df_stats['missing_compounds'] <= 7287)]

#100 randomly sampled assays from the filtered 
df_sampled = df_filtered_assays.sample(100, random_state = 0)

#Pull assays based on assays from sampled dataframe
df_sampled_data = df_toxcast[df_sampled['assay_name']]

#Compound based split
train, test = train_test_split(df_sampled_data, test_size=0.2)

#This essentially initialises the dataframe for testing and training data (ITHINK)
train, test = train_test_split(df_sampled_data['TOX21_p53_BLA_p1_ch2'].dropna(), test_size=0.2)

train_df = pd.DataFrame(train)
test_df = pd.DataFrame(test)

#List of headers

headers = list(df_sampled_data.columns)

#Remove first entry since this already exists 
del headers[0]

#Iterate through assays updating the datarames
for i in headers:
    traini, testi = train_test_split(df_sampled_data[i].dropna(), test_size=0.2)
    
    traini_df = pd.DataFrame(traini)
    testi_df = pd.DataFrame(testi)
    
    train_df = traini_df.join(train_df, on = None, how = 'outer')
    
    test_df = testi_df.join(test_df, on = None, how = 'outer')


#This is needed to make the dataframe full sized

df_smiles = df_toxcast['standardised_smiles']
df_smiles = pd.DataFrame(df_smiles)

#This joins the smiles data, and also makes them both correspond to each other 

df_train = df_smiles.join(train_df, on = None, how = 'outer')
df_test = df_smiles.join(test_df, on = None, how = 'outer')

#Save them to csv
df_train.to_csv('data/df_train.csv')
df_test.to_csv('data/df_test.csv')