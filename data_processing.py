import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Import aggregated & normalised data
df_toxcast = pd.read_csv('data/Toxcast_aggregated.csv')

# Also read other datasets
df_ames = pd.read_csv('data/Ames_aggregated.csv')
df_tox21 = pd.read_csv('data/Tox21_aggregated.csv')



# Data Exploration



def stats(df,lines=1):
    stats = np.zeros((df.shape[1]-lines,4))
    for i in range(df.shape[1]-lines): #iterate over columns
        #print(i) # Just to see the progress of the loop

        for j in range(df.shape[0]): #iterate over rows

            if np.isnan(df.iloc[j,i]): #update numpy array
                    stats[i,3] += 1

            elif df.iloc[j,i] == 0.0:
                stats[i,2] += 1

            elif df.iloc[j,i] == 1.0:
                stats[i,1] += 1
            else:
                print(i,j) #Locate any errors if they arise

    df_stats = pd.DataFrame(stats)

    for i in range(df.shape[1]-lines):
        df_stats.iloc[i,0] = df.columns[i]


    df_stats.columns = ['assay_name','active_compounds','inactive_compounds','missing_compounds']

    total = df_stats['active_compounds']+df_stats['inactive_compounds']+df_stats['missing_compounds']
    df_stats['total'] = total

    fraction_active = df_stats['active_compounds']/df_stats['total']
    df_stats['fraction_active'] = fraction_active

    fraction_inactive = df_stats['inactive_compounds']/df_stats['total']
    df_stats['fraction_inactive'] = fraction_inactive

    fraction_missing = df_stats['missing_compounds']/df_stats['total']
    df_stats['fraction_missing'] = fraction_missing
    
    return(df_stats)

df_stats = stats(df_toxcast)
#df_stats.to_csv('data/assay_stats.csv')
print(df_stats)


# Also processing ames and tox21

df_ames_stats = stats(df_ames,2)
df_tox21_stats = stats(df_tox21,2)

df_ames_stats.to_csv('data/ames_assay_stats.csv')
df_tox21_stats.to_csv('data/tox21_assay_stats.csv')

#Plotting function to create plot of fractions of active and missing data in the assays
def plot_stats(df, name):
    plt.clf()
    plt.scatter(df['fraction_active'], df['fraction_missing'], marker = 'x', s=20, label = name)
    plt.title('Fractions of missing and active labels in '+ name +' Data')
    plt.xlabel('Fraction of Assay Active')
    plt.ylabel('Fraction of Assay Missing Label')
    plt.legend()
    plt.savefig('plots/'+name+'_fractions_missing_active.pdf')
    return()

plot_stats(df_ames_stats, 'Ames')
plot_stats(df_tox21_stats, 'Tox21')
# Creating the plots

