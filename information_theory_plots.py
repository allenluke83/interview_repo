# This sctipt plots information theory related graphs

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

df_relations = pd.read_csv('data/assay_information.csv')

#Plotting a histogram of mutual information in toxcast data
plt.hist(df_relations['mutual_information'], bins=50, label = 'Mutual Information', edgecolor = '#1f77b4')
plt.title('Mutual Information for Assay Pairs')
plt.xlabel('Mutual Information')
plt.ylabel('Count')
plt.savefig('plots/mutual_info.pdf')
plt.plot()

# Plotting histogram of joint entropy in the toxcast data
plt.hist(df_relations['joint_entropy'], bins=50, edgecolor = '#1f77b4')
plt.title('Joint Entropy of Assay Pairs')
plt.xlabel('Joint Entropy')
plt.ylabel('Count')
plt.savefig('plots/joint_entropy.pdf')
plt.plot()

# Ratio of mutual information and joint entropy
plt.hist(df_relations['mutual_information']/df_relations['joint_entropy'], bins=50, label = 'Mutual Information / Joint Entropy ratio',edgecolor = '#1f77b4')
plt.xlabel('Ratio')
plt.title('Ratio of mutual information to joint entropy for each assay combination')
plt.ylabel('Count')
plt.legend()
plt.savefig('plots/ratio_graph.pdf')

#Plotting the above with adjusted y axis limits such that higher values can be distinguished
plt.hist(df_relations['mutual_information']/df_relations['joint_entropy'], bins=50, label = 'Mutual Information / Joint Entropy ratio', edgecolor = '#1f77b4')
plt.xlabel('Ratio')
plt.title('Ratio of mutual information to joint entropy for each assay combination')
plt.ylabel('Count')
plt.ylim(0,500)
plt.legend()
plt.savefig('plots/ratio_graph_ylim.pdf')


#Ames and toxcast data

#read_data
tox21_relations = pd.read_csv('data/tox21_assay_information.csv')
ames_relations = pd.read_csv('data/ames_assay_information.csv')


# Ratio of mutual information and joint entropy
plt.hist(ames_relations['mutual_information']/ames_relations['joint_entropy'], bins=50, label = 'Ames Data')
plt.xlabel('Ratio')
plt.title('Ratio of mutual information to joint entropy for each assay combination')
plt.ylabel('Count')
plt.legend()
plt.savefig('plots/ames_ratio_graph.pdf')
plt.plot()

# Ratio of mutual information and joint entropy
plt.hist(tox21_relations['mutual_information']/tox21_relations['joint_entropy'], bins=50, label = 'Tox21 Data')
plt.xlabel('Ratio')
plt.title('Ratio of mutual information to joint entropy for each assay combination')
plt.ylabel('Count')
plt.legend()
plt.savefig('plots/tox21_ratio_graph.pdf')
plt.plot()





# Putting them all together 
# Import more data required for the plot
df_stats = pd.read_csv('data/assay_stats.csv')
df_ames_stats = pd.read_csv('data/ames_assay_stats.csv')
df_tox21_stats = pd.read_csv('data/tox21_assay_stats.csv')


fig, axs = plt.subplots(3, 2, figsize=(15,15))
axs[0,0].hist(df_relations['mutual_information']/df_relations['joint_entropy'], bins=50, label = 'Toxcast', edgecolor = '#1f77b4')
axs[0,0].set_ylim(0,1000)
axs[0,0].set_title('Mutual Info - Joint Entropy ratio')
axs[0,0].set_ylabel('Count')
axs[0,0].set_xlabel('Ratio')
axs[0,0].legend()

axs[2,0].hist(ames_relations['mutual_information']/ames_relations['joint_entropy'], bins=50, label = 'Ames Data', edgecolor = '#1f77b4')
axs[2,0].set_title('Mutual Info - Joint Entropy ratio')
axs[2,0].set_ylabel('Count')
axs[2,0].set_xlabel('Ratio')
axs[2,0].legend()

axs[1,0].hist(tox21_relations['mutual_information']/tox21_relations['joint_entropy'], bins=50, label = 'Tox21 Data', edgecolor = '#1f77b4')
axs[1,0].set_title('Mutual Info - Joint Entropy ratio')
axs[1,0].set_ylabel('Count')
axs[1,0].set_xlabel('Ratio')
axs[1,0].legend()

axs[0,1].scatter(df_stats['fraction_active'], df_stats['fraction_missing'], marker = 'x', s=0.8, label = 'Toxcast')
axs[0,1].set_title('Fraction of Missing and Active Data Points')
axs[0,1].set_ylabel('Fraction Missing')
axs[0,1].set_xlabel('Fraction Active')
axs[0,1].legend()

axs[1,1].scatter(df_tox21_stats['fraction_active'], df_tox21_stats['fraction_missing'], marker = 'x', s=20, label = 'Tox21 Data')
axs[1,1].set_title('Fraction of Missing and Active Data Points')
axs[1,1].set_ylabel('Fraction Missing')
axs[1,1].set_xlabel('Fraction Active')
axs[1,1].legend()

axs[2,1].scatter(df_ames_stats['fraction_active'], df_ames_stats['fraction_missing'], marker = 'x', s=20, label = 'Ames Data')
axs[2,1].set_title('Fraction of Missing and Active Data Points')
axs[2,1].set_ylabel('Fraction Missing')
axs[2,1].set_xlabel('Fraction Active')
axs[2,1].legend()

plt.savefig('plots/dataset_comparison.pdf')


