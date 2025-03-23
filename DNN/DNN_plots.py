# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:30:15 2025

Plotting DNN test results

@author: rosac
"""


import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve, confusion_matrix, class_likelihood_ratios
from process import X_test_tensor, y_test_tensor, name, y, cols_dnn, Lumi_test
import seaborn as sns
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)




#plotting model output 
print(name)
path = f'../DNN/Results/result_{name}.h5'
f = h5py.File(path, 'r')

#data
preds =  pd.DataFrame(f['preds'])
score =  np.array(f['score'])
score = score.squeeze()

#data truth labels - y_test_tensor
labels = np.array(y_test_tensor) 

#colour scheme 
layers = [1, 1, 1, 1, 1] 
cmap = plt.get_cmap('viridis')
colors = []
for i, layer_size in enumerate(layers):
    colors.extend([cmap(i / (len(layers) - 1))] * layer_size)
mainc = colors[1]
secondc = colors[3]
thirdc = colors[2]


#ROC
#plotting ROC curve 
y_preds = score[:,1]
 #probabilities for Z_L - roc_curve function uses 1 class probabilities 
fpr, tpr, thresholds = roc_curve(labels, y_preds) 
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8,6),dpi=1000)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', color=mainc)
plt.plot(np.linspace(0,1), np.linspace(0,1), linestyle='--', color='lightgrey' )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.ylabel('True Positive Rate', loc='center',fontsize=17)
plt.xlabel('False Positive Rate', loc='center', fontsize=17)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
#plt.title(f'ROC Curve (Zt) ({name})', pad=30)
plt.legend(fontsize=14)
plt.savefig(f'../DNN/Models/{name}/roc_{name}.png', dpi=1000)
plt.show()



mask0 = labels == 0
mask1 = labels == 1

#Discriminant 
#Calculating discriminants 
log_ratio = []
for i in range(len(score)):
    log_ratio.append(np.log(score[i,0] / score[i,1])) #no weights 
log_ratio = np.array(log_ratio)

#plotting log ratio with 2 distributions overlayed 
labels = np.array(labels)

#matching discriminant scores to classifications
dis0 = log_ratio[mask0]
dis1 = log_ratio[mask1]

#matching luminosity weightings to classifications, accounting 
#fact that only 40% of dataset is used in testing
Lumi = np.array(Lumi_test.values)
weights0 = Lumi[mask0] * 2.5
weights1 = Lumi[mask1] * 2.5

#plotting discriminant distribution
fig, ax = plt.subplots(figsize=(8,6), dpi=600)

plt.hist(dis0, density=False, histtype='step', bins=50, weights = weights0, color=mainc, label='$\mathrm{Z}_L$', linewidth=2) #alpha=0.7
plt.hist(dis1, density=False, histtype='step', bins=50, weights = weights1, color=secondc, label='$\mathrm{Z}_T$', linewidth=2)

ax.set_xlabel('$D_{Z_L}$', loc='center', fontsize=17)
ax.set_ylabel('Event Fraction', loc='center', fontsize=17)

plt.minorticks_on()
ax.xaxis.set_tick_params(top=True, which='both', labeltop=False, labelsize=14)
ax.yaxis.set_tick_params(right=True, which='both', labelright=False, labelsize=14)

plt.legend(loc='upper left', fontsize=14)
plt.savefig(f'../DNN/Models/{name}/discriminant{name}.png', dpi=600)



#matching network scores to classifications
score0 = score[mask0, 0]
score1 = score[mask1, 0]


#plot network score, weighted by luminosity
fig, ax = plt.subplots(figsize=(8,6), dpi=600)
counts_L, edges_L, plot_L = plt.hist(score0, density=False, histtype='step', bins=20, weights = weights0, color=mainc, label='$\mathrm{Z}_L$', linewidth=2) #alpha=0.7
counts_T, edges_T, plot_T = plt.hist(score1, density=False, histtype='step', bins=20, weights = weights1, color=secondc, label='$\mathrm{Z}_T$', linewidth=2)

ax.set_xlabel('Network score', loc='center', fontsize=17)
ax.set_ylabel('Event Fraction', loc='center', fontsize=17)

plt.minorticks_on()
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_tick_params(top=True, which='both', labeltop=False, labelsize=14)
ax.yaxis.set_tick_params(right=True, which='both', labelright=False, labelsize=14)

plt.legend(fontsize=14, loc = 'upper left')
plt.xlim(0,1)
plt.savefig(f'../DNN/Models/{name}/NetworkScore{name}.png', dpi=600)


#calculate staterrors on the network score
dig_L = np.digitize(score0, edges_L)
dig_T = np.digitize(score1, edges_T)
staterrs_L = []
staterrs_T = []
for i in range(1,len(edges_L)):
    mask_L = dig_L == i
    mask_T = dig_T == i
    weights_L = weights0[mask_L]
    weights_T = weights1[mask_T]
    err_L = np.sqrt(np.sum(weights_L**2))
    staterrs_L.append(err_L)
    err_T = np.sqrt(np.sum(weights_T**2))
    staterrs_T.append(err_T)
 

#Input distributions for each DNN feature
'''
#for each feature 
feature_names = np.array([
 'DEta_VH',
 'DPhi_ll',
 'Higgs_E',
 'Higgs_mass',
 'Higgs_pT',
 'Higgs_phi',
 'Higgs_rapidity',
 'LeadPhoton_E',
 'LeadPhoton_Eta',
 'LeadPhoton_Phi',
 'LeadPhoton_pT',
 'Lep_pT_balance',
 'NegLep_E',
 'NegLep_Eta',
 'NegLep_Phi',
 'NegLep_pT',
 'Phi',
 'Phi1',
 'PosLep_E',
 'PosLep_Eta',
 'PosLep_Phi',
 'PosLep_pT',
 'SubLeadPhoton_E',
 'SubLeadPhoton_Eta',
 'SubLeadPhoton_Phi',
 'SubLeadPhoton_pT',
 'VBoson_E',
 'VBoson_mass',
 'VBoson_pT',
 'VBoson_phi',
 'VBoson_rapidity',
 'cosThetaStar',
 'costheta1',
 'costheta2',
 'm_VH'
])

new_feature_names = feature_names[cols_dnn]

all_labels = np.array(y_test_tensor)
dis0 = all_labels==0
dis1 = all_labels==1



X = X_test_tensor[:, cols_dnn]


for i, v_name in enumerate(new_feature_names):
    plt.figure(figsize=(8, 6))
    plt.hist(np.array(X[dis0])[:,i], density=True, color=mainc, histtype='step', bins=100, label='$\mathrm{Z}_L$')
    plt.hist(np.array(X[dis1])[:,i], density=True, color=secondc, histtype='step', bins=100,  label='$\mathrm{Z}_T$')

    plt.xlabel(f'{v_name}', loc='right')
    plt.ylabel('Event Fraction', loc='top')
    plt.title(v_name, fontsize=14)

    
    plt.legend()
    plt.savefig(f'../DNN/Models/{name}/input_{v_name}_{name}.png', dpi=600)

plt.show()
'''


labels = np.array(y_test_tensor) 

#Plot confusion matrix
cm = confusion_matrix(labels, preds, normalize= 'true')
legend = ['$\mathrm{Z}_L$','$\mathrm{Z}_T$'] #check correct way around 

plt.figure(figsize=(8,6))
sns.heatmap(cm*100, annot=True, fmt='.0f', xticklabels=legend, yticklabels=legend, cmap='Greens', vmin=0, vmax=100, cbar_kws={'label': 'True and False Rates [%]'}) #blues is standard
plt.xlabel('Predicted Label', loc='right')
plt.ylabel('Truth Label', loc='top')
plt.savefig(f'../DNN/Models/{name}/ConfMat_{name}.png', dpi=1000)
plt.show()

f.close()
