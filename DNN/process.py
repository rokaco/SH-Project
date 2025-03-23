"""
Created on Sat Jan 25 18:39:33 2025

Processing the dataset to prepare it to be used for DNN training.

@author: rosac
"""

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler


pd.options.mode.chained_assignment = None  #default='warn'

hfivesdir = r"../polarisations_final.h5" #absolute path to dataset 

#open the h5 file containing kinematic information
with h5py.File(hfivesdir) as f:
    df = pd.DataFrame(f['PolarisationStudies']['1d'][:])

#relabel polarisations 
df.replace({'polarisation_type':{1:0, 2:1}}, inplace=True) 

#take the log of all pT and energy values
#print(df.columns)
df['VBoson_pT'] = np.log(df['VBoson_pT'])
df['Higgs_pT'] = np.log(df['Higgs_pT'])
df['LeadPhoton_pT'] = np.log(df['LeadPhoton_pT'])
df['PosLep_pT'] = np.log(df['PosLep_pT'])
df['NegLep_pT'] = np.log(df['NegLep_pT'])
df['SubLeadPhoton_pT'] = np.log(df['SubLeadPhoton_pT'])
df['VBoson_E'] = np.log(df['VBoson_E'])
df['Higgs_E'] = np.log(df['Higgs_E'])
df['NegLep_E'] = np.log(df['NegLep_E'])
df['PosLep_E'] = np.log(df['PosLep_E'])
df['LeadPhoton_E'] = np.log(df['LeadPhoton_E'])
df['SubLeadPhoton_E'] = np.log(df['SubLeadPhoton_E'])


print(df.columns)
#set the polarisation_type label as the truth record for training
X = df.drop(['polarisation_type'], axis=1) #all columns for initial training -> then need to use MI to drop unimportant features 

#remove pseudorapidities in favour of rapidities for the vector bosons
X = X.drop(['Higgs_eta'], axis=1)
X = X.drop(['VBoson_eta'], axis=1)

#print(X.columns)
X = X.drop(['Lumi_weight'], axis=1)
y = df['polarisation_type']
Lumi = df['Lumi_weight']

#split the dataset into training and testing. Setting random seed ensures X and lumi will be split the same way
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1234)
Lumi_train, Lumi_test, y_train, y_test = train_test_split(Lumi, y, test_size = 0.4, random_state = 1234)

#reset indices after split
Lumi_train.reset_index(drop = True, inplace=True)
Lumi_test.reset_index(drop=True,inplace=True)
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

#normalise the dataset 
cols_to_normalize = X_train.select_dtypes(include='number').columns.to_list()  
scaler = MinMaxScaler()
scaler.fit(X_train[cols_to_normalize])

X_train[cols_to_normalize] = scaler.transform(X_train[cols_to_normalize])
X_test[cols_to_normalize] = scaler.transform(X_test[cols_to_normalize])

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

#columns that will be needed for inverting the normalisation
cols_inv = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34] #all cols 
#columns used to train the DNN following MI search
cols_dnn = [0,1,2,4,7,10,11,12,15,18,21,22,25,26,28,31,32,34]


name = "mi15" #name of the model 
weights = torch.tensor([0.5]) #since resampled 
