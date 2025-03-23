# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:40:51 2025

Calculating the MI for each feature

@author: rosac
"""

import pandas as pd 
import h5py
import numpy as np
from process import X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, name, cols_dnn
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

#calculate and normalise MI 

X = X_test_tensor[:, cols_dnn]

mi = mutual_info_classif(X, y_test_tensor, random_state=42)

mi_normalized = MinMaxScaler().fit_transform(mi.reshape(-1, 1))

#print results 
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

feature_importances = pd.DataFrame({'Feature': new_feature_names, 'Importance': mi_normalized.flatten()})

#print results
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print(feature_importances)
