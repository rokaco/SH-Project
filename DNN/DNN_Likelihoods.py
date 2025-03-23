# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:47:10 2025

Performing likelihood fits on the DNN results and plotting the likelihood curve

@author: rosac
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import pyhf
from DNN_plots import counts_T, counts_L, mainc, secondc, thirdc, name, staterrs_L, staterrs_T
import iminuit
from scipy.optimize import fsolve


T_counts = np.sum(counts_T)
L_counts = np.sum(counts_L)
print('Total L counts: ', L_counts)
print('Total T counts: ', T_counts)


#fitting the NLL curve for each luminosity
def fitting(lumi, counts_L, counts_T):
    
    twicenll_list = np.empty(0)
    percentage = np.empty(0)
    
    #updating bin counts due to luminosity
    counts_L = counts_L/lumi
    counts_T = counts_T/lumi
    
    total_T_counts = np.sum(counts_T)
    total_L_counts = np.sum(counts_L)
    
    #calculating proportion of L events
    L_prop = total_L_counts/(total_T_counts + total_L_counts)
    
    #loading pyhf model    
    with open("spec.json") as serialized:
        spec = json.load(serialized)  
        #range and number of different models scanned over
        weightplot = np.linspace(-0.8,1.4,10)
        for x in weightplot:
            
            #adjusting L weighting to scan around the DNN predictions - T stays constant
            L = 1 + x
            T = 1
            
            #updating signal counts
            spec['channels'][0]['samples'][0]['data'] = counts_L * L 
            
            #updating background counts
            spec['channels'][0]['samples'][1]['data'] =  counts_T
            
            #updating staterrors - important to account for luminosity change
            spec['channels'][0]['samples'][0]['modifiers'][2]['data'] = np.array(staterrs_L)*L/lumi
            spec['channels'][0]['samples'][1]['modifiers'][2]['data'] = np.array(staterrs_T)*T/lumi
            
            #updating observations, which is the same for every model tested
            spec['observations'][0]['data'] = counts_L + counts_T
            
            #initialising the model
            workspace = pyhf.Workspace(spec)
            model = workspace.model()
            aux = model.config.auxdata
            
            #calculating total counts for each model, which can change due to L weightings
            total_counts=np.sum(spec['channels'][0]['samples'][1]['data']+spec['channels'][0]['samples'][0]['data'])
            
            #adding in auxiliary data
            data = np.append(spec['observations'][0]['data'],aux)
               
            #performing likelihood fit
            bestfit_pars, twice_nll = pyhf.infer.mle.fit(data, model, return_fitted_val = True)
            
            #calculating (1) signal strength parameter or (2) raw percentage of L events for each model               
            percentage = np.append(percentage, (L_counts*L/(L_counts*L + T_counts*T)/L_prop))
            #percentage = np.append(percentage, (L_counts*L*100/(L_counts*L + T_counts*T)))
            
            twicenll_list = np.append(twicenll_list,twice_nll)
            
            
        #calculate best model - expect this to be the one for which L=1
        nll_0 = min(twicenll_list)
        
        #calculate -2DeltaNLL test statistic
        Delta = twicenll_list - nll_0
             
        return Delta, percentage

#fitting the models for each luminosity
Delta_1, percentage_1 = fitting(1, counts_L, counts_T)
Delta_3, percentage_3 = fitting(3, counts_L, counts_T)
Delta_10, percentage_10 = fitting(10, counts_L, counts_T)

#plotting the likelihood curve
fig, ax = plt.subplots(figsize=(8,6), dpi=1000)
ax.set_ylim(0,max(Delta_1)+1)
plt.minorticks_on()
ax.xaxis.set_tick_params(top=True, which='both', labeltop=False, labelsize=14)
ax.yaxis.set_tick_params(right=True, which='both', labelsize=14)

plt.plot(percentage_1, Delta_1, color=mainc, linewidth = 2, label = '$3000\,$fb$^{-1}$')
plt.plot(percentage_3, Delta_3, color=secondc, linewidth = 2, linestyle = 'dashed', label = '$1000\,$fb$^{-1}$')
plt.plot(percentage_10, Delta_10, color='olive', linewidth = 2, linestyle = 'dashed', label = '$300\,$fb$^{-1}$')

#plotting sigma lines
plt.xlabel('$\mu_{ZH}$', loc='center', labelpad = 10, fontsize=17)
plt.ylabel('$-2 \Delta$NLL',loc='center', labelpad = 10, fontsize=17)
plt.axhline(1, color='indianred', linestyle='dotted', linewidth = 1)
plt.axhline(3.841, color='indianred', linestyle='dotted', linewidth = 1)
plt.axhline(8.99, color='indianred', linestyle='dotted', linewidth = 1)
plt.axhline(16.01, color='indianred', linestyle='dotted', linewidth = 1)

ax.text(0.9, 10, '  Simulation \n $\sqrt{s} = 14\,$TeV', fontsize = 17)
ax.text(1.42, 0.75, '1$\sigma$', fontsize = 12)
ax.text(1.42, 3.59, '2$\sigma$', fontsize = 12)
ax.text(1.42, 8.74, '3$\sigma$', fontsize = 12)
ax.text(1.42, 15.76, '4$\sigma$', fontsize = 12)

plt.legend(loc = 'upper center', fontsize = 14)

plt.savefig(f'../DNN/Models/{name}/SignalStrength{name}.png', dpi=600)

#y-values of sigma lines
sigma_vals = [1, 3.841, 8.99, 16.01]

#finding intersection points with sigma lines
def find_intersection(Delta, percentage, y):

  values = np.empty(0)
    
  idx_1 = np.argwhere(np.diff(np.sign(Delta - y[0]))).flatten()
  values = np.append(values, percentage[idx_1])
    
  idx_2 = np.argwhere(np.diff(np.sign(Delta - y[1]))).flatten()
  values = np.append(values, percentage[idx_2])
    
  idx_3 = np.argwhere(np.diff(np.sign(Delta - y[2]))).flatten()
  values = np.append(values, percentage[idx_3])
    
  idx_4 = np.argwhere(np.diff(np.sign(Delta - y[3]))).flatten()
  values = np.append(values, percentage[idx_4])
    
  return values


values_1 = find_intersection(Delta_1, percentage_1, sigma_vals)
values_3 = find_intersection(Delta_3, percentage_3, sigma_vals)
values_10 = find_intersection(Delta_10, percentage_10, sigma_vals)

#print(values_1)
#print(values_3)
#print(values_10)
