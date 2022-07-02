import sys
import os
os.chdir("..")
os.chdir("..")
os.chdir("./src")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
from tqdm import tqdm
from scipy.stats import invgamma, chi2, t
from numba import njit, jit
from time import time
from LDMIBSS import *
from scipy.signal import lfilter
import mne 
from mne.preprocessing import ICA
import warnings
warnings.filterwarnings("ignore")

notebook_name = 'Antisparse_Copula'

N = 250000
NumberofSources = 5
NumberofMixtures = 10

M = NumberofMixtures
r = NumberofSources
#Define number of sampling points
n_samples = N
#Degrees of freedom 
df = 4

# Correlation values
rholist=np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
SNR = 30 # dB

NumAverages = 100

seed_list = np.array([1575*i for i in range(1, NumAverages+1)])


########################################################################################
########################################################################################
###                                                                                  ###
###                        SIMULATION                                                ###
###                                                                                  ###
########################################################################################
########################################################################################
CorInfoMax_SINR_DF = pd.DataFrame(columns = ['rho','trial', 'seed', 'Model', 'SINR', 'SINRlist', 'SNR', 'S', 'A', 'Wf', 'SNRinp'])

####### YOU MIGHT WANT TO CHANGE THE DEBUG ITERATION POINT FOR MEMORY PURPOSES #######
debug_iteration_point = 10000 # SIR measurement per 10000 iteration

for iter1 in range(NumAverages):
    seed_ = seed_list[iter1]
    np.random.seed(seed_)
    iter0=-1
    trial = iter1
    for rho in (rholist):
        
        iter0=iter0+1
        
        S = generate_correlated_copula_sources(rho = rho, df = 4, n_sources = NumberofSources, size_sources = N , 
                                               decreasing_correlation = True)
        S = 2 * S - 1
        # Generate Mxr random mixing from i.i.d N(0,1)
        A = np.random.randn(NumberofMixtures,NumberofSources)
        X = np.dot(A,S) 
        X, NoisePart = addWGN(X, SNR, return_noise = True)
        SNRinp = 10 * np.log10(np.sum(np.mean((X - NoisePart)**2, axis = 1)) / np.sum(np.mean(NoisePart**2, axis = 1)))

        #######################################################
        #                   CorInfoMax                        #
        #######################################################
        try: # Try Except for SVD did not converge error
            lambday = 1 - 1e-1/10
            lambdae = 1 - 1e-1/10
            s_dim = S.shape[0]
            x_dim = X.shape[0]

            # Inverse output covariance
            By = 5 * np.eye(s_dim)
            # Inverse error covariance
            Be = 10000 * np.eye(s_dim)

            debug_iteration_point = 10000
            modelCorInfo = OnlineLDMIBSS(s_dim = s_dim, x_dim = x_dim, muW = 30*1e-3, lambday = lambday,
                                  lambdae = lambdae, By = By, Be = Be, neural_OUTPUT_COMP_TOL = 1e-6,
                                  set_ground_truth = True, S = S, A = A)

            modelCorInfo.fit_batch_antisparse(X = X, n_epochs = 1, neural_dynamic_iterations = 500,
                                       plot_in_jupyter = True, neural_lr_start = 0.9,
                                       neural_lr_stop = 0.001, debug_iteration_point = debug_iteration_point, 
                                       shuffle = True)

            Wldmi = modelCorInfo.compute_overall_mapping(return_mapping = True)
        except Exception as e:
            print(str(e))
            CorInfoMax_dict = {'rho' : rho,'trial' : trial, 'seed' : seed_, 'Model' : 'CorInfoMax', 
                               'SINR' : -999, 'SINRlist' : str(e),  'SNR' : None, 'S' : S, 'A' :A, 'Wf': None, 'SNRinp' : SNRinp}