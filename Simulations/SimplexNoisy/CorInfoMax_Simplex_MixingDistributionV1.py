########## IMPORT REQUIIRED LIBRARIES ##########
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from numba import njit
########## IMPORT UTILITY SCRIPTS ##############
import sys
sys.path.insert(0, '../../src')
from general_utils import *
from dsp_utils import *
from bss_utils import *
from numba_utils import *
######## IMPORT THE REQUIRED ALGORITHMS ########
from LDMIBSS import LDMIBSS
from CorInfoMaxBSS import OnlineCorInfoMax

import warnings
warnings.filterwarnings("ignore")

if not os.path.exists("../Results"):
    os.mkdir("../Results")

pickle_name_for_results = "simulation_results_simplex_mixing_distributionV1.pkl"

N = 500000 ## Number of data points
NumberofSources = 5 ## Number of source vectors
NumberofMixtures = 10 ## Number of mixture vectors

s_dim = NumberofSources
x_dim = NumberofMixtures

mixing_distribution_list = ["normal", "uniform", "uniform2", "laplace"]
SNRlevel = 30 ## Signal to noise ratio in terms of dB (for adding noise to the mixtures)

NumAverages = 50 ## Number of realizations to average for each algorithm
seed_list = np.array([1846 * i for i in range(NumAverages)]) ## Seeds for reproducibility

########################################################################################
########################################################################################
###                                                                                  ###
###                        SIMULATION                                                ###
###                                                                                  ###
########################################################################################
########################################################################################

RESULTS_DF = pd.DataFrame( columns = ['mixing_dist', 'trial', 'seed', 'Model', 'SINR', 'SINRlist', 'SNR', 'S', 'A', 'X', 'Wf', 'SNRinp', 'execution_time'])

####### YOU MIGHT WANT TO CHANGE THE DEBUG ITERATION POINT FOR MEMORY PURPOSES #######
debug_iteration_point = 25000 # SINR measurement per 10000 iteration

for iter1 in range(NumAverages): ## Loop over number of averages
    seed_ = seed_list[iter1] ## Set the seed for reproducibility
    np.random.seed(seed_)
    trial = iter1
    for mixing_dist in (mixing_distribution_list):
        S = generate_uniform_points_in_simplex(NumberofSources, N)

        Szeromean = S - S.mean(axis = 1).reshape(-1,1)

        if mixing_dist == "normal":
            A = np.random.standard_normal(size=(NumberofMixtures, NumberofSources))
        elif mixing_dist == "uniform":
            A = np.random.uniform(-1,1, (NumberofMixtures,NumberofSources))
        elif mixing_dist == "uniform2":
            A = np.random.uniform(-2,2, (NumberofMixtures,NumberofSources))
        elif mixing_dist == "laplace":
            A = np.random.laplace(0,1, (NumberofMixtures,NumberofSources))
        
        X = np.dot(A,S)

        Xnoisy, NoisePart = addWGN(X, SNRlevel, return_noise = True)

        SNRinplevel = 10 * np.log10(np.sum(np.mean((X - NoisePart)**2, axis = 1)) / np.sum(np.mean(NoisePart**2, axis = 1)))

        #######################################################
        #        Online CorInfoMax Setup                      #
        #######################################################
        try: # Try Except for SVD did not converge error (or for any other error)
            lambday = 1 - 1e-1/10
            lambdae = 1 - 1e-1/10
            s_dim = S.shape[0]
            x_dim = X.shape[0]

            # Inverse output covariance
            By = 5 * np.eye(s_dim)
            # Inverse error covariance
            Be = 1000 * np.eye(s_dim)

            modelCorInfoMax = OnlineCorInfoMax(s_dim = s_dim, x_dim = x_dim, muW = 30*1e-3, lambday = lambday,
                                                lambdae = lambdae, By = By, Be = Be, neural_OUTPUT_COMP_TOL = 1e-6,
                                                set_ground_truth = True, S = S, A = A)
            with Timer() as t:
                modelCorInfoMax.fit_batch_simplex( X = Xnoisy, n_epochs = 1, neural_dynamic_iterations = 500,
                                                    plot_in_jupyter = False, neural_lr_start = 0.1,
                                                    neural_lr_stop = 0.001, debug_iteration_point = debug_iteration_point, 
                                                    shuffle = True)
            ######### Evaluate the Performance of CorInfoMax Framework ###########################
            SINRlistCorInfoMax = modelCorInfoMax.SIR_list
            WfCorInfoMax = modelCorInfoMax.compute_overall_mapping(return_mapping = True)
            YCorInfoMax = WfCorInfoMax @ Xnoisy
            SINRCorInfoMax, SNRCorInfoMax, _, _, _ = evaluate_bss(WfCorInfoMax, YCorInfoMax, A, Szeromean, mean_normalize_estimations = True)
            
            # YCorInfoMax_zeromean = YCorInfoMax - YCorInfoMax.mean(axis = 1).reshape(-1,1)
            # perm = find_permutation_between_source_and_estimation(Szeromean,YCorInfoMax_zeromean)
            # YCorInfoMax_ = YCorInfoMax[perm,:]
            # coef_ = ((YCorInfoMax_ * S).sum(axis = 1) / (YCorInfoMax_ * YCorInfoMax_).sum(axis = 1)).reshape(-1,1)
            # YCorInfoMax_ = coef_ * YCorInfoMax_
            # SINRCorInfoMax = 10*np.log10(CalculateSINRjit(YCorInfoMax_, S, False)[0])
            # SNRCorInfoMax = snr_jit(S, YCorInfoMax_)
            
            CorInfoMax_Dict = {'mixing_dist' : mixing_dist, 'trial' : trial, 'seed' : seed_, 'Model' : 'CorInfoMax',
                               'SINR' : SINRCorInfoMax, 'SINRlist':  SINRlistCorInfoMax, 'SNR' : SNRCorInfoMax,
                               'S' : S, 'A' : A, 'X': Xnoisy, 'Wf' : WfCorInfoMax, 'SNRinp' : SNRinplevel, 
                               'execution_time' : t.interval}
        except Exception as e:
            print(str(e))
            CorInfoMax_Dict = { 'mixing_dist' : mixing_dist, 'trial' : trial, 'seed' : seed_, 'Model' : 'CorInfoMax',
                                'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                                'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                                'execution_time' : None}

        #######################################################
        #                 LDMI BATCH                          #
        #######################################################
        try:
            modelLDMI = LDMIBSS(s_dim = s_dim, x_dim = x_dim,    #LDMIBSS(s_dim=s_dim, x_dim=x_dim, set_ground_truth=True, S=S, A=A)
                                set_ground_truth = True, S = S[:,:10000], A = A)
            with Timer() as t:
                ## Feed 10000 samples of the mixtures, that is enough for LDMI
                modelLDMI.fit_batch_simplex(  Xnoisy[:,:10000], epsilon = 1e-5, mu_start = 100, n_iterations = 10000, 
                                              method = "correlation", debug_iteration_point = debug_iteration_point,
                                              plot_in_jupyter = False)
            
            ######### Evaluate the Performance of LDMIBSS Framework ###########################
            SINRlistLDMI = modelLDMI.SIR_list 
            WfLDMI = modelLDMI.W
            YLDMI = WfLDMI @ Xnoisy
            # SINRLDMI, SNRLDMI, _, _, _ = evaluate_bss(WfLDMI, YLDMI, A, Szeromean, mean_normalize_estimations = True)

            YLDMI_zeromean = YLDMI - YLDMI.mean(axis = 1).reshape(-1,1)
            perm = find_permutation_between_source_and_estimation(Szeromean,YLDMI_zeromean)
            YLDMI_ = YLDMI[perm,:]
            coef_ = ((YLDMI_ * S).sum(axis = 1) / (YLDMI_ * YLDMI_).sum(axis = 1)).reshape(-1,1)
            YLDMI_ = coef_ * YLDMI_
            SINRLDMI = 10*np.log10(CalculateSINRjit(YLDMI_, S, False)[0])
            SNRLDMI = snr_jit(S, YLDMI_)

            LDMI_Dict = {'mixing_dist' : mixing_dist, 'trial' : trial, 'seed' : seed_, 'Model' : 'LDMI',
                         'SINR' : SINRLDMI, 'SINRlist':  SINRlistLDMI, 'SNR' : SNRLDMI,
                         'S' : None, 'A' : None, 'X': None, 'Wf' : WfLDMI, 'SNRinp' : None, 
                         'execution_time' : t.interval}
        except Exception as e:
            print(str(e))
            LDMI_Dict = {'mixing_dist' : mixing_dist, 'trial' : trial, 'seed' : seed_, 'Model' : 'LDMI',
                         'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                         'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                         'execution_time' : None}

        RESULTS_DF = RESULTS_DF.append(CorInfoMax_Dict, ignore_index = True)
        RESULTS_DF = RESULTS_DF.append(LDMI_Dict, ignore_index = True)
        RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))

RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))