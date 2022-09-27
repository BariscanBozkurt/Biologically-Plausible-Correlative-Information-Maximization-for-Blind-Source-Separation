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
from CorInfoMaxBSS import OnlineCorInfoMax

import warnings
warnings.filterwarnings("ignore")

if not os.path.exists("../Results"):
    os.mkdir("../Results")

pickle_name_for_results = "simulation_results_Be_InitializationV1.pkl"

N = 500000 ## Number of data points
NumberofSources = 5 ## Number of source vectors
NumberofMixtures = 10 ## Number of mixture vectors

s_dim = NumberofSources
x_dim = NumberofMixtures

SNRlevel = 30 ## Signal to noise ratio in terms of dB (for adding noise to the mixtures)
Be_gain_list = [1000, 2000, 5000, 10000]
NumAverages = 50 ## Number of realizations to average for each algorithm
seed_list = np.array([1436852 * i for i in range(NumAverages)]) ## Seeds for reproducibility
########################################################################################
########################################################################################
###                                                                                  ###
###                        SIMULATION                                                ###
###                                                                                  ###
########################################################################################
########################################################################################

RESULTS_DF = pd.DataFrame( columns = ['trial', 'seed', 'Model', 'Be_gain', 'SINR', 'SINRlist', 'SNR', 'S', 'A', 'X', 'Wf', 'SNRinp', 'execution_time'])

####### YOU MIGHT WANT TO CHANGE THE DEBUG ITERATION POINT FOR MEMORY PURPOSES #######
debug_iteration_point = 25000 # SINR measurement per 10000 iteration

for iter1 in range(NumAverages): ## Loop over number of averages
    seed_ = seed_list[iter1] ## Set the seed for reproducibility
    np.random.seed(seed_)
    trial = iter1
    ###### GAUSSIAN MIXING SCENARIO AND WGN NOISE ADDING ##############################
    S = generate_correlated_copula_sources(rho = 0.0, df = 4, n_sources = NumberofSources, 
                                            size_sources = N , decreasing_correlation = False) ## GENERATE CORRELATED COPULA

    A = np.random.randn(NumberofMixtures,NumberofSources)
    X = np.dot(A,S)
    Xnoisy, NoisePart = addWGN(X, SNRlevel, return_noise = True) ## Add White Gaussian Noise with 30 dB SNR
    SNRinplevel = 10 * np.log10(np.sum(np.mean((Xnoisy - NoisePart) ** 2, axis = 1)) / np.sum(np.mean(NoisePart ** 2, axis = 1)))

    for Be_gain in Be_gain_list:
        #######################################################
        #        Online CorInfoMax Setup 1                    #
        #######################################################
        try: # Try Except for SVD did not converge error (or for any other error)
            lambday = 1 - 1e-1/10
            lambdae = 1 - 1e-1/3

            # Inverse output covariance
            By = 5 * np.eye(s_dim)
            # Inverse error covariance
            Be = Be_gain * np.eye(s_dim)

            modelCorInfoMax = OnlineCorInfoMax( s_dim = s_dim, x_dim = x_dim, muW = 30*1e-3, lambday = lambday,
                                                lambdae = lambdae, By = By, Be = Be, neural_OUTPUT_COMP_TOL = 1e-6,
                                                set_ground_truth = True, S = S, A = A)
            with Timer() as t:
                modelCorInfoMax.fit_batch_nnantisparse( X = Xnoisy, n_epochs = 1, neural_dynamic_iterations = 500,
                                                        plot_in_jupyter = False,
                                                        neural_lr_start = 0.9,
                                                        neural_lr_stop = 0.001, 
                                                        debug_iteration_point = debug_iteration_point, 
                                                        shuffle = False)
            ######### Evaluate the Performance of CorInfoMax Framework ###########################
            SINRlistCorInfoMax = modelCorInfoMax.SIR_list
            WfCorInfoMax = modelCorInfoMax.compute_overall_mapping(return_mapping = True)
            YCorInfoMax = WfCorInfoMax @ Xnoisy
            SINRCorInfoMax, SNRCorInfoMax, _, _, _ = evaluate_bss(WfCorInfoMax, YCorInfoMax, A, S, mean_normalize_estimations = False)
            
            CorInfoMax_Dict = {'trial' : trial, 'seed' : seed_, 'Model' : 'CorInfoMax', 'Be_gain': Be_gain,
                               'SINR' : SINRCorInfoMax, 'SINRlist':  SINRlistCorInfoMax, 'SNR' : SNRCorInfoMax,
                               'S' : S, 'A' : A, 'X': Xnoisy, 'Wf' : WfCorInfoMax, 'SNRinp' : SNRinplevel, 
                               'execution_time' : t.interval}
        except Exception as e:
            print(str(e))
            CorInfoMax_Dict = {'trial' : trial, 'seed' : seed_, 'Model' : 'CorInfoMax', 'Be_gain': Be_gain,
                               'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                               'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                               'execution_time' : None}

        RESULTS_DF = RESULTS_DF.append(CorInfoMax_Dict, ignore_index = True)

        RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))

RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))