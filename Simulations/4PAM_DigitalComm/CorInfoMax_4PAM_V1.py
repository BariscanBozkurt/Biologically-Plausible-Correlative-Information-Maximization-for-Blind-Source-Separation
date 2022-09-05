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
from dsp_utils import addWGN, map_estimates_to_symbols, SER
from bss_utils import evaluate_bss, signed_and_permutation_corrected_sources
from numba_utils import *
######## IMPORT THE REQUIRED ALGORITHMS ########
from CorInfoMaxBSS import OnlineCorInfoMax

import warnings
warnings.filterwarnings("ignore")

if not os.path.exists("../Results"):
    os.mkdir("../Results")

pickle_name_for_results = "simulation_results_4PAM_V1.pkl"

N = 100000
NumberofSources = 5
NumberofMixtures = 10

s_dim = NumberofSources
x_dim = NumberofMixtures

SNRlevel = 30 ## Signal to noise ratio in terms of dB (for adding noise to the mixtures)

NumAverages = 100 ## Number of realizations to average for each algorithm
seed_list = np.array([1496258 * i for i in range(NumAverages)]) ## Seeds for reproducibility
########################################################################################
########################################################################################
###                                                                                  ###
###                        SIMULATION                                                ###
###                                                                                  ###
########################################################################################
########################################################################################

RESULTS_DF = pd.DataFrame( columns = ['trial', 'seed', 'Model', 'SINR', 'SINRlist', 'SNR', 'SER', 'S', 'A', 'X', 'Wf', 'SNRinp', 'execution_time'])

####### YOU MIGHT WANT TO CHANGE THE DEBUG ITERATION POINT FOR MEMORY PURPOSES #######
debug_iteration_point = 25000 # SINR measurement per 10000 iteration

for iter1 in range(NumAverages): ## Loop over number of averages
    seed_ = seed_list[iter1] ## Set the seed for reproducibility
    np.random.seed(seed_)
    trial = iter1

    ###### GAUSSIAN MIXING SCENARIO AND WGN NOISE ADDING ##############################
    S = 2 * (np.random.randint(0,4,(NumberofSources, N))) - 3

    A = np.random.randn(NumberofMixtures,NumberofSources)
    X = np.dot(A,S)
    Xnoisy, NoisePart = addWGN(X, SNRlevel, return_noise = True) ## Add White Gaussian Noise with 30 dB SNR
    SNRinplevel = 10 * np.log10(np.sum(np.mean((Xnoisy - NoisePart) ** 2, axis = 1)) / np.sum(np.mean(NoisePart ** 2, axis = 1)))
    
    #######################################################
    #        Online CorInfoMax Setup 1                    #
    #######################################################
    try: # Try Except for SVD did not converge error (or for any other error)
        lambday = 1 - 1e-1 / 10
        lambdae = 1 - 1e-1 / 10
        s_dim = S.shape[0]
        x_dim = X.shape[0]

        # Inverse output covariance
        By = 5 * np.eye(s_dim)
        # Inverse error covariance
        Be = 1000 * np.eye(s_dim)

        debug_iteration_point = 25000
        modelCorInfoMax = OnlineCorInfoMax(
                                            s_dim=s_dim,
                                            x_dim=x_dim,
                                            muW=30 * 1e-3,
                                            lambday=lambday,
                                            lambdae=lambdae,
                                            By=By,
                                            Be=Be,
                                            neural_OUTPUT_COMP_TOL=1e-6,
                                            set_ground_truth=True,
                                            S=S,
                                            A=A,
                                        )
        with Timer() as t:
            modelCorInfoMax.fit_batch_antisparse(
                                                    X=Xnoisy/3,
                                                    n_epochs=1,
                                                    neural_dynamic_iterations=500,
                                                    plot_in_jupyter=False,
                                                    neural_lr_start=0.9,
                                                    neural_lr_stop=0.001,
                                                    debug_iteration_point=debug_iteration_point,
                                                    shuffle=False,
                                                )
        ######### Evaluate the Performance of CorInfoMax Framework ###########################
        SINRlistCorInfoMax = modelCorInfoMax.SIR_list
        WfCorInfoMax = modelCorInfoMax.compute_overall_mapping(return_mapping = True)
        YCorInfoMax = WfCorInfoMax @ Xnoisy
        SINRCorInfoMax, SNRCorInfoMax, _, _, _ = evaluate_bss(WfCorInfoMax, YCorInfoMax, A, S, mean_normalize_estimations = False)
        
        Y_ = signed_and_permutation_corrected_sources(S, YCorInfoMax)
        Y_pred = map_estimates_to_symbols(Y_, np.array([-3, -1, 1, 3]))
        symbol_error_rate = SER(S, Y_pred)
        CorInfoMax_Dict = {'trial' : trial, 'seed' : seed_, 'Model' : 'CorInfoMax',
                            'SINR' : SINRCorInfoMax, 'SINRlist':  SINRlistCorInfoMax, 'SNR' : SNRCorInfoMax,
                            'SER': symbol_error_rate, 'S' : S, 'A' : A, 'X': Xnoisy, 'Wf' : WfCorInfoMax, 
                            'SNRinp' : SNRinplevel, 'execution_time' : t.interval}
    except Exception as e:
        print(str(e))
        CorInfoMax_Dict = { 'trial' : trial, 'seed' : seed_, 'Model' : 'CorInfoMax',
                            'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None, 'SER': None,
                            'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                            'execution_time' : None}

    RESULTS_DF = RESULTS_DF.append(CorInfoMax_Dict, ignore_index = True)

    RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))

RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))