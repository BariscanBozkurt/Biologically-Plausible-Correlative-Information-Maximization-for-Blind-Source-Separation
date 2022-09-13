########## IMPORT REQUIIRED LIBRARIES ##########
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from numba import njit
########## IMPORT UTILITY SCRIPTS ##############
import sys
sys.path.insert(0, '../../src')
sys.path.insert(0, '../Results')
from general_utils import *
from dsp_utils import *
from bss_utils import *
from numba_utils import *
######## IMPORT THE REQUIRED ALGORITHMS ########
from WSMBSS import *
import warnings
warnings.filterwarnings("ignore")

import warnings
warnings.filterwarnings("ignore")

if not os.path.exists("../ResultsWSM"):
    os.mkdir("../ResultsWSM")

pickle_name_for_results = "simulation_resultsWSM_correlated_antisparseV2.pkl"

df_anti_results = pd.read_pickle(r"../Results/simulation_results_correlated_antisparseV2.pkl").iloc[:int(9*7*100)]
df_anti_results = df_anti_results.loc[df_anti_results['Model'] == "CorInfoMax"][["rho", "trial", "seed", "S", "A", "X", "SNRinp"]]

N = 500000 ## Number of data points
NumberofSources = 5 ## Number of source vectors
NumberofMixtures = 10 ## Number of mixture vectors

s_dim = NumberofSources
x_dim = NumberofMixtures
WSM_INPUT_STD = 0.5

########################################################################################
########################################################################################
###                                                                                  ###
###                        SIMULATION                                                ###
###                                                                                  ###
########################################################################################
########################################################################################

RESULTS_DF = pd.DataFrame( columns = ['rho', 'trial', 'seed', 'Model', 'SINR', 'SINRlist', 'SNR', 'S', 'A', 'X', 'Wf', 'SNRinp', 'execution_time'])

####### YOU MIGHT WANT TO CHANGE THE DEBUG ITERATION POINT FOR MEMORY PURPOSES #######
debug_iteration_point = 25000 # SINR measurement per 10000 iteration

for ii in range(df_anti_results.shape[0]):
    rho = df_anti_results['rho'].iloc[ii]
    trial = df_anti_results['trial'].iloc[ii]
    seed_ = df_anti_results['seed'].iloc[ii]
    S = df_anti_results['S'].iloc[ii]
    A = df_anti_results['A'].iloc[ii]
    X = df_anti_results['X'].iloc[ii]
    X = (WSM_INPUT_STD * (X / X.std(1)[:,np.newaxis]))

    #######################################################
    #        Online CorInfoMax Setup 1                    #
    #######################################################
    try: # Try Except for SVD did not converge error (or for any other error)
        if rho > 0.4:
            gamma_start = 0.25
            gamma_stop = 5 * 1e-4
        else:
            gamma_start = 0.6
            gamma_stop = 1e-3

        gammaM_start = [gamma_start, gamma_start]
        gammaM_stop = [gamma_stop, gamma_stop]
        gammaW_start = [gamma_start, gamma_start]
        gammaW_stop = [gamma_stop, gamma_stop]

        OUTPUT_COMP_TOL = 1e-6
        MAX_OUT_ITERATIONS = 3000
        LayerGains = [1, 1]
        LayerMinimumGains = [0.2, 0.2]
        LayerMaximumGains = [1e6, 5]
        WScalings = [0.005, 0.005]
        GamScalings = [2, 1]
        zeta = 5 * 1e-5
        beta = 0.5
        muD = [1.125, 0.2]

        h_dim = s_dim
        samples = S.shape[1]
        W_HX = np.eye(h_dim, x_dim)
        W_YH = np.eye(s_dim, h_dim)

        modelWSM = OnlineWSMBSS(
                                s_dim=s_dim,
                                x_dim=x_dim,
                                h_dim=h_dim,
                                gammaM_start=gammaM_start,
                                gammaM_stop=gammaM_stop,
                                gammaW_start=gammaW_start,
                                gammaW_stop=gammaW_stop,
                                beta=beta,
                                zeta=zeta,
                                muD=muD,
                                WScalings=WScalings,
                                GamScalings=GamScalings,
                                W_HX=W_HX,
                                W_YH=W_YH,
                                DScalings=LayerGains,
                                LayerMinimumGains=LayerMinimumGains,
                                LayerMaximumGains=LayerMaximumGains,
                                neural_OUTPUT_COMP_TOL=OUTPUT_COMP_TOL,
                                set_ground_truth=True,
                                S=S,
                                A=A,
                            )

        modelWSM.fit_batch_antisparse(
                                        X,
                                        n_epochs=1,
                                        neural_lr_start=0.75,
                                        neural_lr_stop=0.05,
                                        synaptic_lr_decay_divider=5,
                                        debug_iteration_point=debug_iteration_point,
                                        plot_in_jupyter=False,
                                    )
        ######### Evaluate the Performance of WSM Framework ###########################
        SINRlistWSM = modelWSM.SIR_list
        WfWSM = modelWSM.compute_overall_mapping(return_mapping = True)
        YWSM = WfWSM @ X
        SINRWSM, SNRWSM, _, _, _ = evaluate_bss(WfWSM, YWSM, A, S, mean_normalize_estimations = False)
        
        WSM_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'WSM',
                            'SINR' : SINRWSM, 'SINRlist':  SINRlistWSM, 'SNR' : SNRWSM,
                            'S' : None, 'A' : None, 'X': None, 'Wf' : WfWSM, 'SNRinp' : None, 
                            'execution_time' : None}
    except Exception as e:
        print(str(e))
        WSM_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'WSM',
                            'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                            'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                            'execution_time' : None}

    RESULTS_DF = RESULTS_DF.append(WSM_Dict, ignore_index = True)
    RESULTS_DF.to_pickle(os.path.join("../ResultsWSM", pickle_name_for_results))

RESULTS_DF.to_pickle(os.path.join("../ResultsWSM", pickle_name_for_results))











