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

pickle_name_for_results = "simulation_resultsWSM_sparse_noisyV1.pkl"

df_sparse_results = pd.read_pickle(r"../Results/simulation_results_sparse_noisyV1.pkl")
df_sparse_results = df_sparse_results.loc[df_sparse_results['Model'] == "CorInfoMax"][["SNRlevel", "trial", "seed", "S", "A", "X", "SNRinp"]]

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

RESULTS_DF = pd.DataFrame( columns = ['SNRlevel', 'trial', 'seed', 'Model', 'SINR', 'SINRlist', 'SNR', 'S', 'A', 'X', 'Wf', 'SNRinp', 'execution_time'])

####### YOU MIGHT WANT TO CHANGE THE DEBUG ITERATION POINT FOR MEMORY PURPOSES #######
debug_iteration_point = 25000 # SINR measurement per 10000 iteration

for ii in range(df_sparse_results.shape[0]):
    SNRlevel = df_sparse_results['SNRlevel'].iloc[ii]
    trial = df_sparse_results['trial'].iloc[ii]
    seed_ = df_sparse_results['seed'].iloc[ii]
    S = df_sparse_results['S'].iloc[ii]
    A = df_sparse_results['A'].iloc[ii]
    X = df_sparse_results['X'].iloc[ii]
    X = (WSM_INPUT_STD * (X / X.std(1)[:,np.newaxis]))

    #######################################################
    #        Online WSM Setup                             #
    #######################################################
    try: # Try Except for SVD did not converge error (or for any other error)
        MUS = 0.25

        gammaM_start = [MUS, MUS]
        gammaM_stop = [1e-3, 1e-3]
        gammaW_start = [MUS, MUS]
        gammaW_stop = [1e-3, 1e-3]

        OUTPUT_COMP_TOL = 1e-5
        LayerGains = [8, 1]
        LayerMinimumGains = [1e-6, 1]
        LayerMaximumGains = [1e6, 1.001]
        WScalings = [0.0033, 0.0033]
        GamScalings = [0.02, 0.02]
        zeta = 1 * 1e-4
        beta = 0.5
        muD = [20, 1e-2]

        s_dim = S.shape[0]
        x_dim = X.shape[0]
        h_dim = s_dim
        samples = S.shape[1]
        mixtures_power_normalized = True
        # OPTIONS FOR synaptic_lr_rule: "constant", "divide_by_log_index", "divide_by_index"
        synaptic_lr_rule = "divide_by_log_index"
        # OPTIONS FOR neural_loop_lr_rule: "constant", "divide_by_loop_index", "divide_by_slow_loop_index"
        neural_loop_lr_rule = "constant"


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
                                DScalings=LayerGains,
                                LayerMinimumGains=LayerMinimumGains,
                                LayerMaximumGains=LayerMaximumGains,
                                neural_OUTPUT_COMP_TOL=OUTPUT_COMP_TOL,
                                set_ground_truth=True,
                                S=S,
                                A=A,
                            )

        modelWSM.fit_batch_sparse(
                                    X,
                                    n_epochs=1,
                                    neural_lr_start=0.4,
                                    synaptic_lr_rule=synaptic_lr_rule,
                                    synaptic_lr_decay_divider=1,
                                    neural_loop_lr_rule=neural_loop_lr_rule,
                                    mixtures_power_normalized=mixtures_power_normalized,
                                    debug_iteration_point=debug_iteration_point,
                                    plot_in_jupyter=False,
                                )
        ######### Evaluate the Performance of WSM Framework ###########################
        SINRlistWSM = modelWSM.SIR_list
        WfWSM = modelWSM.compute_overall_mapping(return_mapping = True)
        YWSM = WfWSM @ X
        SINRWSM, SNRWSM, _, _, _ = evaluate_bss(WfWSM, YWSM, A, S, mean_normalize_estimations = False)
        
        WSM_Dict = {'SNRlevel' : SNRlevel, 'trial' : trial, 'seed' : seed_, 'Model' : 'WSM',
                    'SINR' : SINRWSM, 'SINRlist':  SINRlistWSM, 'SNR' : SNRWSM,
                    'S' : None, 'A' : None, 'X': None, 'Wf' : WfWSM, 'SNRinp' : None, 
                    'execution_time' : None}

    except Exception as e:
        print(str(e))
        WSM_Dict = {'SNRlevel' : SNRlevel, 'trial' : trial, 'seed' : seed_, 'Model' : 'WSM',
                    'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                    'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                    'execution_time' : None}

    RESULTS_DF = RESULTS_DF.append(WSM_Dict, ignore_index = True)
    RESULTS_DF.to_pickle(os.path.join("../ResultsWSM", pickle_name_for_results))

RESULTS_DF.to_pickle(os.path.join("../ResultsWSM", pickle_name_for_results))