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
from PMF import PMFv2
from WSMBSS import OnlineWSMBSS

import warnings
warnings.filterwarnings("ignore")

if not os.path.exists("../Results"):
    os.mkdir("../Results")

pickle_name_for_results = "simulation_results_sparse_noisyV1.pkl"

N = 500000 ## Number of data points
NumberofSources = 5 ## Number of source vectors
NumberofMixtures = 10 ## Number of mixture vectors

s_dim = NumberofSources
x_dim = NumberofMixtures

SNR_level_list = np.array([30, 25, 20, 15, 10])

NumAverages = 50 ## Number of realizations to average for each algorithm
seed_list = np.array([74698 * i for i in range(NumAverages)]) ## Seeds for reproducibility

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

for iter1 in range(NumAverages): ## Loop over number of averages
    seed_ = seed_list[iter1] ## Set the seed for reproducibility
    np.random.seed(seed_)
    trial = iter1
    for SNRlevel in (SNR_level_list):
        S = generate_correlated_copula_sources(
            rho=0.0,
            df=4,
            n_sources=NumberofSources,
            size_sources=N,
            decreasing_correlation=True,
        )
        S = 4 * S - 2
        S = ProjectRowstoL1NormBall(S.T).T

        # Szeromean = S - S.mean(axis = 1).reshape(-1,1)
        
        A = np.random.standard_normal(size=(NumberofMixtures, NumberofSources))
        X = np.dot(A,S)

        Xnoisy, NoisePart = addWGN(X, SNRlevel, return_noise = True)

        SNRinplevel = 10 * np.log10(np.sum(np.mean((X - NoisePart)**2, axis = 1)) / np.sum(np.mean(NoisePart**2, axis = 1)))

        #######################################################
        #        Online CorInfoMax Setup                      #
        #######################################################
        try: # Try Except for SVD did not converge error (or for any other error)
            lambday = 1 - 1e-1/10
            lambdae = 1 - 1e-1/10

            # Inverse output covariance
            By = 1 * np.eye(s_dim)
            # Inverse error covariance
            Be = 1000 * np.eye(s_dim)

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
                modelCorInfoMax.fit_batch_sparse(
                                                    X=Xnoisy,
                                                    n_epochs=1,
                                                    neural_dynamic_iterations=500,
                                                    plot_in_jupyter=False,
                                                    neural_lr_start=0.1,
                                                    neural_lr_stop=0.001,
                                                    debug_iteration_point=debug_iteration_point,
                                                    shuffle=False,
                                                )
            ######### Evaluate the Performance of CorInfoMax Framework ###########################
            SINRlistCorInfoMax = modelCorInfoMax.SIR_list
            WfCorInfoMax = modelCorInfoMax.compute_overall_mapping(return_mapping = True)
            YCorInfoMax = WfCorInfoMax @ Xnoisy
            SINRCorInfoMax, SNRCorInfoMax, _, _, _ = evaluate_bss(WfCorInfoMax, YCorInfoMax, A, S, mean_normalize_estimations = False)            

            CorInfoMax_Dict = {'SNRlevel' : SNRlevel, 'trial' : trial, 'seed' : seed_, 'Model' : 'CorInfoMax',
                               'SINR' : SINRCorInfoMax, 'SINRlist':  SINRlistCorInfoMax, 'SNR' : SNRCorInfoMax,
                               'S' : S, 'A' : A, 'X': Xnoisy, 'Wf' : WfCorInfoMax, 'SNRinp' : SNRinplevel, 
                               'execution_time' : t.interval}
        except Exception as e:
            print(str(e))
            CorInfoMax_Dict = { 'SNRlevel' : SNRlevel, 'trial' : trial, 'seed' : seed_, 'Model' : 'CorInfoMax',
                                'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                                'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                                'execution_time' : None}

        #######################################################
        #        Online WSM Setup                             #
        #######################################################
        try: # Try Except for SVD did not converge error (or for any other error)
            MUS = 0.25
            WSM_INPUT_STD = 0.5
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

            XnoisyWSM = (WSM_INPUT_STD * (Xnoisy / Xnoisy.std(1)[:,np.newaxis]))
            with Timer() as t:
                modelWSM.fit_batch_sparse(
                                            XnoisyWSM,
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
            YWSM = WfWSM @ XnoisyWSM
            SINRWSM, SNRWSM, _, _, _ = evaluate_bss(WfWSM, YWSM, A, S, mean_normalize_estimations = False)
            
            WSM_Dict = {'SNRlevel' : SNRlevel, 'trial' : trial, 'seed' : seed_, 'Model' : 'WSM',
                        'SINR' : SINRWSM, 'SINRlist':  SINRlistWSM, 'SNR' : SNRWSM,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : WfWSM, 'SNRinp' : None, 
                        'execution_time' : t.interval}

        except Exception as e:
            print(str(e))
            WSM_Dict = {'SNRlevel' : SNRlevel, 'trial' : trial, 'seed' : seed_, 'Model' : 'WSM',
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
                modelLDMI.fit_batch_sparse(
                                            Xnoisy[:,:10000],
                                            epsilon=1e-5,
                                            mu_start=100,
                                            n_iterations=10000,
                                            method="correlation",
                                            debug_iteration_point=debug_iteration_point,
                                            plot_in_jupyter=False,
                                        )
            
            ######### Evaluate the Performance of LDMIBSS Framework ###########################
            SINRlistLDMI = modelLDMI.SIR_list 
            WfLDMI = modelLDMI.W
            YLDMI = WfLDMI @ Xnoisy
            SINRLDMI, SNRLDMI, _, _, _ = evaluate_bss(WfLDMI, YLDMI, A, S, mean_normalize_estimations = False)

            LDMI_Dict = {'SNRlevel' : SNRlevel, 'trial' : trial, 'seed' : seed_, 'Model' : 'LDMI',
                         'SINR' : SINRLDMI, 'SINRlist':  SINRlistLDMI, 'SNR' : SNRLDMI,
                         'S' : None, 'A' : None, 'X': None, 'Wf' : WfLDMI, 'SNRinp' : None, 
                         'execution_time' : t.interval}
        except Exception as e:
            print(str(e))
            LDMI_Dict = {'SNRlevel' : SNRlevel, 'trial' : trial, 'seed' : seed_, 'Model' : 'LDMI',
                         'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                         'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                         'execution_time' : None}

        #######################################################
        #                 PMF BATCH                           #
        #######################################################
        try:
            modelPMF = PMFv2(s_dim = s_dim, y_dim = x_dim,
                             set_ground_truth = True, Sgt = S[:,:10000], Agt = A)
            with Timer() as t:
                modelPMF.fit_batch_sparse(
                                            Xnoisy[:,:10000],
                                            n_iterations=100000,
                                            step_size_scale=100,
                                            debug_iteration_point=debug_iteration_point,
                                            plot_in_jupyter=False,
                                        )
            ######### Evaluate the Performance of PMF Framework ###########################
            SINRlistPMF = modelPMF.SIR_list 
            WfPMF = modelPMF.W
            # YPMF = modelPMF.S
            YPMF = WfPMF @ Xnoisy
            SINRPMF, SNRPMF, _, _, _ = evaluate_bss(WfPMF, YPMF, A, S, mean_normalize_estimations = False)
            
            PMF_Dict = { 'SNRlevel' : SNRlevel, 'trial' : trial, 'seed' : seed_, 'Model' : 'PMF',
                         'SINR' : SINRPMF, 'SINRlist':  SINRlistPMF, 'SNR' : SNRPMF,
                         'S' : None, 'A' : None, 'X': None, 'Wf' : WfPMF, 'SNRinp' : None, 
                         'execution_time' : t.interval}
        except Exception as e:
            PMF_Dict = { 'SNRlevel' : SNRlevel, 'trial' : trial, 'seed' : seed_, 'Model' : 'PMF',
                         'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                         'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                         'execution_time' : t.interval}


        RESULTS_DF = RESULTS_DF.append(CorInfoMax_Dict, ignore_index = True)
        RESULTS_DF = RESULTS_DF.append(WSM_Dict, ignore_index = True)
        RESULTS_DF = RESULTS_DF.append(LDMI_Dict, ignore_index = True)
        RESULTS_DF = RESULTS_DF.append(PMF_Dict, ignore_index = True)
        RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))

RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))