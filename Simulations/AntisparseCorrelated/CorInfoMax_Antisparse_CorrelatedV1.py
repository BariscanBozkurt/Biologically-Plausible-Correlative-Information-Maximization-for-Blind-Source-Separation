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
from PMF import PMF, PMFv2
from LDMIBSS import LDMIBSS
from BCA import OnlineBCA
from BSMBSS import OnlineBSM
from ICA import fit_icainfomax
from CorInfoMaxBSS import OnlineCorInfoMax

import warnings
warnings.filterwarnings("ignore")

if not os.path.exists("../Results"):
    os.mkdir("../Results")

pickle_name_for_results = "simulation_results_correlated_antisparseV1.pkl"

N = 500000 ## Number of data points
NumberofSources = 5 ## Number of source vectors
NumberofMixtures = 10 ## Number of mixture vectors

s_dim = NumberofSources
x_dim = NumberofMixtures

rholist = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) ## Correlation parameters
SNRlevel = 30 ## Signal to noise ratio in terms of dB (for adding noise to the mixtures)

NumAverages = 250 ## Number of realizations to average for each algorithm
seed_list = np.array([431 * i for i in range(NumAverages)]) ## Seeds for reproducibility
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

for iter1 in range(NumAverages): ## Loop over number of averages
    seed_ = seed_list[iter1] ## Set the seed for reproducibility
    np.random.seed(seed_)
    trial = iter1
    for rho in (rholist):
        ###### GAUSSIAN MIXING SCENARIO AND WGN NOISE ADDING ##############################
        S = generate_correlated_copula_sources(rho = rho, df = 4, n_sources = NumberofSources, 
                                               size_sources = N , decreasing_correlation = False) ## GENERATE CORRELATED COPULA
        S = 2 * S - 1
        A = np.random.randn(NumberofMixtures,NumberofSources)
        X = np.dot(A,S)
        Xnoisy, NoisePart = addWGN(X, SNRlevel, return_noise = True) ## Add White Gaussian Noise with 30 dB SNR
        SNRinplevel = 10 * np.log10(np.sum(np.mean((Xnoisy - NoisePart) ** 2, axis = 1)) / np.sum(np.mean(NoisePart ** 2, axis = 1)))
        
        #######################################################
        #        Online CorInfoMax Setup 1                    #
        #######################################################
        try: # Try Except for SVD did not converge error (or for any other error)
            lambday = 1 - 1e-1/10
            lambdae = 1 - 1e-1/3

            # Inverse output covariance
            By = 5 * np.eye(s_dim)
            # Inverse error covariance
            Be = 10000 * np.eye(s_dim)

            modelCorInfoMax = OnlineCorInfoMax( s_dim = s_dim, x_dim = x_dim, muW = 30*1e-3, lambday = lambday,
                                                lambdae = lambdae, By = By, Be = Be, neural_OUTPUT_COMP_TOL = 1e-6,
                                                set_ground_truth = True, S = S, A = A)
            with Timer() as t:
                modelCorInfoMax.fit_batch_antisparse(   X = Xnoisy, n_epochs = 1, neural_dynamic_iterations = 500,
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
            
            CorInfoMax_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'CorInfoMax',
                               'SINR' : SINRCorInfoMax, 'SINRlist':  SINRlistCorInfoMax, 'SNR' : SNRCorInfoMax,
                               'S' : S, 'A' : A, 'X': Xnoisy, 'Wf' : WfCorInfoMax, 'SNRinp' : SNRinplevel, 
                               'execution_time' : t.interval}
        except Exception as e:
            print(str(e))
            CorInfoMax_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'CorInfoMax',
                               'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                               'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                               'execution_time' : None}

        #######################################################
        #        Online CorInfoMax Setup 2                    #
        #######################################################
        try: # Try Except for SVD did not converge error (or for any other error)
            lambday = 1 - 1e-1/10
            lambdae = 1 - 1e-1/10

            # Inverse output covariance
            By = 5 * np.eye(s_dim)
            # Inverse error covariance
            Be = 10000 * np.eye(s_dim)

            modelCorInfoMax2 = OnlineCorInfoMax( s_dim = s_dim, x_dim = x_dim, muW = 30*1e-3, lambday = lambday,
                                                lambdae = lambdae, By = By, Be = Be, neural_OUTPUT_COMP_TOL = 1e-6,
                                                set_ground_truth = True, S = S, A = A)
            with Timer() as t:
                modelCorInfoMax2.fit_batch_antisparse(  X = Xnoisy, n_epochs = 1, neural_dynamic_iterations = 500,
                                                        plot_in_jupyter = False,
                                                        neural_lr_start = 0.9,
                                                        neural_lr_stop = 0.001, 
                                                        debug_iteration_point = debug_iteration_point, 
                                                        shuffle = False)
            ######### Evaluate the Performance of CorInfoMax Framework ###########################
            SINRlistCorInfoMax2 = modelCorInfoMax2.SIR_list
            WfCorInfoMax2 = modelCorInfoMax2.compute_overall_mapping(return_mapping = True)
            YCorInfoMax2 = WfCorInfoMax2 @ Xnoisy
            SINRCorInfoMax2, SNRCorInfoMax2, _, _, _ = evaluate_bss(WfCorInfoMax2, YCorInfoMax2, A, S, mean_normalize_estimations = False)
            
            CorInfoMax_Dict2 = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'CorInfoMax2',
                               'SINR' : SINRCorInfoMax2, 'SINRlist':  SINRlistCorInfoMax2, 'SNR' : SNRCorInfoMax2,
                               'S' : None, 'A' : None, 'X': None, 'Wf' : WfCorInfoMax2, 'SNRinp' : SNRinplevel, 
                               'execution_time' : t.interval}
        except Exception as e:
            print(str(e))
            CorInfoMax_Dict2 = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'CorInfoMax2',
                               'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                               'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                               'execution_time' : None}

        #######################################################
        #        Online BCA                                   #
        #######################################################
        try: # Try Except for SVD did not converge error (or for any other error)
            # HYPERPARAMETERS OF ONLINE BCA
            lambda_ = 0.99
            mu_F = 1e-3
            beta = 30
            debug_iteration_point = 25000

            modelBCA = OnlineBCA(s_dim = s_dim, x_dim = x_dim, 
                                 lambda_ = lambda_, mu_F = mu_F, beta = beta, 
                                 set_ground_truth = True, S = S, A = A)
            with Timer() as t:
                modelBCA.fit_batch_antisparse(  Xnoisy, n_epochs = 1, neural_lr_start = 0.9, 
                                                neural_dynamic_iterations = 500, 
                                                shuffle = False, 
                                                debug_iteration_point = debug_iteration_point,
                                                plot_in_jupyter = False)

            ######### Evaluate the Performance of Online BCA Framework ###########################
            SINRlistBCA = modelBCA.SIR_list
            WfBCA = modelBCA.compute_overall_mapping(return_mapping = True)
            YBCA = WfBCA @ Xnoisy
            SINRBCA, SNRBCA, _, _, _ = evaluate_bss(WfBCA, YBCA, A, S, mean_normalize_estimations = False)
            
            BCA_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'OnlineBCA',
                        'SINR' : SINRBCA, 'SINRlist':  SINRlistBCA, 'SNR' : SNRBCA,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : WfBCA, 'SNRinp' : None, 
                        'execution_time' : t.interval}
            
        except Exception as e:
            print(str(e))
            BCA_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'OnlineBCA',
                        'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                        'execution_time' : None}

        #######################################################
        #                   BSM                               #
        #######################################################
        try: # Try Except for SVD did not converge error
            gamma = np.sqrt(1 - 4e-3)
            modelBSM = OnlineBSM(   s_dim = s_dim, x_dim = x_dim, beta = 1e-6, 
                                    gamma = gamma, whiten_input_ = True,
                                    W = np.eye(s_dim), M = np.eye(s_dim),
                                    neural_OUTPUT_COMP_TOL = 1e-7,
                                    set_ground_truth = True, S = S, A = A)
            with Timer() as t:
                modelBSM.fit_batch_antisparse(  X = Xnoisy, n_epochs = 1, neural_dynamic_iterations = 10,
                                                neural_lr_start = 0.9, neural_lr_stop = 1e-15, 
                                                fast_start = True, debug_iteration_point = debug_iteration_point,
                                                plot_in_jupyter = False)

            ######### Evaluate the Performance of NSM Framework ###########################
            SINRlistBSM = modelBSM.SIR_list 
            WfBSM = modelBSM.compute_overall_mapping(return_mapping = True)
            YBSM = WfBSM @ Xnoisy
            SINRBSM, SNRBSM, _, _, _ = evaluate_bss(WfBSM, YBSM, A, S, mean_normalize_estimations = False)
            
            BSM_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'BSM',
                        'SINR' : SINRBSM, 'SINRlist':  SINRlistBSM, 'SNR' : SNRBSM,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : WfBSM, 'SNRinp' : None, 
                        'execution_time' : t.interval}
        except Exception as e:
            print(str(e))
            BSM_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'BSM',
                        'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                        'execution_time' : None}

        #######################################################
        #                 ICA INFOMAX                         #
        #######################################################
        try:
            with Timer() as t:
                YICA = fit_icainfomax(Xnoisy, s_dim)

            ######### Evaluate the Performance of InfoMax-ICA Framework ###########################
            SINRlistICA = None 
            WfICA = YICA @ np.linalg.pinv(Xnoisy)
            SINRICA, SNRICA, _, _, _ = evaluate_bss(WfICA, YICA, A, S, mean_normalize_estimations = False)
            
            ICA_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'ICA',
                        'SINR' : SINRICA, 'SINRlist':  SINRlistICA, 'SNR' : SNRICA,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : WfICA, 'SNRinp' : None, 
                        'execution_time' : t.interval}
        except Exception as e:
            print(str(e))
            ICA_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'ICA',
                        'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                        'execution_time' : None}

        #######################################################
        #                 LDMI BATCH                          #
        #######################################################
        try:
            modelLDMI = LDMIBSS(s_dim = s_dim, x_dim = x_dim,
                                set_ground_truth = True, S = S[:,:10000], A = A)
            with Timer() as t:
                ## Feed 10000 samples of the mixtures, that is enough for LDMI
                modelLDMI.fit_batch_antisparse(  Xnoisy[:,:10000], epsilon = 1e-5, mu_start = 100, n_iterations = 10000, 
                                                 method = "correlation", debug_iteration_point = debug_iteration_point,
                                                 plot_in_jupyter = False)
            
            ######### Evaluate the Performance of LDMIBSS Framework ###########################
            SINRlistLDMI = modelLDMI.SIR_list 
            WfLDMI = modelLDMI.W
            YLDMI = WfLDMI @ Xnoisy
            SINRLDMI, SNRLDMI, _, _, _ = evaluate_bss(WfLDMI, YLDMI, A, S, mean_normalize_estimations = False)
            
            LDMI_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'LDMI',
                         'SINR' : SINRLDMI, 'SINRlist':  SINRlistLDMI, 'SNR' : SNRLDMI,
                         'S' : None, 'A' : None, 'X': None, 'Wf' : WfLDMI, 'SNRinp' : None, 
                         'execution_time' : t.interval}
        except Exception as e:
            print(str(e))
            LDMI_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'LDMI',
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
                modelPMF.fit_batch_antisparse(  Xnoisy[:,:10000], n_iterations = 100000,
                                                step_size_scale = 100,
                                                debug_iteration_point = debug_iteration_point,
                                                plot_in_jupyter = False)
            ######### Evaluate the Performance of PMF Framework ###########################
            SINRlistPMF = modelPMF.SIR_list 
            WfPMF = modelPMF.W
            # YPMF = modelPMF.S
            YPMF = WfPMF @ Xnoisy
            SINRPMF, SNRPMF, _, _, _ = evaluate_bss(WfPMF, YPMF, A, S, mean_normalize_estimations = False)
            
            PMF_Dict = { 'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'PMF',
                         'SINR' : SINRPMF, 'SINRlist':  SINRlistPMF, 'SNR' : SNRPMF,
                         'S' : None, 'A' : None, 'X': None, 'Wf' : WfPMF, 'SNRinp' : None, 
                         'execution_time' : t.interval}
        except Exception as e:
            PMF_Dict = { 'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'PMF',
                         'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                         'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                         'execution_time' : None}

        RESULTS_DF = RESULTS_DF.append(CorInfoMax_Dict, ignore_index = True)
        RESULTS_DF = RESULTS_DF.append(CorInfoMax_Dict2, ignore_index = True)
        RESULTS_DF = RESULTS_DF.append(BCA_Dict, ignore_index = True)
        RESULTS_DF = RESULTS_DF.append(BSM_Dict, ignore_index = True)
        RESULTS_DF = RESULTS_DF.append(ICA_Dict, ignore_index = True)
        RESULTS_DF = RESULTS_DF.append(LDMI_Dict, ignore_index = True)
        RESULTS_DF = RESULTS_DF.append(PMF_Dict, ignore_index = True)
        RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))

RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))