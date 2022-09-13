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
from NSMBSS import OnlineNSM
from ICA import fit_icainfomax
from CorInfoMaxBSS import OnlineCorInfoMax
from WSMBSS import OnlineWSMBSS

import warnings
warnings.filterwarnings("ignore")

if not os.path.exists("../Results"):
    os.mkdir("../Results")

pickle_name_for_results = "simulation_results_correlated_nnantisparseV5.pkl"

N = 500000 ## Number of data points
NumberofSources = 5 ## Number of source vectors
NumberofMixtures = 10 ## Number of mixture vectors

s_dim = NumberofSources
x_dim = NumberofMixtures

rholist = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) ## Correlation parameters
SNRlevel = 30 ## Signal to noise ratio in terms of dB (for adding noise to the mixtures)

NumAverages = 34 ## Number of realizations to average for each algorithm
seed_list = np.array([8877*i for i in range(25, NumAverages+26)]) ## Seeds for reproducibility
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
            Be = 2000 * np.eye(s_dim)

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
            lambdae = 1 - 1e-1/5

            # Inverse output covariance
            By = 5 * np.eye(s_dim)
            # Inverse error covariance
            Be = 2000 * np.eye(s_dim)

            modelCorInfoMax2 = OnlineCorInfoMax( s_dim = s_dim, x_dim = x_dim, muW = 30*1e-3, lambday = lambday,
                                                lambdae = lambdae, By = By, Be = Be, neural_OUTPUT_COMP_TOL = 1e-6,
                                                set_ground_truth = True, S = S, A = A)
            with Timer() as t:
                modelCorInfoMax2.fit_batch_nnantisparse( X = Xnoisy, n_epochs = 1, neural_dynamic_iterations = 500,
                                                        plot_in_jupyter = False,
                                                        neural_lr_start = 0.5,
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
                modelBCA.fit_batch_nnantisparse(Xnoisy, n_epochs = 1, neural_lr_start = 0.9, 
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
        #                   WSM                               #
        #######################################################
        try:
            WSM_INPUT_STD = 0.28
            # if rho > 0.4:
            gamma_start = 0.05
            gamma_stop = 5 * 1e-4
            # else:
            #     gamma_start = 0.1
            #     gamma_stop = 1e-3

            gammaM_start = [gamma_start, gamma_start]
            gammaM_stop = [gamma_stop, gamma_stop]
            gammaW_start = [gamma_start, gamma_start]
            gammaW_stop = [gamma_stop, gamma_stop]

            OUTPUT_COMP_TOL = 1e-7
            MAX_OUT_ITERATIONS = 3000
            LayerGains = [1, 1]
            LayerMinimumGains = np.array([1e-3, 1e-3])
            LayerMaximumGains = np.array([1e6, 20])
            WScalings = [0.0033, 0.0033]
            GamScalings = [2, 1]
            zeta = 1 * 1e-5
            beta = 0.5
            muD = np.array([1.0, 1e-2])

            s_dim = S.shape[0]
            x_dim = X.shape[0]
            h_dim = s_dim
            samples = S.shape[1]
            W_HX = np.eye(h_dim, x_dim)
            W_YH = np.eye(s_dim, h_dim)

            neural_dynamic_iterations = 500
            neural_lr_start = 0.5
            neural_lr_stop = 0.05
            synaptic_lr_rule="divide_by_log_index"
            neural_loop_lr_rule = "divide_by_slow_loop_index"
            neural_lr_decay_multiplier = 0.005
            hidden_layer_gain=100
            neural_OUTPUT_COMP_TOL = OUTPUT_COMP_TOL

            synaptic_lr_decay_divider = 1
            clip_gain_gradients=False
            gain_grads_clipping_multiplier = 1

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

            XnoisyWSM = (WSM_INPUT_STD * (Xnoisy / Xnoisy.std(1)[:,np.newaxis]))
            with Timer() as t:
                modelWSM.fit_batch_nnantisparse(
                                                XnoisyWSM,
                                                n_epochs=1,
                                                neural_dynamic_iterations=neural_dynamic_iterations,
                                                neural_lr_start=neural_lr_start,
                                                neural_lr_stop=neural_lr_stop,
                                                synaptic_lr_rule=synaptic_lr_rule,
                                                neural_loop_lr_rule=neural_loop_lr_rule,
                                                neural_fast_start = True,
                                                synaptic_lr_decay_divider=synaptic_lr_decay_divider,
                                                neural_lr_decay_multiplier=neural_lr_decay_multiplier,
                                                hidden_layer_gain=hidden_layer_gain,
                                                clip_gain_gradients=clip_gain_gradients,
                                                gain_grads_clipping_multiplier = gain_grads_clipping_multiplier,
                                                use_newton_steps_for_gains = False,
                                                shuffle=True,
                                                debug_iteration_point=debug_iteration_point,
                                                plot_in_jupyter=False,
                                            )

            ######### Evaluate the Performance of Online WSM Framework ###########################
            SINRlistWSM = modelWSM.SIR_list
            WfWSM = modelWSM.compute_overall_mapping(return_mapping = True)
            YWSM = WfWSM @ XnoisyWSM
            SINRWSM, SNRWSM, _, _, _ = evaluate_bssV2(WfWSM, YWSM, A, S, mean_normalize_estimations = False)

            WSM_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'WSM',
                        'SINR' : SINRWSM, 'SINRlist':  SINRlistWSM, 'SNR' : SNRWSM,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : WfWSM, 'SNRinp' : None, 
                        'execution_time' : t.interval}

        except Exception as e:
            print(str(e))
            WSM_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'WSM',
                        'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                        'execution_time' : None}

        #######################################################
        #                   NSM                               #
        #######################################################
        try: # Try Except for SVD did not converge error
            modelNSM = OnlineNSM(s_dim = s_dim, x_dim = x_dim, 
                                 set_ground_truth = True, S = S, A = A)
            with Timer() as t:
                modelNSM.fit_batch_nsm(X = Xnoisy, debug_iteration_point = debug_iteration_point, shuffle = False, 
                                       plot_in_jupyter = False)

            ######### Evaluate the Performance of NSM Framework ###########################
            SINRlistNSM = modelNSM.SIR_list 
            WfNSM = modelNSM.compute_overall_mapping(return_mapping = True)
            YNSM = WfNSM @ Xnoisy
            SINRNSM, SNRNSM, _, _, _ = evaluate_bss(WfNSM, YNSM, A, S, mean_normalize_estimations = False)
            
            NSM_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'NSM',
                        'SINR' : SINRNSM, 'SINRlist':  SINRlistNSM, 'SNR' : SNRNSM,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : WfNSM, 'SNRinp' : None, 
                        'execution_time' : t.interval}
        except Exception as e:
            print(str(e))
            NSM_Dict = {'rho' : rho, 'trial' : trial, 'seed' : seed_, 'Model' : 'NSM',
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
                modelLDMI.fit_batch_nnantisparse(Xnoisy[:,:10000], epsilon = 1e-5, mu_start = 100, n_iterations = 10000, 
                                                 method = "covariance", debug_iteration_point = debug_iteration_point,
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
                modelPMF.fit_batch_nnantisparse( Xnoisy[:,:10000], n_iterations = 100000,
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
                         'execution_time' : t.interval}

        RESULTS_DF = RESULTS_DF.append(CorInfoMax_Dict, ignore_index = True)
        RESULTS_DF = RESULTS_DF.append(CorInfoMax_Dict2, ignore_index = True)
        RESULTS_DF = RESULTS_DF.append(BCA_Dict, ignore_index = True)
        RESULTS_DF = RESULTS_DF.append(WSM_Dict, ignore_index = True)
        RESULTS_DF = RESULTS_DF.append(NSM_Dict, ignore_index = True)
        RESULTS_DF = RESULTS_DF.append(ICA_Dict, ignore_index = True)
        RESULTS_DF = RESULTS_DF.append(LDMI_Dict, ignore_index = True)
        RESULTS_DF = RESULTS_DF.append(PMF_Dict, ignore_index = True)
        RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))

RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))