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
from PMF import PMF
from polytope_utils import generate_practical_polytope

import warnings
warnings.filterwarnings("ignore")

if not os.path.exists("../Results"):
    os.mkdir("../Results")

pickle_name_for_results = "simulation_results_general_polytope_5dimV1.pkl"

N = 500000 ## Number of data points
NumberofSources = 5 ## Number of source vectors
NumberofMixtures = 10 ## Number of mixture vectors

s_dim = NumberofSources
x_dim = NumberofMixtures

SNRlevel = 30

NumAverages = 100 ## Number of realizations to average for each algorithm
seed_list = np.array([22223333 * i for i in range(NumAverages)]) ## Seeds for reproducibility

dim = NumberofSources

signed_dims = np.array([0, 1, 3])
nn_dims = np.array([2, 4])
sparse_dims_list = [np.array([0, 1, 4]), np.array([1, 2, 3])]
(Apoly, bpoly), Verts_poly = generate_practical_polytope(dim, signed_dims, nn_dims, sparse_dims_list)

########################################################################################
########################################################################################
###                                                                                  ###
###                        SIMULATION                                                ###
###                                                                                  ###
########################################################################################
########################################################################################

RESULTS_DF = pd.DataFrame( columns = ['trial', 'seed', 'Model', 'SINR', 'SINRlist', 'SNR', 'S', 'A', 'X', 'Wf', 'SNRinp', 'execution_time'])

####### YOU MIGHT WANT TO CHANGE THE DEBUG ITERATION POINT FOR MEMORY PURPOSES #######
debug_iteration_point = 25000 # SINR measurement per 10000 iteration

for iter1 in range(NumAverages): ## Loop over number of averages
    seed_ = seed_list[iter1] ## Set the seed for reproducibility
    np.random.seed(seed_)
    trial = iter1

    S = generate_uniform_points_in_polytope(Verts_poly, N)

    A = np.random.randn(NumberofMixtures, NumberofSources)
    X = np.dot(A, S)
    Xnoisy, NoisePart = addWGN(X, SNRlevel, return_noise = True) ## Add White Gaussian Noise with 30 dB SNR
    SNRinplevel = 10 * np.log10(np.sum(np.mean((Xnoisy - NoisePart) ** 2, axis = 1)) / np.sum(np.mean(NoisePart ** 2, axis = 1)))

    #######################################################
    #        Online CorInfoMax Setup 1                    #
    #######################################################
    try: # Try Except for SVD did not converge error (or for any other error)
        lambday = 1 - 1e-1/10
        lambdae = 1 - 1e-1/10

        # Inverse output covariance
        By = 5 * np.eye(s_dim)
        # Inverse error covariance
        Be = 1000 * np.eye(s_dim)

        modelCorInfoMax = OnlineCorInfoMax( s_dim = s_dim, x_dim = x_dim, muW = 50*1e-3, lambday = lambday,
                                            lambdae = lambdae, By = By, Be = Be, neural_OUTPUT_COMP_TOL = 1e-6,
                                            set_ground_truth = True, S = S, A = A)
        with Timer() as t:
            modelCorInfoMax.fit_batch_general_polytope(
                                                        X=Xnoisy,
                                                        signed_dims=signed_dims,
                                                        nn_dims=nn_dims,
                                                        sparse_dims_list=sparse_dims_list,
                                                        n_epochs=1,
                                                        neural_dynamic_iterations=500,
                                                        plot_in_jupyter=False,
                                                        neural_lr_start=0.1,
                                                        neural_lr_stop=1e-10,
                                                        debug_iteration_point=debug_iteration_point,
                                                        shuffle=False,
                                                    )
        ######### Evaluate the Performance of CorInfoMax Framework ###########################
        SINRlistCorInfoMax = modelCorInfoMax.SIR_list
        WfCorInfoMax = modelCorInfoMax.compute_overall_mapping(return_mapping = True)
        YCorInfoMax = WfCorInfoMax @ Xnoisy
        SINRCorInfoMax, SNRCorInfoMax, _, _, _ = evaluate_bss(WfCorInfoMax, YCorInfoMax, A, S, mean_normalize_estimations = False)
        
        CorInfoMax_Dict = {'trial' : trial, 'seed' : seed_, 'Model' : 'CorInfoMax',
                            'SINR' : SINRCorInfoMax, 'SINRlist':  SINRlistCorInfoMax, 'SNR' : SNRCorInfoMax,
                            'S' : S, 'A' : A, 'X': Xnoisy, 'Wf' : WfCorInfoMax, 'SNRinp' : SNRinplevel, 
                            'execution_time' : t.interval}
    except Exception as e:
        print(str(e))
        CorInfoMax_Dict = {'trial' : trial, 'seed' : seed_, 'Model' : 'CorInfoMax',
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
        Be = 2500 * np.eye(s_dim)

        modelCorInfoMax2 = OnlineCorInfoMax( s_dim = s_dim, x_dim = x_dim, muW = 50*1e-3, lambday = lambday,
                                            lambdae = lambdae, By = By, Be = Be, neural_OUTPUT_COMP_TOL = 1e-6,
                                            set_ground_truth = True, S = S, A = A)
        with Timer() as t:
            modelCorInfoMax2.fit_batch_general_polytope(
                                                        X=Xnoisy,
                                                        signed_dims=signed_dims,
                                                        nn_dims=nn_dims,
                                                        sparse_dims_list=sparse_dims_list,
                                                        n_epochs=1,
                                                        neural_dynamic_iterations=500,
                                                        plot_in_jupyter=False,
                                                        neural_lr_start=0.1,
                                                        neural_lr_stop=1e-10,
                                                        debug_iteration_point=debug_iteration_point,
                                                        shuffle=False,
                                                    )
        ######### Evaluate the Performance of CorInfoMax Framework ###########################
        SINRlistCorInfoMax2 = modelCorInfoMax2.SIR_list
        WfCorInfoMax2 = modelCorInfoMax2.compute_overall_mapping(return_mapping = True)
        YCorInfoMax2 = WfCorInfoMax2 @ Xnoisy
        SINRCorInfoMax2, SNRCorInfoMax2, _, _, _ = evaluate_bss(WfCorInfoMax2, YCorInfoMax2, A, S, mean_normalize_estimations = False)
        
        CorInfoMax_Dict2 = {'trial' : trial, 'seed' : seed_, 'Model' : 'CorInfoMax2',
                            'SINR' : SINRCorInfoMax2, 'SINRlist':  SINRlistCorInfoMax2, 'SNR' : SNRCorInfoMax2,
                            'S' : None, 'A' : None, 'X': None, 'Wf' : WfCorInfoMax2, 'SNRinp' : None, 
                            'execution_time' : t.interval}
    except Exception as e:
        print(str(e))
        CorInfoMax_Dict2 = {'trial' : trial, 'seed' : seed_, 'Model' : 'CorInfoMax2',
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
            modelLDMI.fit_batch_general_polytope(
                                                    Xnoisy[:,:10000],
                                                    signed_dims,
                                                    nn_dims,
                                                    sparse_dims_list,
                                                    epsilon=1e-5,
                                                    mu_start=200,
                                                    n_iterations=10000,
                                                    method="correlation",
                                                    lr_rule="inv_sqrt",
                                                    debug_iteration_point=debug_iteration_point,
                                                    plot_in_jupyter=False,
                                                )
        ######### Evaluate the Performance of LDMIBSS Framework ###########################
        SINRlistLDMI = modelLDMI.SIR_list 
        WfLDMI = modelLDMI.W
        YLDMI = WfLDMI @ Xnoisy
        SINRLDMI, SNRLDMI, _, _, _ = evaluate_bss(WfLDMI, YLDMI, A, S, mean_normalize_estimations = False)
        
        LDMI_Dict = { 'trial' : trial, 'seed' : seed_, 'Model' : 'LDMI',
                        'SINR' : SINRLDMI, 'SINRlist':  SINRlistLDMI, 'SNR' : SNRLDMI,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : WfLDMI, 'SNRinp' : None, 
                        'execution_time' : t.interval}
    except Exception as e:
        print(str(e))
        LDMI_Dict = { 'trial' : trial, 'seed' : seed_, 'Model' : 'LDMI',
                        'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                        'execution_time' : None}

    #######################################################
    #                 PMF BATCH                           #
    #######################################################
    try:
        modelPMF = PMF(s_dim = s_dim, y_dim = x_dim,
                       set_ground_truth = True, Sgt = S[:,:10000], Agt = A)
        with Timer() as t:
            modelPMF.fit_batch_general_polytope(
                                                Xnoisy[:,:10000],
                                                n_iterations=25000,
                                                signed_dims=signed_dims,
                                                nn_dims=nn_dims,
                                                sparse_dims_list=sparse_dims_list,
                                                Lt=250,
                                                lambda_=28,
                                                tau=1,
                                                debug_iteration_point=debug_iteration_point,
                                                plot_in_jupyter=False,
                                            )
        ######### Evaluate the Performance of PMF Framework ###########################
        SINRlistPMF = modelPMF.SIR_list 
        WfPMF = modelPMF.W
        # YPMF = modelPMF.S
        YPMF = WfPMF @ Xnoisy
        SINRPMF, SNRPMF, _, _, _ = evaluate_bss(WfPMF, YPMF, A, S, mean_normalize_estimations = False)
        
        PMF_Dict = {'trial' : trial, 'seed' : seed_, 'Model' : 'PMF',
                        'SINR' : SINRPMF, 'SINRlist':  SINRlistPMF, 'SNR' : SNRPMF,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : WfPMF, 'SNRinp' : None, 
                        'execution_time' : t.interval}
    except Exception as e:
        PMF_Dict = { 'trial' : trial, 'seed' : seed_, 'Model' : 'PMF',
                        'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                        'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                        'execution_time' : None}

    RESULTS_DF = RESULTS_DF.append(CorInfoMax_Dict, ignore_index = True)
    RESULTS_DF = RESULTS_DF.append(CorInfoMax_Dict2, ignore_index = True)
    RESULTS_DF = RESULTS_DF.append(LDMI_Dict, ignore_index = True)
    RESULTS_DF = RESULTS_DF.append(PMF_Dict, ignore_index = True)
    RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))

RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))