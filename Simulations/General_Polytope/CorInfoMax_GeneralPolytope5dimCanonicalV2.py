########## IMPORT REQUIIRED LIBRARIES ##########
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from numba import njit
import pypoman
########## IMPORT UTILITY SCRIPTS ##############
import sys
sys.path.insert(0, '../../src')
from general_utils import *
from dsp_utils import *
from bss_utils import *
from numba_utils import *
######## IMPORT THE REQUIRED ALGORITHMS ########
from CorInfoMaxBSS import OnlineCorInfoMax, OnlineCorInfoMaxCanonical

from polytope_utils import generate_practical_polytope

import warnings
warnings.filterwarnings("ignore")

if not os.path.exists("../Results"):
    os.mkdir("../Results")

pickle_name_for_results = "simulation_results_general_polytope_5dim_CanonicalV2.pkl"

N = 500000 ## Number of data points
NumberofSources = 5 ## Number of source vectors
NumberofMixtures = 10 ## Number of mixture vectors

s_dim = NumberofSources
x_dim = NumberofMixtures

SNRlevel = 40

NumAverages = 100 ## Number of realizations to average for each algorithm
seed_list = np.array([21123333 * i for i in range(NumAverages)]) ## Seeds for reproducibility

dim = NumberofSources

signed_dims = np.array([0, 1, 3])
nn_dims = np.array([2, 4])
sparse_dims_list = [np.array([0, 1, 4]), np.array([1, 2, 3])]
(Apoly, bpoly), Verts_poly = generate_practical_polytope(dim, signed_dims, nn_dims, sparse_dims_list)
Apoly, bpoly = pypoman.compute_polytope_halfspaces(Verts_poly.T)
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
    #        Online CorInfoMax Setup Canonical            #
    #######################################################
    try: # Try Except for SVD did not converge error (or for any other error)
        lambday = 1 - 1e-1 / 10
        lambdae = 1 - 1e-1 / 10
        s_dim = S.shape[0]
        x_dim = X.shape[0]

        # Inverse output covariance
        By = 5 * np.eye(s_dim)
        # Inverse error covariance
        Be = 500 * np.eye(s_dim)

        modelCorInfoMaxCanonical = OnlineCorInfoMaxCanonical(
                                                                s_dim=s_dim,
                                                                x_dim=x_dim,
                                                                muW=50 * 1e-3,
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
            modelCorInfoMaxCanonical.fit_batch(
                                                    X=Xnoisy,
                                                    Apoly = Apoly,
                                                    bpoly = bpoly,
                                                    n_epochs=1,
                                                    neural_dynamic_iterations=500,
                                                    plot_in_jupyter=False,
                                                    neural_lr_start=0.9, 
                                                    neural_lr_stop=1e-4,
                                                    lagrangian_lambd_lr = 0.5,
                                                    debug_iteration_point=debug_iteration_point,
                                                    shuffle=False,
                                                )
        ######### Evaluate the Performance of CorInfoMax Framework ###########################
        SINRlistCorInfoMaxCanonical = modelCorInfoMaxCanonical.SIR_list
        WfCorInfoMaxCanonical = modelCorInfoMaxCanonical.compute_overall_mapping(return_mapping = True)
        YCorInfoMaxCanonical = WfCorInfoMaxCanonical @ Xnoisy
        SINRCorInfoMaxCanonical, SNRCorInfoMaxCanonical, _, _, _ = evaluate_bss(WfCorInfoMaxCanonical, YCorInfoMaxCanonical, A, S, mean_normalize_estimations = False)
        
        CorInfoMax_Dict_Canonical = {   'trial' : trial, 'seed' : seed_, 'Model' : 'CorInfoMaxCanonical',
                                        'SINR' : SINRCorInfoMaxCanonical, 'SINRlist':  SINRlistCorInfoMaxCanonical, 
                                        'SNR' : SNRCorInfoMaxCanonical,
                                        'S' : None, 'A' : None, 'X': None, 'Wf' : WfCorInfoMaxCanonical, 'SNRinp' : SNRinplevel, 
                                        'execution_time' : t.interval}
    except Exception as e:
        print(str(e))
        CorInfoMax_Dict_Canonical = {   'trial' : trial, 'seed' : seed_, 'Model' : 'CorInfoMaxCanonical',
                                        'SINR' : -999, 'SINRlist':  str(e), 'SNR' : None,
                                        'S' : None, 'A' : None, 'X': None, 'Wf' : None, 'SNRinp' : None, 
                                        'execution_time' : None}

    RESULTS_DF = RESULTS_DF.append(CorInfoMax_Dict, ignore_index = True)
    RESULTS_DF = RESULTS_DF.append(CorInfoMax_Dict_Canonical, ignore_index = True)

    RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))

RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))