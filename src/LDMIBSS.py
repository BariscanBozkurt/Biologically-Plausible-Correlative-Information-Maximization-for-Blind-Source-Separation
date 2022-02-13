"""
Title: LDMIBSS.py

Two Layer Recurrent Neural Network for Blind Source Separation

Code Writer: Bariscan Bozkurt (Koç University - EEE & Mathematics)

Date: 17.02.2022
"""

import numpy as np
import scipy
from scipy.stats import invgamma, chi2, t
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
import matplotlib as mpl
import math
import pylab as pl
from numba import njit, jit
from tqdm import tqdm
from IPython.display import display, Latex, Math, clear_output
from IPython import display as display1
import warnings
warnings.filterwarnings("ignore")

############# Log-Det Mutual Information Based Blind Source Separation Neural Network ####################################
class OnlineLDMIBSS:

    """
    Implementation of online Log-Det Mutual Information Based Blind Source Separation Framework
    Parameters:
    =================================
    s_dim          -- Dimension of the sources
    x_dim          -- Dimension of the mixtures
    W              -- Feedforward Synapses
    By             -- Inverse Output Covariance
    Be             -- Inverse Error Covariance
    lambday        -- Ry forgetting factor
    lambdae        -- Re forgetting factor

    
    Methods:
    ==================================
    run_neural_dynamics_antisparse
    fit_batch_antisparse
    fit_batch_nnantisparse

    """
    
    def __init__(self, s_dim, x_dim, lambday = 0.999, lambdae = 1, muW = 1e-3, beta = 5, W = None, By = None, Be = None, neural_OUTPUT_COMP_TOL = 1e-6, set_ground_truth = False, S = None, A = None):
        if W is not None:
            assert W.shape == (s_dim, x_dim), "The shape of the initial guess W must be (s_dim, x_dim) = (%d,%d)" % (s_dim, x_dim)
            W = W
        else:
            W = np.eye(s_dim, x_dim)
            
        if By is not None:
            assert By.shape == (s_dim,s_dim), "The shape of the initial guess By must be (s_dim, s_dim) = (%d,%d)" % (s_dim,s_dim)
            By = By
        else:
            By = 5*np.eye(s_dim)

        if Be is not None:
            assert Be.shape == (s_dim,s_dim), "The shape of the initial guess Be must be (s_dim, s_dim) = (%d,%d)" % (s_dim,s_dim)
            Be = Be
        else:
            Be = 1*np.eye(s_dim)
            
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.lambday = lambday
        self.lambdae = lambdae
        self.beta = beta
        self.muW = muW
        self.gamy = (1-lambday) / lambday
        self.game = (1 - lambdae) / lambdae
        self.W = W
        self.By = By
        self.Be = Be
        self.neural_OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL
        ### Ground Truth Sources and Mixing Matrix For Debugging
        self.set_ground_truth = set_ground_truth
        self.S = S # Sources
        self.A = A # Mixing Matrix
        self.SIR_list = []
        self.SNR_list = []

    # Calculate SIR Function
    def CalculateSIR(self, H,pH, return_db = True):
        G=pH@H
        Gmax=np.diag(np.max(abs(G),axis=1))
        P=1.0*((np.linalg.inv((Gmax))@np.abs(G))>0.99)
        T=G@P.T
        rankP=np.linalg.matrix_rank(P)
        diagT = np.diag(T)
        # Signal Power
        sigpow = np.linalg.norm(diagT,2)**2
        # Interference Power
        intpow = np.linalg.norm(T,'fro')**2 - sigpow
        SIRV = sigpow/intpow
        # SIRV=np.linalg.norm((np.diag(T)))**2/(np.linalg.norm(T,'fro')**2-np.linalg.norm(np.diag(T))**2)
        if return_db:
            SIRV = 10*np.log10(sigpow/intpow)

        return SIRV,rankP

    def compute_overall_mapping(self, return_mapping = False):
        W, By, Be, gamy, game, beta = self.W, self.By, self.Be, self.gamy, self.game, self.beta
        # Wf = np.linalg.pinv((gamy/beta) * By - np.eye(self.s_dim)) @ W
        Wf = np.linalg.pinv(gamy * By - game * Be - beta * np.eye(self.s_dim)) @ (game * Be @ W + beta * W)
        if return_mapping:
            return Wf
        else:
            return None

    def whiten_signal(self, X, mean_normalize = True, type_ = 3):
        """
        Input : X  ---> Input signal to be whitened
        type_ : Defines the type for preprocesing matrix. type_ = 1 and 2 uses eigenvalue decomposition whereas type_ = 3 uses SVD.
        Output: X_white  ---> Whitened signal, i.e., X_white = W_pre @ X where W_pre = (R_x^0.5)^+ (square root of sample correlation matrix)
        """
        if mean_normalize:
            X = X - np.mean(X,axis = 0, keepdims = True)

        cov = np.cov(X.T)

        if type_ == 3: # Whitening using singular value decomposition
            U,S,V = np.linalg.svd(cov)
            d = np.diag(1.0 / np.sqrt(S))
            W_pre = np.dot(U, np.dot(d, U.T))

        else: # Whitening using eigenvalue decomposition
            d,S = np.linalg.eigh(cov)
            D = np.diag(d)

            D_sqrt = np.sqrt(D * (D>0))

            if type_ == 1: # Type defines how you want W_pre matrix to be
                W_pre = np.linalg.pinv(S@D_sqrt)
            elif type_ == 2:
                W_pre = np.linalg.pinv(S@D_sqrt@S.T)

        X_white = (W_pre @ X.T).T

        return X_white, W_pre
    
    def ProjectOntoLInfty(self, X):
        
        return X*(X>=-1.0)*(X<=1.0)+(X>1.0)*1.0-1.0*(X<-1.0)
    
    @staticmethod
    @njit
    def run_neural_dynamics_antisparse(x, y, W, My, Be, beta, gamy, game, lr_start = 0.9, lr_stop = 1e-15, neural_dynamic_iterations = 100, neural_OUTPUT_COMP_TOL = 1e-7):
        def ProjectOntoLInfty(X, thresh = 1.0):
            return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)

        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            mu_y = max(lr_start / (j + 1), lr_stop)
            # mu_y = lr_start / (j + 1)
            y_old = y.copy()
            e = yke - y
            grady = -y + gamy * My @ y + game * Be @ e + beta * e
            y = y + mu_y * (grady)
            y = ProjectOntoLInfty(y)

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    @staticmethod
    @njit
    def run_neural_dynamics_nnantisparse(x, y, W, My, Be, beta, gamy, game, lr_start = 0.9, lr_stop = 1e-15, neural_dynamic_iterations = 100, neural_OUTPUT_COMP_TOL = 1e-7):
        def ProjectOntoNNLInfty(X, thresh = 1.0):
            return X*(X>=0.0)*(X<=thresh)+(X>thresh)*thresh #-thresh*(X<-thresh)

        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            mu_y = max(lr_start / (j + 1), lr_stop)
            # mu_y = lr_start / (j + 1)
            y_old = y.copy()
            e = yke - y
            grady = -y + gamy * My @ y + game * Be @ e + beta * e
            y = y + mu_y * (grady)
            y = ProjectOntoNNLInfty(y)

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y
        
    @staticmethod
    @njit
    def run_neural_dynamics_sparse(x, y, W, My, Be, beta, gamy, game, lr_start = 0.9, lr_stop = 1e-15, neural_dynamic_iterations = 100, neural_OUTPUT_COMP_TOL = 1e-7):
        
        STLAMBD = 0
        dval = 0
        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            mu_y = max(lr_start / (j + 1), lr_stop)
            # mu_y = lr_start / (j + 1)
            y_old = y.copy()
            e = yke - y
            grady = -y + gamy * My @ y + game * Be @ e + beta * e
            y = y + mu_y * (grady)

            # SOFT THRESHOLDING
            y_absolute = np.abs(y)
            y_sign = np.sign(y)

            y = (y_absolute > STLAMBD) * (y_absolute - STLAMBD) * y_sign
            dval = np.linalg.norm(y, 1) - 1
            STLAMBD = max(STLAMBD + 1 * dval, 0)

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    @staticmethod
    @njit
    def run_neural_dynamics_nnsparse(x, y, W, My, Be, beta, gamy, game, lr_start = 0.9, lr_stop = 1e-15, neural_dynamic_iterations = 100, neural_OUTPUT_COMP_TOL = 1e-7):
        
        STLAMBD = 0
        dval = 0
        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            mu_y = max(lr_start / (j + 1), lr_stop)
            # mu_y = lr_start / (j + 1)
            y_old = y.copy()
            e = yke - y
            grady = -y + gamy * My @ y + game * Be @ e + beta * e
            y = y + mu_y * (grady)

            y = np.maximum(y - STLAMBD, 0)

            dval = np.sum(y) - 1
            STLAMBD = max(STLAMBD + 1 * dval, 0)

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    def fit_next_antisparse(self,x_current, neural_dynamic_iterations = 250, lr_start = 0.9):
        
        lambday, lambdae, beta, muW, gamy, game, W, By, Be = self.lambday, self.lambdae, self.beta, self.muW, self.gamy, self.game, self.W, self.By, self.Be
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        h = 1 / gamy # Hopefield parameter

        y = np.zeros(self.s_dim)

        # Output recurrent weights
        My = By + h * np.eye(self.s_dim)
        
        y = self.run_neural_dynamics_antisparse(x_current, y, W, My, Be, beta, gamy, game, 
                                                lr_start = lr_start, neural_dynamic_iterations = neural_dynamic_iterations, 
                                                neural_OUTPUT_COMP_TOL = neural_dynamic_tol)
        
        e = y - W @ x_current

        W = W + muW * beta * np.outer(e, x_current)

        By = (1/lambday) * (By - gamy * np.outer(By @ y, By @ y))        
        
        ee = np.dot(Be,e)
        Be = 1 / lambdae * (Be - game * np.dot(ee, ee.T))

        self.W = W
        self.By = By
        self.Be = self.Be
        
    def fit_batch_antisparse(self, X, n_epochs = 1, neural_dynamic_iterations = 250, neural_lr_start = 0.9, neural_lr_stop = 1e-3, whiten = False, whiten_type = 2, shuffle = False, verbose = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        
        lambday, lambdae, beta, muW, gamy, game, W, By, Be = self.lambday, self.lambdae, self.beta, self.muW, self.gamy, self.game, self.W, self.By, self.Be
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        h = 1 / gamy # Hopefield parameter

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            S = self.S
            A = self.A

        # Y = np.zeros((self.s_dim, samples))
        Y = np.random.randn(self.s_dim, samples)
        
        
        if shuffle:
            idx = np.random.permutation(samples) # random permutation
        else:
            idx = np.arange(samples)
            
        if whiten:
            X_white, W_pre = self.whiten_signal(X.T, type_ = whiten_type)
            X_white = X_white.T
            A = W_pre @ A
            self.A = A
        else:
            X_white = X 
            
            
        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X_white[:,idx[i_sample]]
                y = np.zeros(self.s_dim)

                # Output recurrent weights
                My = By + h * np.eye(self.s_dim)
                y = self.run_neural_dynamics_antisparse(x_current, y, W, My, Be, beta, gamy, game, 
                                                        lr_start = neural_lr_start, neural_dynamic_iterations = neural_dynamic_iterations, 
                                                        neural_OUTPUT_COMP_TOL = neural_dynamic_tol)
                        
                e = y - W @ x_current

                W = W + muW * beta * np.outer(e, x_current)
                
                z = By @ y
                By = (1/lambday) * (By - gamy * np.outer(z, z))

                ee = np.dot(Be,e)
                Be = 1 / lambdae * (Be - game * np.dot(ee, ee.T))
                # Record the seperated signal
                Y[:,idx[i_sample]] = y

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.By = By
                        self.Be = Be
                        Wf = self.compute_overall_mapping(return_mapping = True)
                        SIR = self.CalculateSIR(A, Wf)[0]
                        SIRlist.append(SIR)
                        self.SIR_list = SIRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth = 3)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
                            pl.ylabel("SIR (dB)", fontsize = 15)
                            pl.title("SIR Behaviour", fontsize = 15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())   
        self.W = W
        self.By = By
        self.Be = Be
        
    def fit_batch_nnantisparse(self, X, n_epochs = 1, neural_dynamic_iterations = 250, neural_lr_start = 0.9, neural_lr_stop = 1e-3, whiten = False, whiten_type = 2, shuffle = False, verbose = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        
        lambday, lambdae, beta, muW, gamy, game, W, By, Be = self.lambday, self.lambdae, self.beta, self.muW, self.gamy, self.game, self.W, self.By, self.Be
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        h = 1 / gamy # Hopefield parameter

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            S = self.S
            A = self.A

        # Y = np.zeros((self.s_dim, samples))
        Y = np.random.randn(self.s_dim, samples)
        
        
        if shuffle:
            idx = np.random.permutation(samples) # random permutation
        else:
            idx = np.arange(samples)
            
        if whiten:
            X_white, W_pre = self.whiten_signal(X.T, type_ = whiten_type)
            X_white = X_white.T
            A = W_pre @ A
            self.A = A
        else:
            X_white = X 
            
            
        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X_white[:,idx[i_sample]]
                y = np.zeros(self.s_dim)
                # y = np.random.uniform(0,1, size = (self.s_dim,))
                # Output recurrent weights
                My = By + h * np.eye(self.s_dim)
                y = self.run_neural_dynamics_nnantisparse(x_current, y, W, My, Be, beta, gamy, game, 
                                                             neural_lr_start, neural_lr_stop, neural_dynamic_iterations, 
                                                             neural_dynamic_tol)
                        
                e = y - W @ x_current

                W = (1 - 1e-6) * W + muW * beta * np.outer(e, x_current)
                
                z = By @ y
                By = (1/lambday) * (By - gamy * np.outer(z, z))

                ee = np.dot(Be,e)
                Be = 1 / lambdae * (Be - game * np.dot(ee, ee.T))
                # Record the seperated signal
                Y[:,idx[i_sample]] = y

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.By = By
                        self.Be = Be
                        Wf = self.compute_overall_mapping(return_mapping = True)
                        SIR = self.CalculateSIR(A, Wf)[0]
                        SIRlist.append(SIR)
                        self.SIR_list = SIRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth = 3)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
                            pl.ylabel("SIR (dB)", fontsize = 15)
                            pl.title("SIR Behaviour", fontsize = 15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())   
        self.W = W
        self.By = By
        self.Be = Be

    def fit_batch_sparse(self, X, n_epochs = 1, neural_dynamic_iterations = 250, neural_lr_start = 0.9, neural_lr_stop = 1e-3, whiten = False, whiten_type = 2, shuffle = False, verbose = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        
        lambday, lambdae, beta, muW, gamy, game, W, By, Be = self.lambday, self.lambdae, self.beta, self.muW, self.gamy, self.game, self.W, self.By, self.Be
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        h = 1 / gamy # Hopefield parameter

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            S = self.S
            A = self.A

        # Y = np.zeros((self.s_dim, samples))
        Y = np.random.randn(self.s_dim, samples)
        
        
        if shuffle:
            idx = np.random.permutation(samples) # random permutation
        else:
            idx = np.arange(samples)
            
        if whiten:
            X_white, W_pre = self.whiten_signal(X.T, type_ = whiten_type)
            X_white = X_white.T
            A = W_pre @ A
            self.A = A
        else:
            X_white = X 
            
            
        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X_white[:,idx[i_sample]]
                y = np.zeros(self.s_dim)
                # y = np.random.uniform(0,1, size = (self.s_dim,))
                # Output recurrent weights
                My = By + h * np.eye(self.s_dim)
                y = self.run_neural_dynamics_sparse(x_current, y, W, My, Be, beta, gamy, game, 
                                                    neural_lr_start, neural_lr_stop, neural_dynamic_iterations, 
                                                    neural_dynamic_tol)
                        
                e = y - W @ x_current

                W = (1 - 1e-6) * W + muW * beta * np.outer(e, x_current)
                
                z = By @ y
                By = (1/lambday) * (By - gamy * np.outer(z, z))

                ee = np.dot(Be,e)
                Be = 1 / lambdae * (Be - game * np.dot(ee, ee.T))
                # Record the seperated signal
                Y[:,idx[i_sample]] = y

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.By = By
                        self.Be = Be
                        Wf = self.compute_overall_mapping(return_mapping = True)
                        SIR = self.CalculateSIR(A, Wf)[0]
                        SIRlist.append(SIR)
                        self.SIR_list = SIRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth = 3)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
                            pl.ylabel("SIR (dB)", fontsize = 15)
                            pl.title("SIR Behaviour", fontsize = 15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())   
        self.W = W
        self.By = By
        self.Be = Be

    def fit_batch_nnsparse(self, X, n_epochs = 1, neural_dynamic_iterations = 250, neural_lr_start = 0.9, neural_lr_stop = 1e-3, whiten = False, whiten_type = 2, shuffle = False, verbose = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        
        lambday, lambdae, beta, muW, gamy, game, W, By, Be = self.lambday, self.lambdae, self.beta, self.muW, self.gamy, self.game, self.W, self.By, self.Be
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        h = 1 / gamy # Hopefield parameter

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            S = self.S
            A = self.A

        # Y = np.zeros((self.s_dim, samples))
        Y = np.random.randn(self.s_dim, samples)
        
        
        if shuffle:
            idx = np.random.permutation(samples) # random permutation
        else:
            idx = np.arange(samples)
            
        if whiten:
            X_white, W_pre = self.whiten_signal(X.T, type_ = whiten_type)
            X_white = X_white.T
            A = W_pre @ A
            self.A = A
        else:
            X_white = X 
            
            
        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X_white[:,idx[i_sample]]
                y = np.zeros(self.s_dim)
                # y = np.random.uniform(0,1, size = (self.s_dim,))
                # Output recurrent weights
                My = By + h * np.eye(self.s_dim)
                y = self.run_neural_dynamics_nnsparse(x_current, y, W, My, Be, beta, gamy, game, 
                                                    neural_lr_start, neural_lr_stop, neural_dynamic_iterations, 
                                                    neural_dynamic_tol)
                        
                e = y - W @ x_current

                W = (1 - 1e-6) * W + muW * beta * np.outer(e, x_current)
                
                z = By @ y
                By = (1/lambday) * (By - gamy * np.outer(z, z))

                ee = np.dot(Be,e)
                Be = 1 / lambdae * (Be - game * np.dot(ee, ee.T))
                # Record the seperated signal
                Y[:,idx[i_sample]] = y

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.By = By
                        self.Be = Be
                        Wf = self.compute_overall_mapping(return_mapping = True)
                        SIR = self.CalculateSIR(A, Wf)[0]
                        SIRlist.append(SIR)
                        self.SIR_list = SIRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth = 3)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
                            pl.ylabel("SIR (dB)", fontsize = 15)
                            pl.title("SIR Behaviour", fontsize = 15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())   
        self.W = W
        self.By = By
        self.Be = Be

class OnlineNSM:
    """
    Implementation of Online Nonnegative Similarity Matching.

    Parameters:
    ==========================================================
    s_dim           --Dimension of the sources
    x_dim           --Dimension of the mixtures
    W1
    W2
    Dt
    neural_OUTPUT_COMP_TOL
    set_ground_truth
    S               --Original Sources (for debugging only)
    A               --Mixing Matrix (for debugging only)

    Methods:
    ===========================================================
    whiten_input
    snr
    ZeroOneNormalizeData
    ZeroOneNormalizeColumns
    compute_overall_mapping
    CalculateSIR
    predict
    run_neural_dynamics
    fit_batch_nsm
    """
    def __init__(self, s_dim, x_dim, W1 = None, W2 = None, Dt = None, whiten_input_ = True, set_ground_truth = False, S = None, A = None):

        if W1 is not None:
            if whiten_input_:
                assert W1.shape == (s_dim, s_dim), "The shape of the initial guess W1 must be (s_dim, s_dim) = (%d, %d)" % (s_dim, s_dim)
                W1 = W1
            else:
                assert W1.shape == (s_dim, x_dim), "The shape of the initial guess W1 must be (s_dim, x_dim) = (%d, %d)" % (s_dim, x_dim)
                W1 = W1
        else:
            if whiten_input_:
                W1 = np.eye(s_dim, s_dim)
            else:
                W1 = np.eye(s_dim, x_dim)

        if W2 is not None:
            assert W2.shape == (s_dim, s_dim), "The shape of the initial guess W2 must be (s_dim, s_dim) = (%d, %d)" % (s_dim, s_dim)
            W2 = W2
        else:
            W2 = np.zeros((s_dim, s_dim))

        if Dt is not None:
            assert Dt.shape == (s_dim, 1), "The shape of the initial guess Dt must be (s_dim, 1) = (%d, %d)" % (s_dim, 1)
            Dt = Dt
        else:
            Dt = 0.1 * np.ones((s_dim, 1))

        
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.W1 = W1
        self.W2 = W2
        self.Dt = Dt
        self.whiten_input_ = whiten_input_
        self.Wpre = np.eye(x_dim)
        self.set_ground_truth = set_ground_truth
        self.S = S
        self.A = A
        self.SIRlist = []

    def whiten_input(self, X):
        x_dim = self.x_dim
        s_dim = self.s_dim
        N = X.shape[1]
        # Mean of the mixtures
        mX = np.mean(X, axis = 1).reshape((x_dim, 1))
        # Covariance of Mixtures
        Rxx = np.dot(X, X.T)/N - np.dot(mX, mX.T)
        # Eigenvalue Decomposition
        d, V = np.linalg.eig(Rxx)
        D = np.diag(d)
        # Sorting indexis for eigenvalues from large to small
        ie = np.argsort(-d)
        # Inverse square root of eigenvalues
        ddinv = 1/np.sqrt(d[ie[:s_dim]])
        # Pre-whitening matrix
        Wpre = np.dot(np.diag(ddinv), V[:, ie[:s_dim]].T)#*np.sqrt(12)
        # Whitened mixtures
        H = np.dot(Wpre, X)
        self.Wpre = Wpre
        return H


    def snr(self, S_original, S_noisy):
        N_hat = S_original - S_noisy
        N_P = (N_hat ** 2).sum(axis = 0)
        S_P = (S_original ** 2).sum(axis = 0)
        snr = 10 * np.log10(S_P / N_P)
        return snr

    def ZeroOneNormalizeData(self,data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def ZeroOneNormalizeColumns(self,X):
        X_normalized = np.empty_like(X)
        for i in range(X.shape[1]):
            X_normalized[:,i] = self.ZeroOneNormalizeData(X[:,i])

        return X_normalized

    def compute_overall_mapping(self, return_mapping = False):
        W1, W2 = self.W1, self.W2
        Wpre = self.Wpre
        W = np.linalg.pinv(np.eye(self.s_dim) + W2) @ W1 @ Wpre
        self.W = W
        if return_mapping:
            return W

    # Calculate SIR Function
    def CalculateSIR(self, H,pH, return_db = True):
        G=pH@H
        Gmax=np.diag(np.max(abs(G),axis=1))
        P=1.0*((np.linalg.inv((Gmax))@np.abs(G))>0.99)
        T=G@P.T
        rankP=np.linalg.matrix_rank(P)
        diagT = np.diag(T)
        # Signal Power
        sigpow = np.linalg.norm(diagT,2)**2
        # Interference Power
        intpow = np.linalg.norm(T,'fro')**2 - sigpow
        SIRV = sigpow/intpow
        # SIRV=np.linalg.norm((np.diag(T)))**2/(np.linalg.norm(T,'fro')**2-np.linalg.norm(np.diag(T))**2)
        if return_db:
            SIRV = 10*np.log10(sigpow/intpow)

        return SIRV,rankP

    def predict(self, X):
        Wf = self.compute_overall_mapping(return_mapping = True)
        return Wf @ X

    @staticmethod
    @njit
    def run_neural_dynamics(x, y, W1, W2, n_iterations = 200):
        for j in range(n_iterations):
            ind = math.floor((np.random.rand(1) * y.shape[0])[0])         
            y[ind, :] = np.maximum(np.dot(W1[ind, :], x) - np.dot(W2[ind, :], y), 0)

        return y

    def fit_batch_nsm(self, X, n_epochs = 1, neural_dynamic_iterations = 250, shuffle = True, debug_iteration_point = 100, plot_in_jupyter = False):
        s_dim, x_dim = self.s_dim, self.x_dim
        W1, W2, Dt = self.W1, self.W2, self.Dt
        debugging = self.set_ground_truth
        ZERO_CHECK_INTERVAL = 1500
        nzerocount = np.zeros(s_dim)
        whiten_input_ = self.whiten_input_

        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"
        samples = X.shape[1]

        if whiten_input_:
            X_ = self.whiten_input(X)
            x_dim = X_.shape[0]
        else:
            X_ = X

        Wpre = self.Wpre

        if debugging:
            SIRlist = self.SIRlist
            S = self.S
            A = self.A

        for k in range(n_epochs):
            if shuffle:
                idx = np.random.permutation(samples)
            else:
                idx = np.arange(samples)

            for i_sample in tqdm(range(samples)):
                x_current = X_[:, idx[i_sample]]
                xk = np.reshape(x_current, (-1,1))

                y = np.random.rand(s_dim, 1)

                y = self.run_neural_dynamics(xk, y, W1, W2, neural_dynamic_iterations)

                Dt = np.minimum(3000, 0.94 * Dt + y ** 2)
                DtD = np.diag(1 / Dt.reshape((s_dim)))
                W1 = W1 + np.dot(DtD, (np.dot(y, (xk.T).reshape((1, x_dim))) - np.dot(np.diag((y ** 2).reshape((s_dim))), W1)))
                W2 = W2 + np.dot(DtD, (np.dot(y, y.T) - np.dot(np.diag((y ** 2).reshape((s_dim))), W2)))

                for ind in range(s_dim):
                    W2[ind, ind] = 0

                nzerocount = (nzerocount + (y.reshape(s_dim) == 0) * 1.0) * (y.reshape(s_dim) == 0)
                if k < ZERO_CHECK_INTERVAL:
                    q = np.argwhere(nzerocount > 50)
                    qq = q[:,0]
                    for iter3 in range(len(qq)):
                        W1[qq[iter3], :] = -W1[qq[iter3], :]
                        nzerocount[qq[iter3]] = 0

                self.W1 = W1
                self.W2 = W2
                self.Dt = Dt

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        # self.SIRlist = SIRlist

                        Wf = self.compute_overall_mapping(return_mapping = True)
                        SIR,_ =  self.CalculateSIR(A, Wf)
                        SIRlist.append(SIR)
                        self.SIRlist = SIRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth = 3)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
                            pl.ylabel("SIR (dB)", fontsize = 15)
                            pl.title("SIR Behavior", fontsize = 15)
                            pl.grid()
                            clear_output(wait = True)
                            display(pl.gcf())
 
class OnlineBCA:
    """
    Implementation of online two layer Recurrent Neural Network with Local Update Rule for Unsupervised Seperation of Sources.
    Reference: B. Simsek and A. T. Erdogan, "Online Bounded Component Analysis: A Simple Recurrent Neural Network with Local Update Rule for Unsupervised Separation of Dependent and Independent Sources," 2019
    
    Parameters:
    =================================
    s_dim          -- Dimension of the sources
    x_dim          -- Dimension of the mixtures
    F              -- Feedforward Synaptic Connection Matrix, must be size of (s_dim, x_dim)
    B              -- Recurrent Synaptic Connection Matrix, must be size of (s_dim, s_dim)
    lambda_        -- Forgetting factor (close to 1, but less than 1)
    
    gamma_hat
    beta
    mu_F
    mu_y
    
    
    Methods:
    ==================================
    
    whiten_signal(X)        -- Whiten the given batch signal X
    
    ProjectOntoLInfty(X)   -- Project the given vector X onto L_infinity norm ball
    
    fit_next_antisparse(x_online)     -- Updates the network parameters for one data point x_online
    
    fit_batch_antisparse(X_batch)     -- Updates the network parameters for given batch data X_batch (but in online manner)
    
    """
    
    def __init__(self, s_dim, x_dim, lambda_ = 0.999, mu_F = 0.03, beta = 5, F = None, B = None, neural_OUTPUT_COMP_TOL = 1e-6, set_ground_truth = False, S = None, A = None):
        if F is not None:
            assert F.shape == (s_dim, x_dim), "The shape of the initial guess F must be (s_dim, x_dim) = (%d,%d)" % (s_dim, x_dim)
            F = F
        else:
            F = np.random.randn(s_dim,x_dim)
            F = (F / np.sqrt(np.sum(np.abs(F)**2,axis = 1)).reshape(s_dim,1))
            F = np.eye(s_dim, x_dim)
            
        if B is not None:
            assert B.shape == (s_dim,s_dim), "The shape of the initial guess B must be (s_dim, s_dim) = (%d,%d)" % (s_dim,s_dim)
            B = B
        else:
            B = 5*np.eye(s_dim)
            
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.lambda_ = lambda_
        self.beta = beta
        self.mu_F = mu_F
        self.gamma_hat = (1-lambda_)/lambda_
        self.F = F
        self.B = B
        self.neural_OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL
        self.set_ground_truth = set_ground_truth
        self.SIRlist = []
        self.S = S
        self.A = A
        
    # Calculate SIR Function
    def CalculateSIR(self, H,pH, return_db = True):
        G=pH@H
        Gmax=np.diag(np.max(abs(G),axis=1))
        P=1.0*((np.linalg.inv((Gmax))@np.abs(G))>0.99)
        T=G@P.T
        rankP=np.linalg.matrix_rank(P)
        diagT = np.diag(T)
        # Signal Power
        sigpow = np.linalg.norm(diagT,2)**2
        # Interference Power
        intpow = np.linalg.norm(T,'fro')**2 - sigpow
        SIRV = sigpow/intpow
        # SIRV=np.linalg.norm((np.diag(T)))**2/(np.linalg.norm(T,'fro')**2-np.linalg.norm(np.diag(T))**2)
        if return_db:
            SIRV = 10*np.log10(sigpow/intpow)

        return SIRV,rankP

    def whiten_signal(self, X, mean_normalize = True, type_ = 3):
        """
        Input : X  ---> Input signal to be whitened
        type_ : Defines the type for preprocesing matrix. type_ = 1 and 2 uses eigenvalue decomposition whereas type_ = 3 uses SVD.
        Output: X_white  ---> Whitened signal, i.e., X_white = W_pre @ X where W_pre = (R_x^0.5)^+ (square root of sample correlation matrix)
        """
        if mean_normalize:
            X = X - np.mean(X,axis = 0, keepdims = True)

        cov = np.cov(X.T)

        if type_ == 3: # Whitening using singular value decomposition
            U,S,V = np.linalg.svd(cov)
            d = np.diag(1.0 / np.sqrt(S))
            W_pre = np.dot(U, np.dot(d, U.T))

        else: # Whitening using eigenvalue decomposition
            d,S = np.linalg.eigh(cov)
            D = np.diag(d)

            D_sqrt = np.sqrt(D * (D>0))

            if type_ == 1: # Type defines how you want W_pre matrix to be
                W_pre = np.linalg.pinv(S@D_sqrt)
            elif type_ == 2:
                W_pre = np.linalg.pinv(S@D_sqrt@S.T)

        X_white = (W_pre @ X.T).T

        return X_white, W_pre
    
    def ProjectOntoLInfty(self, X):
        
        return X*(X>=-1.0)*(X<=1.0)+(X>1.0)*1.0-1.0*(X<-1.0)
    
    @staticmethod
    @njit
    def run_neural_dynamics_antisparse(x, y, F, B, beta, gamma_hat, mu_y_start = 0.9, neural_dynamic_iterations = 100, neural_OUTPUT_COMP_TOL = 1e-7):
        def ProjectOntoLInfty(X, thresh = 1.0):
            return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)

        yke = np.dot(F, x)
        for j in range(neural_dynamic_iterations):
            mu_y = mu_y_start / (j+1)
            y_old = y.copy()
            e = yke - y
            y = y + mu_y*(gamma_hat * B @ y + beta * e)
            y = ProjectOntoLInfty(y)

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    @staticmethod
    @njit
    def run_neural_dynamics_nnantisparse(x, y, F, B, beta, gamma_hat, mu_y_start = 0.9, neural_dynamic_iterations = 100, neural_OUTPUT_COMP_TOL = 1e-7):
        def ProjectOntoNNLInfty(X, thresh = 1.0):
            return X*(X>=0.0)*(X<=thresh)+(X>thresh)*thresh #-thresh*(X<-thresh)

        yke = np.dot(F, x)
        for j in range(neural_dynamic_iterations):
            mu_y = mu_y_start / (j+1)
            y_old = y.copy()
            e = yke - y
            y = y + mu_y*(gamma_hat * B @ y + beta * e)
            y = ProjectOntoNNLInfty(y)

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    def compute_overall_mapping(self, return_mapping = False):
        F, B, gamma_hat, beta = self.F, self.B, self.gamma_hat, self.beta
        W = np.linalg.pinv((gamma_hat/beta) * B - np.eye(self.s_dim)) @ F
        if return_mapping:
            return W
        else:
            return None

    def fit_next_antisparse(self,x_current, neural_dynamic_iterations = 250, lr_start = 0.9):
        
        lambda_, beta, mu_F, gamma_hat, F, B = self.lambda_, self.beta, self.mu_F, self.gamma_hat, self.F, self.B
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        
        y = np.zeros(self.s_dim)
        
        y = self.run_neural_dynamics_antisparse(x_current, y, F, B, beta, gamma_hat, 
                                                mu_y_start = lr_start, neural_dynamic_iterations = neural_dynamic_iterations, 
                                                neural_OUTPUT_COMP_TOL = neural_dynamic_tol)
        
        e = F @ x_current - y

        F = F - mu_F * beta * np.outer(e, x_current)

        B = (1/lambda_) * (B - gamma_hat * np.outer(B @ y, B @ y))        
        
        self.F = F
        self.B = B
        
        # return y
        
        
    def fit_batch_antisparse(self, X, n_epochs = 2, neural_dynamic_iterations = 250, lr_start = 0.9, whiten = False, whiten_type = 2, shuffle = False, verbose = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        
        lambda_, beta, mu_F, gamma_hat, F, B = self.lambda_, self.beta, self.mu_F, self.gamma_hat, self.F, self.B
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            S = self.S
            A = self.A

        # Y = np.zeros((self.s_dim, samples))
        Y = np.random.randn(self.s_dim, samples)
        
        
        if shuffle:
            idx = np.random.permutation(samples) # random permutation
        else:
            idx = np.arange(samples)
            
        if whiten:
            X_white, W_pre = self.whiten_signal(X.T, type_ = whiten_type)
            X_white = X_white.T
            A = W_pre @ A
            self.A = A
        else:
            X_white = X 
            
            
        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X_white[:,idx[i_sample]]
                y = np.zeros(self.s_dim)

                y = self.run_neural_dynamics_antisparse(x_current, y, F, B, beta, gamma_hat, 
                                                        mu_y_start = lr_start, neural_dynamic_iterations = neural_dynamic_iterations, 
                                                        neural_OUTPUT_COMP_TOL = neural_dynamic_tol)
                        
                e = F @ x_current - y

                F = F - mu_F * beta * np.outer(e, x_current)
                
                z = B @ y
                B = (1/lambda_) * (B - gamma_hat * np.outer(z, z))
                # Record the seperated signal
                Y[:,idx[i_sample]] = y

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.F = F
                        self.B = B
                        W = self.compute_overall_mapping(return_mapping = True)
                        SIR = self.CalculateSIR(A, W)[0]
                        SIRlist.append(SIR)
                        self.SIR_list = SIRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth = 3)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
                            pl.ylabel("SIR (dB)", fontsize = 15)
                            pl.title("SIR Behaviour", fontsize = 15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())   
        self.F = F
        self.B = B
        
    def fit_batch_nnantisparse(self, X, n_epochs = 2, neural_dynamic_iterations = 250, lr_start = 0.9, whiten = False, whiten_type = 2, shuffle = False, verbose = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        
        lambda_, beta, mu_F, gamma_hat, F, B = self.lambda_, self.beta, self.mu_F, self.gamma_hat, self.F, self.B
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            S = self.S
            A = self.A

        # Y = np.zeros((self.s_dim, samples))
        Y = np.random.randn(self.s_dim, samples)
        
        
        if shuffle:
            idx = np.random.permutation(samples) # random permutation
        else:
            idx = np.arange(samples)
            
        if whiten:
            X_white, W_pre = self.whiten_signal(X.T, type_ = whiten_type)
            X_white = X_white.T
            A = W_pre @ A
            self.A = A
        else:
            X_white = X 
            
            
        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X_white[:,idx[i_sample]]
                y = np.zeros(self.s_dim)

                y = self.run_neural_dynamics_nnantisparse(x_current, y, F, B, beta, gamma_hat, 
                                                        mu_y_start = lr_start, neural_dynamic_iterations = neural_dynamic_iterations, 
                                                        neural_OUTPUT_COMP_TOL = neural_dynamic_tol)
                        
                e = F @ x_current - y

                F = F - mu_F * beta * np.outer(e, x_current)
                
                z = B @ y
                B = (1/lambda_) * (B - gamma_hat * np.outer(z, z))
                # Record the seperated signal
                Y[:,idx[i_sample]] = y

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.F = F
                        self.B = B
                        W = self.compute_overall_mapping(return_mapping = True)
                        SIR = self.CalculateSIR(A, W)[0]
                        SIRlist.append(SIR)
                        self.SIR_list = SIRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth = 3)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
                            pl.ylabel("SIR (dB)", fontsize = 15)
                            pl.title("SIR Behaviour", fontsize = 15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())   
        self.F = F
        self.B = B

class OnlineBSM:
    """
    BOUNDED SIMILARITY MATCHING
    Implementation of online one layer Weighted Bounded Source Seperation Recurrent Neural Network.
    Reference: Alper T. Erdoğan and Cengiz Pehlevan, 'Blind Source Seperation Using Neural Networks with Local Learning Rules',ICASSP 2020
    
    Parameters:
    =================================
    s_dim          -- Dimension of the sources
    x_dim          -- Dimension of the mixtures
    W              -- Initial guess for forward weight matrix W, must be size of s_dim by x_dim
    M              -- Initial guess for lateral weight matrix M, must be size of s_dim by s_dim
    D              -- Initial guess for weight (similarity weights) matrix, must be size of s_dim by s_dim
    gamma          -- Forgetting factor for data snapshot matrix
    mu, beta       -- Similarity weight update parameters, check equation (15) from the paper
    
    Methods:
    ==================================
    
    whiten_signal(X)        -- Whiten the given batch signal X
    
    ProjectOntoLInfty(X)   -- Project the given vector X onto L_infinity norm ball
    
    fit_next_antisparse(x_online)     -- Updates the network parameters for one data point x_online
    
    fit_batch_antisparse(X_batch)     -- Updates the network parameters for given batch data X_batch (but in online manner)
    
    """
    def __init__(self, s_dim, x_dim, gamma = 0.9999, mu = 1e-3, beta = 1e-7, W = None, M = None, D = None, whiten_input_ = True, neural_OUTPUT_COMP_TOL = 1e-6, set_ground_truth = False, S = None, A = None):
        if W is not None:
            if whiten_input_:
                assert W.shape == (s_dim, s_dim), "The shape of the initial guess W must be (s_dim,s_dim)=(%d,%d) (because of whitening)" % (s_dim, x_dim)
                W = W
            else:
                assert W.shape == (s_dim, x_dim), "The shape of the initial guess W must be (s_dim,x_dim)=(%d,%d)" % (s_dim, x_dim)
                W = W
        else:
            if whiten_input_:
                W = np.random.randn(s_dim,s_dim)
                W = 0.0033 * (W / np.sqrt(np.sum(np.abs(W)**2,axis = 1)).reshape(s_dim,1))
            else:
                W = np.random.randn(s_dim,x_dim)
                W = 0.0033 * (W / np.sqrt(np.sum(np.abs(W)**2,axis = 1)).reshape(s_dim,1))
            # for k in range(W_HX.shape[0]):
            #     W_HX[k,:] = WScalings[0] * W_HX[k,:]/np.linalg.norm(W_HX[k,:])
            
        if M is not None:
            assert M.shape == (s_dim, s_dim), "The shape of the initial guess W must be (s_dim,s_dim)=(%d,%d)" % (s_dim, s_dim)
            M = M
        else:
            M = 0.02*np.eye(s_dim)  
            
        if D is not None:
            assert D.shape == (s_dim, s_dim), "The shape of the initial guess W must be (s_dim,s_dim)=(%d,%d)" % (s_dim, s_dim)
            D = D
        else:
            D = 1*np.eye(s_dim)
            
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.gamma = gamma
        self.mu = mu
        self.beta = beta
        self.W = W
        self.M = M
        self.D = D
        self.whiten_input_ = whiten_input_
        self.neural_OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL
        self.set_ground_truth = set_ground_truth
        self.S = S
        self.A = A
        self.SIRlist = []
        
    def whiten_input(self, X):
        x_dim = self.x_dim
        s_dim = self.s_dim
        N = X.shape[1]
        # Mean of the mixtures
        mX = np.mean(X, axis = 1).reshape((x_dim, 1))
        # Covariance of Mixtures
        Rxx = np.dot(X, X.T)/N - np.dot(mX, mX.T)
        # Eigenvalue Decomposition
        d, V = np.linalg.eig(Rxx)
        D = np.diag(d)
        # Sorting indexis for eigenvalues from large to small
        ie = np.argsort(-d)
        # Inverse square root of eigenvalues
        ddinv = 1/np.sqrt(d[ie[:s_dim]])
        # Pre-whitening matrix
        Wpre = np.dot(np.diag(ddinv), V[:, ie[:s_dim]].T)#*np.sqrt(12)
        # Whitened mixtures
        H = np.dot(Wpre, X)
        self.Wpre = Wpre
        return H, Wpre

    # Calculate SIR Function
    def CalculateSIR(self, H,pH, return_db = True):
        G=pH@H
        Gmax=np.diag(np.max(abs(G),axis=1))
        P=1.0*((np.linalg.inv((Gmax))@np.abs(G))>0.99)
        T=G@P.T
        rankP=np.linalg.matrix_rank(P)
        diagT = np.diag(T)
        # Signal Power
        sigpow = np.linalg.norm(diagT,2)**2
        # Interference Power
        intpow = np.linalg.norm(T,'fro')**2 - sigpow
        SIRV = sigpow/intpow
        # SIRV=np.linalg.norm((np.diag(T)))**2/(np.linalg.norm(T,'fro')**2-np.linalg.norm(np.diag(T))**2)
        if return_db:
            SIRV = 10*np.log10(sigpow/intpow)

        return SIRV,rankP
    
    def ProjectOntoLInfty(self, X):

        return X*(X>=-1.0)*(X<=1.0)+(X>1.0)*1.0-1.0*(X<-1.0)
    
    def compute_overall_mapping(self,return_mapping = True):
        W, M, D = self.W, self.M, self.D

        Wf = np.linalg.pinv(M @ D) @ W
        self.Wf = Wf

        if return_mapping:
            return Wf
        else:
            return None

    @staticmethod
    @njit
    def run_neural_dynamics_antisparse(x, y, W, M, D,neural_dynamic_iterations = 250, lr_start = 0.1, lr_stop = 1e-15, tol = 1e-6, fast_start = False):

        def ProjectOntoLInfty(X, thresh = 1.0):
            return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)
        
        Upsilon = np.diag(np.diag(M))
        M_hat = M - Upsilon
        u = Upsilon @ D @ y

        if fast_start:
            u = 0.99*np.linalg.solve(M @ D, W @ x)
            y = ProjectOntoLInfty(u / np.diag(Upsilon * D))

        for j in range(neural_dynamic_iterations):
            lr = max(lr_start/(1 + j), lr_stop)
            yold = y
            du = -u + (W @ x - M_hat @ D @ y)
            # u = u - lr * du
            y = y - lr * du

            y = ProjectOntoLInfty(u / np.diag(Upsilon * D))

            if np.linalg.norm(y - yold) < tol * np.linalg.norm(y):
                break

        return y

    def fit_next_antisparse(self, x_current, neural_dynamic_iterations = 250, neural_lr_start = 0.3, neural_lr_stop = 1e-3, fast_start = False):
        W = self.W
        M = self.M
        D = self.D
        gamma, mu, beta = self.gamma, self.mu, self.beta
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        # Upsilon = np.diag(np.diag(M))
        
        # u = np.linalg.solve(M @ D, W @ x_current)
        # y = self.ProjectOntoLInfty(u / np.diag(Upsilon * D))
        y = np.random.randn(self.s_dim,)
        y = self.run_neural_dynamics_antisparse(x_current, y, W, M, D, neural_dynamic_iterations, neural_lr_start, neural_lr_stop, neural_OUTPUT_COMP_TOL, fast_start)

        
        W = (gamma ** 2) * W + (1 - gamma ** 2) * np.outer(y,x_current)
        M = (gamma ** 2) * M + (1 - gamma ** 2) * np.outer(y,y)
        
        D = (1 - beta) * D + mu * np.diag(np.sum(np.abs(W)**2,axis = 1) - np.diag(M @ D @ M ))
        
        self.W = W
        self.M = M
        self.D = D
        
        
    def fit_batch_antisparse(self, X, n_epochs = 1, shuffle = False, neural_dynamic_iterations = 250, neural_lr_start = 0.3, neural_lr_stop = 1e-3, fast_start = False, debug_iteration_point = 1000, plot_in_jupyter = False):
        gamma, mu, beta, W, M, D = self.gamma, self.mu, self.beta, self.W, self.M, self.D
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
        SIRlist = self.SIRlist
        whiten = self.whiten_input_

        if debugging:
            S = self.S
            A = self.A

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]

        Y = 0.05*np.random.randn(self.s_dim, samples)
        
        if shuffle:
            idx = np.random.permutation(samples) # random permutation
        else:
            idx = np.arange(samples)
            
        if whiten:
            X_white, W_pre = self.whiten_input(X)
            A = W_pre @ A
            self.A = A
        else:
            X_white = X
            

        for k in range(n_epochs):
            for i_sample in tqdm(range(samples)):
                x_current = X_white[:, idx[i_sample]] # Take one input
                y = Y[:, idx[i_sample]]

                # Upsilon = np.diag(np.diag(M)) # Following paragraph of equation (16)
                
                # Neural Dynamics: Equations (17) from the paper
                
                # u = np.linalg.solve(M @ D, W @ x_current)
                # y = self.ProjectOntoLInfty(u / np.diag(Upsilon * D))
                y = self.run_neural_dynamics_antisparse(x_current, y, W, M, D, neural_dynamic_iterations, neural_lr_start, neural_lr_stop, neural_OUTPUT_COMP_TOL, fast_start)
                
                # Synaptic & Similarity weight updates, follows from equations (12,13,14,15,16) from the paper
                
                W = (gamma ** 2) * W + (1 - gamma ** 2) * np.outer(y,x_current)
                M = (gamma ** 2) * M + (1 - gamma ** 2) * np.outer(y,y)
                D = (1 - beta) * D + mu * np.diag(np.sum(np.abs(W)**2,axis = 1) - np.diag(M @ D @ M ))
                
                # Record the seperated signal
                Y[:, idx[i_sample]] = y
                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.M = M
                        self.D = D
                        Wf = self.compute_overall_mapping(return_mapping = True)
                        SIR,_ = self.CalculateSIR(A, Wf)
                        SIRlist.append(SIR)
                        self.SIRlist = SIRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth = 3)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
                            pl.ylabel("SIR (dB)", fontsize = 15)
                            pl.title("SIR Behaviour", fontsize = 15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())         

        self.W = W
        self.M = M
        self.D = D

def whiten_signal(X, mean_normalize = True, type_ = 3):
    """
    Input : X  ---> Input signal to be whitened
    
    type_ : Defines the type for preprocesing matrix. type_ = 1 and 2 uses eigenvalue decomposition whereas type_ = 3 uses SVD.
    
    Output: X_white  ---> Whitened signal, i.e., X_white = W_pre @ X where W_pre = (R_x^0.5)^+ (square root of sample correlation matrix)
    """
    if mean_normalize:
        X = X - np.mean(X,axis = 0, keepdims = True)
    
    cov = np.cov(X.T)
    
    if type_ == 3: # Whitening using singular value decomposition
        U,S,V = np.linalg.svd(cov)
        d = np.diag(1.0 / np.sqrt(S))
        W_pre = np.dot(U, np.dot(d, U.T))
        
    else: # Whitening using eigenvalue decomposition
        d,S = np.linalg.eigh(cov)
        D = np.diag(d)

        D_sqrt = np.sqrt(D * (D>0))

        if type_ == 1: # Type defines how you want W_pre matrix to be
            W_pre = np.linalg.pinv(S@D_sqrt)
        elif type_ == 2:
            W_pre = np.linalg.pinv(S@D_sqrt@S.T)
    
    X_white = (W_pre @ X.T).T
    
    return X_white, W_pre

def whiten_input(X, n_components = None, return_prewhitening_matrix = False):
    """
    X.shape[0] = Number of sources
    X.shape[1] = Number of samples for each signal
    """
    x_dim = X.shape[0]
    if n_components is None:
        n_components = x_dim
    s_dim = n_components
    
    N = X.shape[1]
    # Mean of the mixtures
    mX = np.mean(X, axis = 1).reshape((x_dim, 1))
    # Covariance of Mixtures
    Rxx = np.dot(X, X.T)/N - np.dot(mX, mX.T)
    # Eigenvalue Decomposition
    d, V = np.linalg.eig(Rxx)
    D = np.diag(d)
    # Sorting indexis for eigenvalues from large to small
    ie = np.argsort(-d)
    # Inverse square root of eigenvalues
    ddinv = 1/np.sqrt(d[ie[:s_dim]])
    # Pre-whitening matrix
    Wpre = np.dot(np.diag(ddinv), V[:, ie[:s_dim]].T)#*np.sqrt(12)
    # Whitened mixtures
    H = np.dot(Wpre, X)
    if return_prewhitening_matrix:
        return H, Wpre
    else:
        return H

def ZeroOneNormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def ZeroOneNormalizeColumns(X):
    X_normalized = np.empty_like(X)
    for i in range(X.shape[1]):
        X_normalized[:,i] = ZeroOneNormalizeData(X[:,i])

    return X_normalized

def ProjectOntoLInfty(X, thresh = 1.0):
    return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)

def ProjectOntoNNLInfty(X, thresh = 1.0):
    return X*(X>=0)*(X<=thresh)+(X>thresh)*thresh

def Subplot_gray_images(I, image_shape = [512,512], height = 15, width = 15, title = ''):
    n_images = I.shape[1]
    fig, ax = plt.subplots(1,n_images)
    fig.suptitle(title)
    fig.set_figheight(height)
    fig.set_figwidth(width)
    for i in range(n_images):
        ax[i].imshow(I[:,i].reshape(image_shape[0],image_shape[1]), cmap = 'gray')
    
    plt.show()

def subplot_1D_signals(X, title = '',title_fontsize = 20, figsize = (10,5), linewidth = 1, colorcode = '#050C12'):
    """
    Plot the 1D signals (each column from the given matrix)
    """
    n = X.shape[1] # Number of signals
    
    fig, ax = plt.subplots(n,1, figsize = figsize)
    
    for i in range(n):
        ax[i].plot(X[:,i], linewidth = linewidth, color = colorcode)
        ax[i].grid()
    
    plt.suptitle(title, fontsize = title_fontsize)
    # plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
    plt.draw()

def plot_convergence_plot(metric, xlabel = '', ylabel = '', title = '', figsize = (12,8), fontsize = 15, linewidth = 3, colorcode = '#050C12'):
    
    plt.figure(figsize = figsize)
    plt.plot(metric, linewidth = linewidth, color = colorcode)
    plt.xlabel(xlabel, fontsize = fontsize)
    plt.ylabel(ylabel, fontsize = fontsize)
    plt.title(title, fontsize = fontsize)
    # plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
    plt.grid()
    plt.draw()
    
def find_permutation_between_source_and_estimation(S,Y):
    """
    S    : Original source matrix
    Y    : Matrix of estimations of sources (after BSS or ICA algorithm)
    
    return the permutation of the source seperation algorithm
    """
    
    # perm = np.argmax(np.abs(np.corrcoef(S.T,Y.T) - np.eye(2*S.shape[1])),axis = 0)[S.shape[1]:]
    perm = np.argmax(np.abs(np.corrcoef(Y.T,S.T) - np.eye(2*S.shape[1])),axis = 0)[S.shape[1]:]
    return perm

def signed_and_permutation_corrected_sources(S,Y):
    perm = find_permutation_between_source_and_estimation(S,Y)
    return np.sign((Y[:,perm] * S).sum(axis = 0)) * Y[:,perm]

def psnr(img1, img2, pixel_max = 1):
    """
    Return peak-signal-to-noise-ratio between given two images
    """
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    else:
        return 20 * np.log10(pixel_max / np.sqrt(mse))

def snr(S_original, S_noisy):
    N_hat = S_original - S_noisy
    N_P = (N_hat ** 2).sum(axis = 0)
    S_P = (S_original ** 2).sum(axis = 0)
    snr = 10 * np.log10(S_P / N_P)
    return snr

def ProjectRowstoL1NormBall(H):
    Hshape=H.shape
    #lr=np.ones((Hshape[0],1))@np.reshape((1/np.linspace(1,Hshape[1],Hshape[1])),(1,Hshape[1]))
    lr=np.tile(np.reshape((1/np.linspace(1,Hshape[1],Hshape[1])),(1,Hshape[1])),(Hshape[0],1))
    #Hnorm1=np.reshape(np.sum(np.abs(self.H),axis=1),(Hshape[0],1))

    u=-np.sort(-np.abs(H),axis=1)
    sv=np.cumsum(u,axis=1)
    q=np.where(u>((sv-1)*lr),np.tile(np.reshape((np.linspace(1,Hshape[1],Hshape[1])-1),(1,Hshape[1])),(Hshape[0],1)),np.zeros((Hshape[0],Hshape[1])))
    rho=np.max(q,axis=1)
    rho=rho.astype(int)
    lindex=np.linspace(1,Hshape[0],Hshape[0])-1
    lindex=lindex.astype(int)
    theta=np.maximum(0,np.reshape((sv[tuple([lindex,rho])]-1)/(rho+1),(Hshape[0],1)))
    ww=np.abs(H)-theta
    H=np.sign(H)*(ww>0)*ww
    return H

def display_matrix(array):
    data = ''
    for line in array:
        if len(line) == 1:
            data += ' %.3f &' % line + r' \\\n'
            continue
        for element in line:
            data += ' %.3f &' % element
        data += r' \\' + '\n'
    display(Math('\\begin{bmatrix} \n%s\end{bmatrix}' % data))

# Calculate SIR Function
def CalculateSIR(H,pH, return_db = True):
    G=pH@H
    Gmax=np.diag(np.max(abs(G),axis=1))
    P=1.0*((np.linalg.inv((Gmax))@np.abs(G))>0.99)
    T=G@P.T
    rankP=np.linalg.matrix_rank(P)
    diagT = np.diag(T)
    # Signal Power
    sigpow = np.linalg.norm(diagT,2)**2
    # Interference Power
    intpow = np.linalg.norm(T,'fro')**2 - sigpow
    SIRV = sigpow/intpow
    # SIRV=np.linalg.norm((np.diag(T)))**2/(np.linalg.norm(T,'fro')**2-np.linalg.norm(np.diag(T))**2)
    if return_db:
        SIRV = 10*np.log10(sigpow/intpow)

    return SIRV,rankP

@njit(fastmath = True)
def accumu(lis):
    """
    Cumulative Sum. Same as np.cumsum()
    """
    result = np.zeros_like(lis)
    for i in range(lis.shape[1]):
        result[:,i] = np.sum(lis[:,:i+1])

    return result

@njit(fastmath = True)
def merge_sort(list_):
    """
    Sorts a list in ascending order.
    Returns a new sorted list.
    
    Divide : Find the midpoint of the list and divide into sublist
    Conquer : Recursively sort the sublists created in previous step
    Combine : Merge the sorted sublists created in previous step
    
    Takes O(n log n) time.
    """
    
    def merge(left, right):
        """
        Merges two lists (arrays), sorting them in the process.
        Returns a new merged list
        
        Runs in overall O(n) time
        """
        
        l = []
        i = 0
        j = 0
        
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                l.append(left[i])
                i += 1
            else:
                l.append(right[j])
                j += 1
                
        while i < len(left):
            l.append(left[i])
            i += 1
            
        while j < len(right):
            l.append(right[j])
            j += 1
        
        return l

    def split(list_):
        """
        Divide the unsorted list at midpoint into sublists.
        Returns two sublists - left and right
        
        Takes overall O(log n) time
        """
        
        mid = len(list_) // 2
        
        left = list_[:mid]
        right = list_[mid:]
        
        return left, right

    if len(list_) <= 1:
        return list_
    
    left_half, right_half = split(list_)
    left = merge_sort(left_half)
    right = merge_sort(right_half)
    
    return np.array(merge(left, right))

# @njit
# def ProjectVectortoL1NormBall(H):
#     Hshape = H.shape
#     lr = np.repeat(np.reshape((1/np.linspace(1, H.shape[1], H.shape[1]))))

def generate_correlated_uniform_sources(R, range_ = [-1,1], n_sources = 5, size_sources = 500000):
    """
    R : correlation matrix
    """
    assert R.shape[0] == n_sources, "The shape of correlation matrix must be equal to the number of sources, which is entered as (%d)" % (n_sources)
    S = np.random.uniform(range_[0], range_[1], size = (n_sources, size_sources))
    L = np.linalg.cholesky(R)
    S_ = L @ S
    return S_

def generate_correlated_copula_sources(rho = 0.0, df = 4, n_sources = 5, size_sources = 500000, decreasing_correlation = True):
    """
    rho     : correlation parameter
    df      : degrees for freedom

    required libraries:
    from scipy.stats import invgamma, chi2, t
    from scipy import linalg
    import numpy as np
    """
    if decreasing_correlation:
        first_row = np.array([rho ** j for j in range(n_sources)])
        calib_correl_matrix = linalg.toeplitz(first_row, first_row)
    else:
        calib_correl_matrix = np.eye(n_sources) * (1 - rho) + np.ones((n_sources, n_sources)) * rho

    mu = np.zeros(len(calib_correl_matrix))
    s = chi2.rvs(df, size = size_sources)[:, np.newaxis]
    Z = np.random.multivariate_normal(mu, calib_correl_matrix, size_sources)
    X = np.sqrt(df/s) * Z # chi-square method
    S = t.cdf(X, df).T
    return S

