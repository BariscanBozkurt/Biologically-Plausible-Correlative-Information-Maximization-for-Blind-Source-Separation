import numpy as np
import scipy
from scipy.stats import invgamma, chi2, t
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
import matplotlib as mpl
import pylab as pl
from tqdm import tqdm
from numba import njit, jit
import logging
from time import time
import os
from IPython.display import display, clear_output, Latex, Math
from IPython import display as display1
# np.random.seed(123)
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
    def __init__(self, s_dim, x_dim, W = None, Wscaling = 1, muW = 2e-3, By = None, Be = None, lambday = 1 - 1e-6, lambdae = 1 - 1e-7, 
                    neural_OUTPUT_COMP_TOL = 1e-5, set_ground_truth = False, S = None, A = None):
        if W is not None:
            assert W.shape == (s_dim, x_dim), "The shape of the initial guess W must be (s_dim,x_dim)=(%d,%d)" % (s_dim, x_dim)
            W = W
        else:
            W = np.random.randn(s_dim,x_dim)
            for k in range(W.shape[0]):
                W[k,:] =  Wscaling * W[k,:]/np.linalg.norm(W[k,:])

        if By is not None:
            assert By.shape == (s_dim, s_dim), "The shape of the initial guess By must be (s_dim,s_dim)=(%d,%d)" % (s_dim, s_dim)
            By = By
        else:
            By = 5 * np.eye(s_dim)

        if Be is not None:
            assert Be.shape == (s_dim, s_dim), "The shape of the initial guess Be must be (s_dim,s_dim)=(%d,%d)" % (s_dim, s_dim)
            Be = Be
        else:
            Be = 100 * np.eye(s_dim)

        self.s_dim = s_dim
        self.x_dim = x_dim
        self.W = W
        self.muW = muW
        self.By = By
        self.Be = Be
        self.lambday = lambday
        self.lambdae = lambdae
        self.gamy = (1-lambday)/lambday
        self.game = (1-lambdae)/lambdae
        self.neural_OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL
        
        ### Ground Truth Sources and Mixing Matrix For Debugging
        self.set_ground_truth = set_ground_truth
        self.S = S # Sources
        self.A = A # Mixing Matrix
        self.SIR_list = []
        self.SNR_list = []
        

    def ProjectOntoLInfty(self, X, thresh):
        return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)

    ############################################################################################
    ############### REQUIRED FUNCTIONS FOR DEBUGGING ###########################################
    ############################################################################################
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

    def outer_prod_broadcasting(self, A, B):
        """Broadcasting trick"""
        return A[...,None]*B[:,None]

    def find_permutation_between_source_and_estimation(self, S,Y):
        """
        S    : Original source matrix
        Y    : Matrix of estimations of sources (after BSS or ICA algorithm)
        
        return the permutation of the source seperation algorithm
        """
        # perm = np.argmax(np.abs(np.corrcoef(S.T,Y.T) - np.eye(2*S.shape[1])),axis = 0)[S.shape[1]:]
        # perm = np.argmax(np.abs(np.corrcoef(Y.T,S.T) - np.eye(2*S.shape[1])),axis = 0)[S.shape[1]:]
        # perm = np.argmax(np.abs(outer_prod_broadcasting(S,Y).sum(axis = 0)), axis = 0)
        perm = np.argmax(np.abs(self.outer_prod_broadcasting(Y,S).sum(axis = 0))/(np.linalg.norm(S,axis = 0)*np.linalg.norm(Y,axis=0)), axis = 0)
        return perm

    def signed_and_permutation_corrected_sources(self,S,Y):
        perm = self.find_permutation_between_source_and_estimation(S,Y)
        return np.sign((Y[:,perm] * S).sum(axis = 0)) * Y[:,perm]

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
        gamy, game = self.gamy, self.game
        By, Be = self.By, self.Be
        W = self.W
        Wf = np.linalg.inv(game * Be - gamy * By) @ Be @ W * game

        self.Wf = Wf
        if return_mapping:
            return Wf

    @staticmethod
    @njit
    def run_neural_dynamics_antisparse(yk, yke, My, gamy, Be, game, vk, n_iterations = 250, lr_start = 1.5, lr_stop = 0.001, tol = 1e-8):
        # Hopefield parameter
        h = 1 / gamy
        def ProjectOntoLInfty(X, thresh = 1.0):
            return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)
        
        for j in range(n_iterations):
            muv = max(lr_start/(1 + j * 0.005), lr_stop)
            ykold = yk
            #Find error
            ek = yk - yke
            # Gradient of the entropy cost H(y)
            grady = gamy * np.dot(My, yk)
            # Gradient for H(yk|xk)
            grade = game * (np.dot(Be, ek) + ek)
            # Overall gradient
            gradv = -vk + grady - grade
            # Update v
            vk = vk + muv * gradv / np.linalg.norm(gradv) / (j+1)
            
            yk = ProjectOntoLInfty(vk / (h * gamy))
            if np.linalg.norm(yk - ykold) < tol * np.linalg.norm(ykold):
                break
            
        return yk, ek

    @staticmethod
    @njit
    def run_neural_dynamics_nnantisparse(yk, yke, My, gamy, Be, game, vk, n_iterations = 250, lr_start = 1.5, lr_stop = 0.001, tol = 1e-8):
        # Hopefield parameter
        h = 1 / gamy
        def ProjectOntoNNLInfty(X, thresh = 1.0):
            return X*(X>=0)*(X<=thresh)+(X>thresh)*thresh
        
        for j in range(n_iterations):
            muv = max(lr_start/(1 + j), lr_stop)
            ykold = yk
            #Find error
            ek = yk - yke
            # Gradient of the entropy cost H(y)
            grady = gamy * np.dot(My, yk)
            # Gradient for H(yk|xk)
            grade = game * (np.dot(Be, ek) + ek)
            # Overall gradient
            gradv = -vk + grady - grade
            # Update v
            vk = vk + muv * gradv / np.linalg.norm(gradv) #/ (j+1)
            
            yk = ProjectOntoNNLInfty(vk / (h * gamy))
            if np.linalg.norm(yk - ykold) < tol * np.linalg.norm(ykold):
                break
            
        return yk, ek

    @staticmethod
    @njit
    def run_neural_dynamics_sparse(yk, yke, My, gamy, Be, game, vk, n_iterations = 250, lr_start = 1.5, lr_stop = 0.001, tol = 1e-8):
        # Hopefield parameter
        h = 1 / gamy
        STLAMBD = 0
        dval = 0
        for j in range(n_iterations):
            muv = max(lr_start/(1 + j), lr_stop)
            ykold = yk
            #Find error
            ek = yk - yke
            # Gradient of the entropy cost H(y)
            grady = gamy * np.dot(My, yk)
            # Gradient for H(yk|xk)
            grade = game * (np.dot(Be, ek) + ek)
            # Overall gradient
            gradv = -vk + grady - grade
            # Update v
            vk = vk + muv * gradv / np.linalg.norm(gradv) #/ (j+1)
            yk = (vk / (h * gamy))
            ## SOFT THRESHOLDING
            y_absolute = np.abs(yk)
            y_sign = np.sign(yk)
            yk = (y_absolute > STLAMBD) * (y_absolute - STLAMBD) * y_sign

            dval = np.linalg.norm(yk, 1) - 1
            STLAMBD = max(STLAMBD + 1 * dval, 0)

            if np.linalg.norm(yk - ykold) < tol * np.linalg.norm(ykold):
                break
            
        return yk, ek

    @staticmethod
    @njit
    def run_neural_dynamics_nnsparse(yk, yke, My, gamy, Be, game, vk, n_iterations = 250, lr_start = 1.5, lr_stop = 0.001, tol = 1e-8):
        # Hopefield parameter
        h = 1 / gamy
        STLAMBD = 0
        dval = 0
        for j in range(n_iterations):
            muv = max(lr_start/(1 + j), lr_stop)
            ykold = yk
            #Find error
            ek = yk - yke
            # Gradient of the entropy cost H(y)
            grady = gamy * np.dot(My, yk)
            # Gradient for H(yk|xk)
            grade = game * (np.dot(Be, ek) + ek)
            # Overall gradient
            gradv = -vk + grady - grade
            # Update v
            vk = vk + muv * gradv / np.linalg.norm(gradv) #/ (j+1)
            yk = np.maximum(vk / (h * gamy) - STLAMBD, 0)

            dval = np.sum(yk) - 1
            STLAMBD = max(STLAMBD + 1* dval, 0)
            
            if np.linalg.norm(yk - ykold) < tol * np.linalg.norm(ykold):
                break
            
        return yk, ek

    def fit_batch_antisparse(self, X, n_epochs, neural_dynamic_iterations = 250, neural_lr_start = 1.5, neural_lr_stop = 0.5, shuffle = True,verbose = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        s_dim, x_dim = self.s_dim, self.x_dim
        W, By, Be = self.W, self.By, self.Be
        muW = self.muW
        lambday, lambdae = self.lambday, self.lambdae
        gamy, game = self.gamy, self.game
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        h = 1 / gamy # Hopefield parameter

        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"
        samples = X.shape[1]

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            S = 2*self.ZeroOneNormalizeColumns(self.S.T).T-1
            A = self.A 

        for k in range(n_epochs):
            if shuffle:
                idx = np.random.permutation(samples)
            else:
                idx = np.arange(samples)
                
            for i_sample in tqdm(range(samples)):
                x_current  = X[:,idx[i_sample]] # Take one input
                xk = np.reshape(x_current, (x_dim, 1))
                # Initialize membrane voltage
                vk = np.zeros((s_dim, 1))
                # Initialize output   
                yk = np.random.uniform(-1.1,1.1, size = (s_dim,1))

                vk = h * yk * gamy
                yke = np.dot(W, xk)
                yk = yk/3
                # Output recurrent weights
                My = By + h * np.eye(s_dim)
                # NumberofOutputIterations_ = np.int64(5 + np.min([np.ceil(i_sample / 50000.0),neural_dynamic_iterations]))
                NumberofOutputIterations_ = neural_dynamic_iterations
                yk, ek = self.run_neural_dynamics_antisparse(yk, yke, My, gamy, Be, game, vk, 
                                                            NumberofOutputIterations_, neural_lr_start, neural_lr_stop, neural_OUTPUT_COMP_TOL)

                W = (1 - 1e-6) * W + muW * (np.dot(ek,xk.T))
                ee = np.dot(Be,ek)
                Be = 1 / lambdae * (Be - game * np.dot(ee, ee.T))

                zk = np.dot(By,yk)
                By = 1 / lambday * (By - gamy * np.dot(zk, zk.T))

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.Be = Be
                        self.By = By
                        self.SIR_list = SIR_list
                        self.SNR_list = SNR_list
                        self.compute_overall_mapping()

                        Wf = self.Wf
                        SIR,_ = self.CalculateSIR(A, Wf)
                        SIR_list.append(SIR)

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIR_list), linewidth = 3)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
                            pl.ylabel("SIR (dB)", fontsize = 15)
                            pl.title("SIR Behaviour", fontsize = 15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())   

            self.W = W
            self.Be = Be
            self.By = By
            self.SIR_list = SIR_list
            self.SNR_list = SNR_list

    def fit_batch_nnantisparse(self, X, n_epochs, neural_dynamic_iterations = 250, neural_lr_start = 1.5, neural_lr_stop = 0.5, shuffle = True,verbose = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        s_dim, x_dim = self.s_dim, self.x_dim
        W, By, Be = self.W, self.By, self.Be
        muW = self.muW
        lambday, lambdae = self.lambday, self.lambdae
        gamy, game = self.gamy, self.game
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        h = 1 / gamy # Hopefield parameter

        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"
        samples = X.shape[1]

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            S = 2*self.ZeroOneNormalizeColumns(self.S.T).T-1
            A = self.A 

        for k in range(n_epochs):
            if shuffle:
                idx = np.random.permutation(samples)
            else:
                idx = np.arange(samples)
                
            for i_sample in tqdm(range(samples)):
                x_current  = X[:,idx[i_sample]] # Take one input
                xk = np.reshape(x_current, (x_dim, 1))
                # Initialize membrane voltage
                vk = np.zeros((s_dim, 1))
                # Initialize output   
                yk = np.zeros((s_dim, 1))
                # yk = np.random.uniform(-1.1,1.1, size = (s_dim,1))

                # vk = h * yk * gamy
                yke = np.dot(W, xk)
                # yk = yk/3
                # Output recurrent weights
                My = By + h * np.eye(s_dim)
                # NumberofOutputIterations_ = np.int64(5 + np.min([np.ceil(i_sample / 50000.0),neural_dynamic_iterations]))
                NumberofOutputIterations_ = neural_dynamic_iterations
                yk, ek = self.run_neural_dynamics_nnantisparse(yk, yke, My, gamy, Be, game, vk, 
                                                            NumberofOutputIterations_, neural_lr_start, neural_lr_stop, neural_OUTPUT_COMP_TOL)

                W = (1 - 1e-6) * W + muW * (np.dot(ek,xk.T))
                ee = np.dot(Be,ek)
                Be = 1 / lambdae * (Be - game * np.dot(ee, ee.T))

                zk = np.dot(By,yk)
                By = 1 / lambday * (By - gamy * np.dot(zk, zk.T))

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.Be = Be
                        self.By = By
                        self.SIR_list = SIR_list
                        self.SNR_list = SNR_list
                        self.compute_overall_mapping()

                        Wf = self.Wf
                        SIR,_ = self.CalculateSIR(A, Wf)
                        SIR_list.append(SIR)

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIR_list), linewidth = 3)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
                            pl.ylabel("SIR (dB)", fontsize = 15)
                            pl.title("SIR Behaviour", fontsize = 15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())   

            self.W = W
            self.Be = Be
            self.By = By
            self.SIR_list = SIR_list
            self.SNR_list = SNR_list

    def fit_batch_sparse(self, X, n_epochs, neural_dynamic_iterations = 250, neural_lr_start = 1.5, neural_lr_stop = 0.5, shuffle = True,verbose = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        s_dim, x_dim = self.s_dim, self.x_dim
        W, By, Be = self.W, self.By, self.Be
        muW = self.muW
        lambday, lambdae = self.lambday, self.lambdae
        gamy, game = self.gamy, self.game
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        h = 1 / gamy # Hopefield parameter

        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"
        samples = X.shape[1]

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            S = self.S
            A = self.A 

        for k in range(n_epochs):
            if shuffle:
                idx = np.random.permutation(samples)
            else:
                idx = np.arange(samples)
                
            for i_sample in tqdm(range(samples)):
                x_current  = X[:,idx[i_sample]] # Take one input
                xk = np.reshape(x_current, (x_dim, 1))
                # Initialize output   
                yk = (2.0 * np.random.rand(s_dim,1) - 1.0)/5

                # Initialize membrane voltage
                vk = h * yk * gamy

                # yk = np.random.uniform(-1.1,1.1, size = (s_dim,1))

                # vk = h * yk * gamy
                yke = np.dot(W, xk)
                # yk = yk/3
                # Output recurrent weights
                My = By + h * np.eye(s_dim)
                # NumberofOutputIterations_ = np.int64(5 + np.min([np.ceil(i_sample / 50000.0),neural_dynamic_iterations]))
                NumberofOutputIterations_ = neural_dynamic_iterations
                yk, ek = self.run_neural_dynamics_sparse(yk, yke, My, gamy, Be, game, vk, 
                                                            NumberofOutputIterations_, neural_lr_start, neural_lr_stop, neural_OUTPUT_COMP_TOL)

                W = (1 - 1e-6) * W + muW * (np.dot(ek,xk.T))
                ee = np.dot(Be,ek)
                Be = 1 / lambdae * (Be - game * np.dot(ee, ee.T))

                zk = np.dot(By,yk)
                By = 1 / lambday * (By - gamy * np.dot(zk, zk.T))

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.Be = Be
                        self.By = By
                        self.SIR_list = SIR_list
                        self.SNR_list = SNR_list
                        self.compute_overall_mapping()

                        Wf = self.Wf
                        SIR,_ = self.CalculateSIR(A, Wf)
                        SIR_list.append(SIR)

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIR_list), linewidth = 3)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
                            pl.ylabel("SIR (dB)", fontsize = 15)
                            pl.title("SIR Behaviour", fontsize = 15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())   

            self.W = W
            self.Be = Be
            self.By = By
            self.SIR_list = SIR_list
            self.SNR_list = SNR_list


    def fit_batch_nnsparse(self, X, n_epochs, neural_dynamic_iterations = 250, neural_lr_start = 1.5, neural_lr_stop = 0.5, shuffle = True,verbose = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        s_dim, x_dim = self.s_dim, self.x_dim
        W, By, Be = self.W, self.By, self.Be
        muW = self.muW
        lambday, lambdae = self.lambday, self.lambdae
        gamy, game = self.gamy, self.game
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        h = 1 / gamy # Hopefield parameter

        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"
        samples = X.shape[1]

        if debugging:
            SIR_list = self.SIR_list
            SNR_list = self.SNR_list
            S = self.S
            A = self.A 

        for k in range(n_epochs):
            if shuffle:
                idx = np.random.permutation(samples)
            else:
                idx = np.arange(samples)
                
            for i_sample in tqdm(range(samples)):
                x_current  = X[:,idx[i_sample]] # Take one input
                xk = np.reshape(x_current, (x_dim, 1))
                # Initialize output   
                yk = np.zeros((s_dim, 1))

                # Initialize membrane voltage
                vk = np.zeros((s_dim, 1))

                # yk = np.random.uniform(-1.1,1.1, size = (s_dim,1))

                # vk = h * yk * gamy
                yke = np.dot(W, xk)
                # yk = yk/3
                # Output recurrent weights
                My = By + h * np.eye(s_dim)
                # NumberofOutputIterations_ = np.int64(5 + np.min([np.ceil(i_sample / 50000.0),neural_dynamic_iterations]))
                NumberofOutputIterations_ = neural_dynamic_iterations
                yk, ek = self.run_neural_dynamics_nnsparse(yk, yke, My, gamy, Be, game, vk, 
                                                            NumberofOutputIterations_, neural_lr_start, neural_lr_stop, neural_OUTPUT_COMP_TOL)

                W = (1 - 1e-6) * W + muW * (np.dot(ek,xk.T))
                ee = np.dot(Be,ek)
                Be = 1 / lambdae * (Be - game * np.dot(ee, ee.T))

                zk = np.dot(By,yk)
                By = 1 / lambday * (By - gamy * np.dot(zk, zk.T))

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.Be = Be
                        self.By = By
                        self.SIR_list = SIR_list
                        self.SNR_list = SNR_list
                        self.compute_overall_mapping()

                        Wf = self.Wf
                        SIR,_ = self.CalculateSIR(A, Wf)
                        SIR_list.append(SIR)

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIR_list), linewidth = 3)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
                            pl.ylabel("SIR (dB)", fontsize = 15)
                            pl.title("SIR Behaviour", fontsize = 15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())   

            self.W = W
            self.Be = Be
            self.By = By
            self.SIR_list = SIR_list
            self.SNR_list = SNR_list
    # def fit_batch_nnantisparse(self, X, n_epochs, neural_dynamic_iterations = 250, neural_lr_start = 1.5, neural_lr_stop = 0.5, shuffle = True,verbose = True, debug_iteration_point = 1000, plot_in_jupyter = False):
    #     s_dim, x_dim = self.s_dim, self.x_dim
    #     W, By, Be = self.W, self.By, self.Be
    #     muW = self.muW
    #     lambday, lambdae = self.lambday, self.lambdae
    #     gamy, game = self.gamy, self.game
    #     neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
    #     debugging = self.set_ground_truth

    #     h = 1 / gamy # Hopefield parameter
    #     mx = X.mean(axis = 1)

    #     assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"
    #     samples = X.shape[1]

    #     if debugging:
    #         SIR_list = self.SIR_list
    #         SNR_list = self.SNR_list
    #         S = 2*self.ZeroOneNormalizeColumns(self.S.T).T-1
    #         A = self.A 

    #     for k in range(n_epochs):
    #         if shuffle:
    #             idx = np.random.permutation(samples)
    #         else:
    #             idx = np.arange(samples)
                
    #         for i_sample in tqdm(range(samples)):
    #             x_current  = X[:,idx[i_sample]] - mx # Take one input
    #             xk = np.reshape(x_current, (x_dim, 1))
    #             # Initialize membrane voltage
    #             vk = np.zeros((s_dim, 1))
    #             # Initialize output   
    #             yk = np.random.uniform(-1.1,1.1, size = (s_dim,1))

    #             vk = h * yk * gamy
    #             yke = np.dot(W, xk)
    #             yk = yk/3
    #             # Output recurrent weights
    #             My = By + h * np.eye(s_dim)
    #             NumberofOutputIterations_ = np.int64(5 + np.min([np.ceil(i_sample / 50000.0),neural_dynamic_iterations]))
    #             yk, ek = self.run_neural_dynamics_antisparse(yk, yke, My, gamy, Be, game, vk,
    #                                                          NumberofOutputIterations_, neural_lr_start, neural_lr_stop, neural_OUTPUT_COMP_TOL)

    #             W = (1 - 1e-6) * W + muW * (np.dot(ek,xk.T))
    #             ee = np.dot(Be,ek)
    #             Be = 1 / lambdae * (Be - game * np.dot(ee, ee.T))

    #             zk = np.dot(By,yk)
    #             By = 1 / lambday * (By - gamy * np.dot(zk, zk.T))

    #             if debugging:
    #                 if (i_sample % debug_iteration_point) == 0:
    #                     self.W = W
    #                     self.Be = Be
    #                     self.By = By
    #                     self.SIR_list = SIR_list
    #                     self.SNR_list = SNR_list
    #                     self.compute_overall_mapping()

    #                     Wf = self.Wf
    #                     SIR,_ = CalculateSIR(A, Wf)
    #                     SIR_list.append(SIR)

    #                     if plot_convergence_plot:
    #                         pl.clf()
    #                         pl.plot(np.array(SIR_list), linewidth = 3)
    #                         pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
    #                         pl.ylabel("SIR (dB)", fontsize = 15)
    #                         pl.title("SIR Behaviour", fontsize = 15)
    #                         pl.grid()
    #                         clear_output(wait=True)
    #                         display(pl.gcf())   
    #         #         break # WILL BE DELETED
    #         #     break # WILL BE DELETED
    #         # break # WILL BE DELETED

    #         self.W = W
    #         self.Be = Be
    #         self.By = By
    #         self.SIR_list = SIR_list
    #         self.SNR_list = SNR_list



# class OnlineWhiten1:

#     def __init__(self, s_dim, x_dim, W = None, B = None, Ro = None, muW = 1e-2/2, p = 0.3, bet = 0.1/5, lambd = 1 - 5e-3, neural_OUTPUT_COMP_TOL = 1e-6):
#         if W is not None:
#             assert W.shape == (x_dim, s_dim), "The shape of the initial guess W must be (x_dim, s_dim) = (%d, %d)" % (x_dim, s_dim)
#             W = W
#         else:
#             W = np.random.randn(x_dim, s_dim)/x_dim/4
#             # for k in range(W.shape[0]):
#             #     W[k,:] =  W[k,:]/np.linalg.norm(W[k,:])

#         if B is not None:
#             assert B.shape == (s_dim, s_dim), "The shape of the initial guess B must be (s_dim, s_dim) = (%d, %d)" % (s_dim, s_dim)
#             B = B
#         else:
#             B = 10 * np.eye(s_dim, s_dim)

#         if Ro is not None:
#             assert Ro.shape == (s_dim, s_dim), "The shape of the initial guess Ro must be (s_dim, s_dim) = (%d, %d)" % (s_dim, s_dim)
#             Ro = Ro
#         else:
#             Ro = 0.1 * np.eye(s_dim, s_dim)

#         self.s_dim = s_dim
#         self.x_dim = x_dim
#         self.W  = W
#         self.B = B
#         self.Ro = Ro
#         self.muW = muW
#         self.p = p 
#         self.bet = bet
#         self.lambd = lambd
#         self.gam = (1 - lambd) / lambd
#         self.mus = np.zeros((s_dim, 1))
#         self.neural_OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL
#         self.WhiteFOM = []
        
#     def ProjectOntoLInfty(self, X, thresh = 1):
#         return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)

#     def compute_overall_mapping(self):
#         # bet = self.bet
#         # p = self.p
#         # lambd = self.lambd
#         # s_dim = self.s_dim
#         # gam = self.gam
#         # W = self.W
#         # B = self.B
#         # We=np.linalg.pinv((bet+(1-lambd)/p/s_dim)*np.eye(s_dim)-gam/s_dim*B)@ W*bet
#         return self.W

#     def whiten_transform(self, X):
#         assert X.shape[0] == self.W.shape[1], "Matrix dimensions do not match for whitening matrix W and input signal X"
#         return self.W @ X

#     @staticmethod
#     @njit
#     def run_neural_dynamics(sk, xk, Wf, B, mus, p, bet, lambd, gam, n_iterations = 1550, neural_lr_start = 5, neural_lr_stop = 1e-2, tol = 1e-6):
        
#         def ProjectOntoLInfty(X, thresh):
#             return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)
        
#         s_dim = sk.shape[0]
#         for i in range(n_iterations):
#             muv = max([neural_lr_start/np.sqrt(i+1),neural_lr_stop])
#             skold = sk
#             skb = sk - mus
#             ek = Wf @ sk - xk

#             gradJ = bet * Wf.T @ ek + (1 - lambd) / p * skb / s_dim - gam / s_dim * B @ skb

#             sk = sk - muv * gradJ / (np.max(np.abs(gradJ)))
#             sk = ProjectOntoLInfty(sk, 1)

#             if np.linalg.norm(sk - skold) < tol * np.linalg.norm(skold):
#                 break

#         return sk, skb, ek

#     def fit_batch_whiten(self, X, n_epochs = 1, neural_dynamic_iterations = 1550, neural_lr_start = 5, neural_lr_stop = 1e-2, shuffle = True, required_SIR = 35, debug_iteration_point = 1000, plot_in_jupyter = False):
#         s_dim, x_dim = self.s_dim, self.x_dim
#         W, B, Ro = self.W, self.B, self.Ro
#         muW, p, bet = self.muW, self.p, self.bet
#         lambd, gam = self.lambd, self.gam
#         mus = self.mus
#         neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
#         WhiteFOM = self.WhiteFOM

#         assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"
#         samples = X.shape[1]

#         for k in range(n_epochs):
#             # bet = self.bet
#             if shuffle:
#                 idx = np.random.permutation(samples)
#             else:
#                 idx = np.arange(samples)
                
#             for i_sample in tqdm(range(samples)):
#                 bet = 0.1/5*(1+0.1*np.log10((i_sample + k * samples)+1))
#                 self.bet = bet
#                 x_current  = X[:,idx[i_sample]] # Take one input
#                 xk = np.reshape(x_current, (x_dim, 1))

#                 sk = np.random.randn(s_dim, 1) / 10

#                 sk, skb, ek = self.run_neural_dynamics(sk, xk, W, B, mus, p, bet, lambd, gam, neural_dynamic_iterations, neural_lr_start, neural_lr_stop, neural_OUTPUT_COMP_TOL)

#                 p = lambd * p + (1- lambd) * np.dot(skb.T, skb)/s_dim
#                 W = W + muW * np.dot(ek, sk.T) / np.sqrt(i_sample + 1)
#                 zk = np.dot(B, skb)

#                 B = 1/lambd * (B - gam * np.dot(zk, zk.T))
#                 Ro = lambd * Ro + (1 - lambd) * np.dot(skb, skb.T)

#                 if np.mod(i_sample, debug_iteration_point) == 0:

#                     Rz = Ro.copy()
#                     Rzd = np.linalg.norm(np.diag(Rz)) ** 2
#                     Rzo = np.linalg.norm(Rz, 'fro') ** 2 - Rzd
#                     SIR = 10 * np.log10(Rzd / Rzo)
#                     WhiteFOM.append(SIR)
                    
#                     self.W = W
#                     self.B = B
#                     self.Ro = Ro
#                     self.WhiteFOM = WhiteFOM

#                     if plot_in_jupyter:
#                         pl.clf()
#                         pl.subplot(1,1,1)
#                         pl.plot(np.array(WhiteFOM))
#                         pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
#                         pl.ylabel("SIR (dB)", fontsize = 15)
#                         pl.title("SIR Behaviour", fontsize = 15)
#                         pl.grid()
#                         display.clear_output(wait=True)
#                         display.display(pl.gcf())
                    
#                 if WhiteFOM[-1] > required_SIR:
#                     break

class OnlineWhiten:

    def __init__(self, s_dim, x_dim, W = None, Winv = None, B = None, Ro = None, muW = 1e-2/2, p = 0.3, bet = 0.1/5, lambd = 1 - 5e-3, neural_OUTPUT_COMP_TOL = 1e-6):
        if W is not None:
            assert W.shape == (x_dim, s_dim), "The shape of the initial guess W must be (x_dim, s_dim) = (%d, %d)" % (x_dim, s_dim)
            W = W
        else:
            W = np.random.randn(x_dim, s_dim)/x_dim/4
            # for k in range(W.shape[0]):
            #     W[k,:] =  W[k,:]/np.linalg.norm(W[k,:])

        if Winv is not None:
            assert Winv.shape == (s_dim, x_dim), "The shape of the initial guess Winv must be (s_dim, x_dim) = (%d, %d)" % (s_dim, x_dim)
            Winv = Winv
        else:
            Winv = np.random.randn(s_dim,x_dim)/x_dim

        if B is not None:
            assert B.shape == (s_dim, s_dim), "The shape of the initial guess B must be (s_dim, s_dim) = (%d, %d)" % (s_dim, s_dim)
            B = B
        else:
            B = 10 * np.eye(s_dim, s_dim)

        if Ro is not None:
            assert Ro.shape == (s_dim, s_dim), "The shape of the initial guess Ro must be (s_dim, s_dim) = (%d, %d)" % (s_dim, s_dim)
            Ro = Ro
        else:
            Ro = 0.1 * np.eye(s_dim, s_dim) + 0.1 * np.random.randn(s_dim, s_dim)

        self.s_dim = s_dim
        self.x_dim = x_dim
        self.W  = W
        self.Winv = Winv
        self.B = B
        self.Ro = Ro
        self.muW = muW
        self.p = p 
        self.bet = bet
        self.lambd = lambd
        self.gam = (1 - lambd) / lambd
        self.mus = np.zeros((s_dim, 1))
        self.neural_OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL
        self.WhiteFOM = []
        self.Rxx = 0.01 * np.random.randn(x_dim, x_dim)
        
    def ProjectOntoLInfty(self, X, thresh = 1):
        return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)

    def compute_overall_mapping(self):
        bet = self.bet
        p = self.p
        lambd = self.lambd
        s_dim = self.s_dim
        gam = self.gam
        W = self.W
        B = self.B

        We = bet * np.linalg.pinv(bet * W.T @ W + ((1 - lambd) / s_dim * p) * np.eye(s_dim, s_dim) - (gam / s_dim) * B) @ W.T
        return We

    def whiten_transform(self, X):
        assert X.shape[0] == self.W.shape[1], "Matrix dimensions do not match for whitening matrix W and input signal X"
        return self.W @ X

    @staticmethod
    @njit
    def run_neural_dynamics(sk, xk, Wf, B, mus, p, bet, lambd, gam, n_iterations = 1550, neural_lr_start = 5, neural_lr_stop = 1e-2, tol = 1e-6):
        
        def ProjectOntoLInfty(X, thresh):
            return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)
        
        s_dim = sk.shape[0]
        for i in range(n_iterations):
            muv = max([neural_lr_start/np.sqrt(i+1),neural_lr_stop])
            skold = sk
            skb = sk - mus
            ek = sk - np.dot(Wf, xk)#Wf @ xk

            gradJ = bet * ek + (1 - lambd) / p * skb / s_dim - gam / s_dim * np.dot(B, skb)#B @ skb

            sk = sk - muv * gradJ / (np.max(np.abs(gradJ)))
            sk = ProjectOntoLInfty(sk, 1)

            if np.linalg.norm(sk - skold) < tol * np.linalg.norm(skold):
                break

        return sk, skb, ek

    def fit_batch_whiten(self, X, n_epochs = 1, neural_dynamic_iterations = 1550, neural_lr_start = 5, neural_lr_stop = 1e-2, shuffle = True, required_SIR = 35, debug_iteration_point = 1000, plot_in_jupyter = False):
        Rxx = self.Rxx
        s_dim, x_dim = self.s_dim, self.x_dim
        W, B, Ro, Winv = self.W, self.B, self.Ro, self.Winv
        muW, p, bet = self.muW, self.p, self.bet
        lambd, gam = self.lambd, self.gam
        mus = self.mus
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        WhiteFOM = self.WhiteFOM

        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"
        samples = X.shape[1]

        for k in range(n_epochs):
            # bet = self.bet
            if shuffle:
                idx = np.random.permutation(samples)
            else:
                idx = np.arange(samples)
                
            for i_sample in tqdm(range(samples)):
                bet = 0.1/5*(1+0.1*np.log10((i_sample + k * samples)+1))
                self.bet = bet
                x_current  = X[:,idx[i_sample]] # Take one input
                xk = np.reshape(x_current, (x_dim, 1))

                sk = np.random.randn(s_dim, 1) / 10

                sk, skb, ek = self.run_neural_dynamics(sk, xk, Winv, B, mus, p, bet, lambd, gam, neural_dynamic_iterations, neural_lr_start, neural_lr_stop, neural_OUTPUT_COMP_TOL)

                p = lambd * p + (1- lambd) * np.dot(skb.T, skb)/s_dim
                # W = W + muW * np.dot(ek, sk.T) / np.sqrt(i_sample + 1)
                # print(np.dot(ek, xk.T).shape)
                # break
                # Winv = Winv + muW * np.dot((sk - Winv @ xk), xk.T) / np.sqrt(i_sample + 1)
                Winv = Winv + muW * np.dot(ek, xk.T) / np.sqrt(i_sample + 1)
                zk = np.dot(B, skb)

                B = 1/lambd * (B - gam * np.dot(zk, zk.T))
                Ro = lambd * Ro + (1 - lambd) * np.dot(skb, skb.T)
                Rxx = lambd * Rxx + (1 - lambd) * np.dot(xk, xk.T)
                mus = lambd * mus + (1 - lambd) * sk
                if np.mod(i_sample, debug_iteration_point) == 0:
                    We=np.linalg.pinv((bet+(1-lambd)/p/s_dim)*np.eye(s_dim)-gam/s_dim*B)@Winv*bet
                    self.We = We
                    Rz=We@Rxx@We.T
                    Rzd = np.linalg.norm(np.diag(Rz)) ** 2
                    Rzo = np.linalg.norm(Rz, 'fro') ** 2 - Rzd
                    SIR = 10 * np.log10(Rzd / Rzo)
                    WhiteFOM.append(SIR)
                    
                    self.W = W
                    self.Winv = Winv
                    self.B = B
                    self.Ro = Ro
                    self.Rxx = Rxx
                    self.mus = mus
                    self.WhiteFOM = WhiteFOM

                    if plot_in_jupyter:
                        pl.clf()
                        pl.subplot(1,1,1)
                        pl.plot(np.array(WhiteFOM))
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
                        pl.ylabel("SIR (dB)", fontsize = 15)
                        pl.title("SIR Behaviour", fontsize = 15)
                        pl.grid()
                        display1.clear_output(wait=True)
                        display1.display(pl.gcf())
                    
                if WhiteFOM[-1] > required_SIR:
                    break
                
    def forward(self, X, n_epochs = 1, neural_dynamic_iterations = 1550, neural_lr_start = 5, neural_lr_stop = 1e-2, shuffle = True, required_SIR = 35, debug_iteration_point = 1000, plot_in_jupyter = False):
        s_dim, x_dim = self.s_dim, self.x_dim
        W, B, Ro, Winv = self.W, self.B, self.Ro, self.Winv
        muW, p, bet = self.muW, self.p, self.bet
        lambd, gam = self.lambd, self.gam
        mus = self.mus
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL


        assert X.shape[0] == self.x_dim, "You must input the transpose, or you need to change one of the following hyperparameters: s_dim, x_dim"
        samples = X.shape[1]

        S = np.zeros((s_dim, samples))
        for k in range(n_epochs):
            # bet = self.bet
            if shuffle:
                idx = np.random.permutation(samples)
            else:
                idx = np.arange(samples)
                
            for i_sample in tqdm(range(samples)):
                x_current  = X[:,idx[i_sample]] # Take one input
                xk = np.reshape(x_current, (x_dim, 1))

                sk = np.random.randn(s_dim, 1) / 10

                sk, skb, ek = self.run_neural_dynamics(sk, xk, W, B, mus, p, bet, lambd, gam, neural_dynamic_iterations, neural_lr_start, neural_lr_stop, neural_OUTPUT_COMP_TOL)

                S[:, idx[i_sample]] = sk.reshape(-1,)

        return S


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

