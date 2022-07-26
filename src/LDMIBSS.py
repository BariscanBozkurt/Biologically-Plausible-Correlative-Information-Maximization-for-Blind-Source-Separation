"""
Title: LDMIBSS.py

Two Layer Recurrent Neural Network for Blind Source Separation

Code Writer: Bariscan Bozkurt (Koç University - EEE & Mathematics)

Date: 17.02.2022
"""

from random import sample
from telnetlib import XAUTH
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
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull  
from numpy.linalg import det
from scipy.stats import dirichlet
import itertools
import pypoman
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
    
    def __init__(self, s_dim, x_dim, lambday = 0.999, lambdae = 1, muW = 1e-3, epsilon = 1e-3, W = None, By = None, Be = None, neural_OUTPUT_COMP_TOL = 1e-6, set_ground_truth = False, S = None, A = None):
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
        self.muW = muW
        self.gamy = (1-lambday) / lambday
        self.game = (1 - lambdae) / lambdae
        self.epsilon = epsilon
        self.W = W
        self.By = By
        self.Be = Be
        self.neural_OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL
        ### Ground Truth Sources and Mixing Matrix For Debugging
        self.set_ground_truth = set_ground_truth
        self.S = S # Sources
        self.A = A # Mixing Matrix
        self.SIR_list = []
        self.SINR_list = []
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

    def CalculateSINR(self, Out, S, compute_permutation = True):
        r=S.shape[0]
        if compute_permutation:
            G=np.dot(Out-np.reshape(np.mean(Out,1),(r,1)),np.linalg.pinv(S-np.reshape(np.mean(S,1),(r,1))))
            indmax=np.argmax(np.abs(G),1)
        else:
            G=np.dot(Out-np.reshape(np.mean(Out,1),(r,1)),np.linalg.pinv(S-np.reshape(np.mean(S,1),(r,1))))
            indmax = np.arange(0,r)
        GG=np.zeros((r,r))
        for kk in range(r):
            GG[kk,indmax[kk]]=np.dot(Out[kk,:]-np.mean(Out[kk,:]),S[indmax[kk],:].T-np.mean(S[indmax[kk],:]))/np.dot(S[indmax[kk],:]-np.mean(S[indmax[kk],:]),S[indmax[kk],:].T-np.mean(S[indmax[kk],:]))#(G[kk,indmax[kk]])
        ZZ=GG@(S-np.reshape(np.mean(S,1),(r,1)))+np.reshape(np.mean(Out,1),(r,1))
        E=Out-ZZ
        MSE=np.linalg.norm(E,'fro')**2
        SigPow=np.linalg.norm(ZZ,'fro')**2
        SINR=(SigPow/MSE)
        return SINR,SigPow,MSE,G

    def snr(self, S_original, S_noisy):
        N_hat = S_original - S_noisy
        N_P = (N_hat ** 2).sum(axis = 0)
        S_P = (S_original ** 2).sum(axis = 0)
        snr = 10 * np.log10(S_P / N_P)
        return snr

    def compute_overall_mapping(self, return_mapping = False):
        W, By, Be, gamy, game = self.W, self.By, self.Be, self.gamy, self.game
        # Wf = np.linalg.pinv(gamy * By - game * Be - beta * np.eye(self.s_dim)) @ (game * Be @ W + beta * W)
        Wf = np.linalg.pinv(gamy * By - game * Be) @ (game * Be @ W)
        if return_mapping:
            return Wf
        else:
            return None

    def outer_prod_broadcasting(self, A, B):
        """Broadcasting trick"""
        return A[...,None]*B[:,None]

    def find_permutation_between_source_and_estimation(self, S, Y):
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

    def signed_and_permutation_corrected_sources(self, S, Y):
        perm = self.find_permutation_between_source_and_estimation(S,Y)
        return np.sign((Y[:,perm] * S).sum(axis = 0)) * Y[:,perm]

    def ProjectOntoLInfty(self, X):
        
        return X*(X>=-1.0)*(X<=1.0)+(X>1.0)*1.0-1.0*(X<-1.0)
    
    @staticmethod
    @njit
    def run_neural_dynamics_antisparse(x, y, W, My, Be, gamy, game, lr_start = 0.9, lr_stop = 1e-15, neural_dynamic_iterations = 100, neural_OUTPUT_COMP_TOL = 1e-7):
        def ProjectOntoLInfty(X, thresh = 1.0):
            return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)

        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            mu_y = max(lr_start / (j + 1), lr_stop)
            # mu_y = lr_start / (j + 1)
            y_old = y.copy()
            e = yke - y
            grady = -y + gamy * My @ y + game * Be @ e
            y = y + mu_y * (grady)
            y = ProjectOntoLInfty(y)

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    @staticmethod
    @njit
    def run_neural_dynamics_nnantisparse(x, y, W, My, Be, gamy, game, lr_start = 0.9, lr_stop = 1e-15, neural_dynamic_iterations = 100, neural_OUTPUT_COMP_TOL = 1e-7):
        def ProjectOntoNNLInfty(X, thresh = 1.0):
            return X*(X>=0.0)*(X<=thresh)+(X>thresh)*thresh #-thresh*(X<-thresh)

        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            mu_y = max(lr_start / (j + 1), lr_stop)
            # mu_y = lr_start / (j + 1)
            y_old = y.copy()
            e = yke - y
            grady = -y + gamy * My @ y + game * Be @ e
            y = y + mu_y * (grady)
            y = ProjectOntoNNLInfty(y)

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    @staticmethod
    @njit
    def run_neural_dynamics_mixedantisparse(x, y, nn_components, signed_components, W, My, Be, gamy, game, lr_start = 0.9, lr_stop = 1e-15, neural_dynamic_iterations = 100, neural_OUTPUT_COMP_TOL = 1e-7):
        def ProjectOntoLInfty(X, thresh = 1.0):
            return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)
        def ProjectOntoNNLInfty(X, thresh = 1.0):
            return X*(X>=0.0)*(X<=thresh)+(X>thresh)*thresh #-thresh*(X<-thresh)

        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            mu_y = max(lr_start / (j + 1), lr_stop)
            # mu_y = lr_start / (j + 1)
            y_old = y.copy()
            e = yke - y
            grady = -y + gamy * My @ y + game * Be @ e
            y = y + mu_y * (grady)
            y[signed_components] = ProjectOntoLInfty(y[signed_components])
            y[nn_components] = ProjectOntoNNLInfty(y[nn_components])

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    @staticmethod
    @njit
    def run_neural_dynamics_sparse(x, y, W, My, Be, gamy, game, lr_start = 0.9, lr_stop = 1e-15, neural_dynamic_iterations = 100, neural_OUTPUT_COMP_TOL = 1e-7):
        
        STLAMBD = 0
        dval = 0
        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            mu_y = max(lr_start / (j + 1), lr_stop)
            # mu_y = lr_start / (j + 1)
            y_old = y.copy()
            e = yke - y
            grady = -y + gamy * My @ y + game * Be @ e
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
    def run_neural_dynamics_nnsparse(x, y, W, My, Be, gamy, game, lr_start = 0.9, lr_stop = 1e-15, neural_dynamic_iterations = 100, neural_OUTPUT_COMP_TOL = 1e-7):
        
        STLAMBD = 0
        dval = 0
        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            mu_y = max(lr_start / (j + 1), lr_stop)
            # mu_y = lr_start / (j + 1)
            y_old = y.copy()
            e = yke - y
            grady = -y + gamy * My @ y + game * Be @ e 
            y = y + mu_y * (grady)

            y = np.maximum(y - STLAMBD, 0)

            dval = np.sum(y) - 1
            STLAMBD = max(STLAMBD + 1 * dval, 0)

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    @staticmethod
    @njit
    def run_neural_dynamics_simplex(x, y, W, My, Be, gamy, game, lr_start = 0.9, lr_stop = 1e-15, neural_dynamic_iterations = 100, neural_OUTPUT_COMP_TOL = 1e-7):
        
        STLAMBD = 0
        dval = 0
        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            mu_y = max(lr_start / (j + 1), lr_stop)
            # mu_y = lr_start / (j + 1)
            y_old = y.copy()
            e = yke - y
            grady = -y + gamy * My @ y + game * Be @ e
            y = y + mu_y * (grady)

            y = np.maximum(y - STLAMBD, 0)

            dval = np.sum(y) - 1
            STLAMBD = STLAMBD + 0.05 * dval

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    @staticmethod
    @njit
    def run_neural_dynamics_nnwsubsparse(x, y, nn_components, sparse_components, W, My, Be, gamy, game, lr_start = 0.9, lr_stop = 1e-15, neural_dynamic_iterations = 100, neural_OUTPUT_COMP_TOL = 1e-7):
        def ProjectOntoLInfty(X, thresh = 1.0):
            return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)
        def ProjectOntoNNLInfty(X, thresh = 1.0):
            return X*(X>=0.0)*(X<=thresh)+(X>thresh)*thresh #-thresh*(X<-thresh)
        STLAMBD = 0
        dval = 0
        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            mu_y = max(lr_start / (j + 1), lr_stop)
            # mu_y = lr_start / (j + 1)
            y_old = y.copy()
            e = yke - y
            grady = -y + gamy * My @ y + game * Be @ e 
            y = y + mu_y * (grady)

            y[nn_components] = ProjectOntoNNLInfty(y[nn_components])
            # SOFT THRESHOLDING
            y_sparse_absolute = np.abs(y[sparse_components])
            y_sparse_sign = np.sign(y[sparse_components])

            y[sparse_components] = (y_sparse_absolute > STLAMBD) * (y_sparse_absolute - STLAMBD) * y_sparse_sign
            y = ProjectOntoLInfty(y)
            dval = np.linalg.norm(y[sparse_components], 1) - 1
            STLAMBD = max(STLAMBD + 1 * dval, 0)

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    @staticmethod
    @njit
    def run_neural_dynamics_nnwsubnnsparse(x, y, nn_components, nnsparse_components, W, My, Be, gamy, game, lr_start = 0.9, lr_stop = 1e-15, neural_dynamic_iterations = 100, neural_OUTPUT_COMP_TOL = 1e-7):

        def ProjectOntoNNLInfty(X, thresh = 1.0):
            return X*(X>=0.0)*(X<=thresh)+(X>thresh)*thresh 

        STLAMBD = 0
        dval = 0
        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            mu_y = max(lr_start / (j + 1), lr_stop)
            # mu_y = lr_start / (j + 1)
            y_old = y.copy()
            e = yke - y
            grady = -y + gamy * My @ y + game * Be @ e
            y = y + mu_y * (grady)

            y[nn_components] = ProjectOntoNNLInfty(y[nn_components])

            y[nnsparse_components] = np.maximum(y[nnsparse_components] - STLAMBD, 0)

            dval = np.sum(y[nnsparse_components]) - 1
            STLAMBD = max(STLAMBD + 1 * dval, 0)
            y = ProjectOntoNNLInfty(y)

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    @staticmethod
    @njit
    def run_neural_dynamics_general_polytope(x, y, signed_dims, nn_dims, sparse_dims_list, W, My, Be, gamy, game, lr_start = 0.9, 
                                             lr_stop = 1e-15, neural_dynamic_iterations = 100, neural_OUTPUT_COMP_TOL = 1e-7):
        def ProjectOntoLInfty(X, thresh = 1.0):
            return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)
        def ProjectOntoNNLInfty(X, thresh = 1.0):
            return X*(X>=0.0)*(X<=thresh)+(X>thresh)*thresh
        def SoftThresholding(X, thresh):
            X_absolute = np.abs(X)
            X_sign = np.sign(X)
            X_thresholded = (X_absolute > thresh) * (X_absolute - thresh) * X_sign
            return X_thresholded
        def ReLU(X):
            return np.maximum(X,0)

        def loop_intersection(lst1, lst2):
            result = []
            for element1 in lst1:
                for element2 in lst2:
                    if element1 == element2:
                        result.append(element1)
            return result

        STLAMBD_list = np.zeros(len(sparse_dims_list))
        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            mu_y = max(lr_start / (j + 1), lr_stop)
            y_old = y.copy()
            e = yke - y
            grady = -y + gamy * My @ y + game * Be @ e
            y = y + mu_y * (grady)
            if sparse_dims_list[0][0] != -1:
                for ss,sparse_dim in enumerate(sparse_dims_list):
                    # y[sparse_dim] = SoftThresholding(y[sparse_dim], STLAMBD_list[ss])
                    # STLAMBD_list[ss] = max(STLAMBD_list[ss] + (np.linalg.norm(y[sparse_dim],1) - 1), 0)
                    if signed_dims[0] != -1:
                        y[np.array(loop_intersection(sparse_dim, signed_dims))] = SoftThresholding(y[np.array(loop_intersection(sparse_dim, signed_dims))], STLAMBD_list[ss])
                    if nn_dims[0] != -1:
                        y[np.array(loop_intersection(sparse_dim, nn_dims))] = ReLU(y[np.array(loop_intersection(sparse_dim, nn_dims))] - STLAMBD_list[ss])
                    STLAMBD_list[ss] = max(STLAMBD_list[ss] + (np.linalg.norm(y[sparse_dim],1) - 1), 0)
            if signed_dims[0] != -1:
                y[signed_dims] = ProjectOntoLInfty(y[signed_dims])
            if nn_dims[0] != -1:
                y[nn_dims] = ProjectOntoNNLInfty(y[nn_dims])

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    def fit_batch_general_polytope(self, X, signed_dims, nn_dims, sparse_dims_list, n_epochs = 1, neural_dynamic_iterations = 250, neural_lr_start = 0.9, neural_lr_stop = 1e-15, shuffle = False, debug_iteration_point = 1000, plot_in_jupyter = False):
        
        lambday, lambdae, muW, gamy, game, W, By, Be = self.lambday, self.lambdae, self.muW, self.gamy, self.game, self.W, self.By, self.Be
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        h = 1 / gamy # Hopefield parameter

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            plt.figure(figsize = (25, 10), dpi = 80)

        # Y = np.zeros((self.s_dim, samples))
        Y = np.random.randn(self.s_dim, samples)
        
        
        if shuffle:
            idx = np.random.permutation(samples) # random permutation
        else:
            idx = np.arange(samples)
        
        if (signed_dims.size == 0):
            signed_dims = np.array([-1])
        if (nn_dims.size == 0):
            nn_dims = np.array([-1])
        if (not sparse_dims_list):
            sparse_dims_list = [np.array([-1])]

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:,idx[i_sample]]
                y = np.zeros(self.s_dim)

                # Output recurrent weights
                My = By + h * np.eye(self.s_dim)
                y = self.run_neural_dynamics_general_polytope(  x_current, y, signed_dims, nn_dims, sparse_dims_list, W, My, Be, gamy, game, 
                                                                lr_start = neural_lr_start, lr_stop = neural_lr_stop,
                                                                neural_dynamic_iterations = neural_dynamic_iterations, 
                                                                neural_OUTPUT_COMP_TOL = neural_dynamic_tol)
                        
                e = y - W @ x_current

                W = W + muW * np.outer(e, x_current)
                
                z = By @ y
                By = (1/lambday) * (By - gamy * np.outer(z, z))

                # ee = np.dot(Be,e)
                # Be = 1 / lambdae * (Be - game * np.outer(ee, ee))
                # Record the seperated signal
                Y[:,idx[i_sample]] = y

                if debugging:
                    if (((i_sample % debug_iteration_point) == 0) | (i_sample == samples - 1)):
                        self.W = W
                        self.By = By
                        self.Be = Be
                        Wf = self.compute_overall_mapping(return_mapping = True)
                        Y_ = Wf @ X
                        Y_ = self.signed_and_permutation_corrected_sources(S.T,Y_.T)
                        coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                        Y_ = coef_ * Y_
                        SIR = self.CalculateSIR(A, Wf)[0]
                        SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                        SIRlist.append(SIR)
                        SINRlist.append(SINR)
                        SNRlist.append(self.snr(S.T,Y_))
                        self.SIR_list = SIRlist
                        self.SINR_list = SINRlist
                        self.SNR_list = SNRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.subplot(1,2,1)
                            pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                            pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                            pl.ylabel("SINR (dB)", fontsize = 35)
                            pl.title("SINR Behaviour", fontsize = 35)
                            pl.xticks(fontsize=45)
                            pl.yticks(fontsize=45)
                            pl.legend(fontsize=25)
                            pl.grid()
                            pl.subplot(1,2,2)
                            pl.plot(np.array(SNRlist), linewidth = 3)
                            pl.grid()
                            pl.title("Component SNR Check", fontsize = 35)
                            pl.ylabel("SNR (dB)", fontsize = 35)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                            pl.xticks(fontsize=45)
                            pl.yticks(fontsize=45)
                            clear_output(wait=True)
                            display(pl.gcf())  
        self.W = W
        self.By = By
        self.Be = Be
        
    def fit_next_antisparse(self,x_current, neural_dynamic_iterations = 250, lr_start = 0.9):
        
        lambday, lambdae, muW, gamy, game, W, By, Be = self.lambday, self.lambdae, self.muW, self.gamy, self.game, self.W, self.By, self.Be
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        h = 1 / gamy # Hopefield parameter

        y = np.zeros(self.s_dim)

        # Output recurrent weights
        My = By + h * np.eye(self.s_dim)
        
        y = self.run_neural_dynamics_antisparse(x_current, y, W, My, Be, gamy, game, 
                                                lr_start = lr_start, neural_dynamic_iterations = neural_dynamic_iterations, 
                                                neural_OUTPUT_COMP_TOL = neural_dynamic_tol)
        
        e = y - W @ x_current

        W = W + muW * np.outer(e, x_current)

        By = (1/lambday) * (By - gamy * np.outer(By @ y, By @ y))        
        
        ee = np.dot(Be,e)
        Be = 1 / lambdae * (Be - game * np.outer(ee, ee))

        self.W = W
        self.By = By
        self.Be = self.Be
        
    def fit_batch_antisparse(self, X, n_epochs = 1, neural_dynamic_iterations = 250, neural_lr_start = 0.9, neural_lr_stop = 1e-15, shuffle = False, debug_iteration_point = 1000, plot_in_jupyter = False):
        
        lambday, lambdae, muW, gamy, game, W, By, Be = self.lambday, self.lambdae, self.muW, self.gamy, self.game, self.W, self.By, self.Be
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        h = 1 / gamy # Hopefield parameter

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            plt.figure(figsize = (25, 10), dpi = 80)

        # Y = np.zeros((self.s_dim, samples))
        Y = np.random.randn(self.s_dim, samples)
        
        
        if shuffle:
            idx = np.random.permutation(samples) # random permutation
        else:
            idx = np.arange(samples)
            
        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:,idx[i_sample]]
                y = np.zeros(self.s_dim)

                # Output recurrent weights
                My = By + h * np.eye(self.s_dim)
                y = self.run_neural_dynamics_antisparse(x_current, y, W, My, Be, gamy, game, 
                                                        lr_start = neural_lr_start, lr_stop = neural_lr_stop,
                                                        neural_dynamic_iterations = neural_dynamic_iterations, 
                                                        neural_OUTPUT_COMP_TOL = neural_dynamic_tol)
                        
                e = y - W @ x_current

                W = W + muW * np.outer(e, x_current)
                
                z = By @ y
                By = (1/lambday) * (By - gamy * np.outer(z, z))

                ee = np.dot(Be,e)
                # Be = 1 / lambdae * (Be - game * ((np.outer(ee, ee))))
                # Be = 1 / lambdae * (Be - game * np.diag(np.diag(np.outer(ee,ee))))
                # Record the seperated signal
                Y[:,idx[i_sample]] = y

                if debugging:
                    if (((i_sample % debug_iteration_point) == 0) | (i_sample == samples - 1)) & (i_sample >= debug_iteration_point):
                        self.W = W
                        self.By = By
                        self.Be = Be
                        Wf = self.compute_overall_mapping(return_mapping = True)
                        Y_ = Wf @ X
                        Y_ = self.signed_and_permutation_corrected_sources(S.T,Y_.T)
                        coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                        Y_ = coef_ * Y_
                        SIR = self.CalculateSIR(A, Wf)[0]
                        SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                        SIRlist.append(SIR)
                        SINRlist.append(SINR)
                        SNRlist.append(self.snr(S.T,Y_))
                        self.SIR_list = SIRlist
                        self.SINR_list = SINRlist
                        self.SNR_list = SNRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.subplot(1,2,1)
                            pl.plot(np.arange(1,len(self.SINR_list)+1), np.array(SIRlist), linewidth = 3, label = "SIR")
                            pl.plot(np.arange(1,len(self.SINR_list)+1), np.array(SINRlist), linewidth = 3, label = "SINR")
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                            pl.ylabel("SINR (dB)", fontsize = 35)
                            pl.title("SINR Behaviour", fontsize = 35)
                            pl.xticks(fontsize=45)
                            pl.yticks(fontsize=45)
                            pl.legend(fontsize=25)
                            pl.grid()
                            pl.subplot(1,2,2)
                            pl.plot(np.arange(1,len(self.SINR_list)+1), np.array(SNRlist), linewidth = 3)
                            pl.grid()
                            pl.title("Component SNR Check", fontsize = 35)
                            pl.ylabel("SNR (dB)", fontsize = 35)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                            pl.xticks(fontsize=45)
                            pl.yticks(fontsize=45)
                            clear_output(wait=True)
                            display(pl.gcf())   
        self.W = W
        self.By = By
        self.Be = Be
        
    def fit_batch_nnantisparse(self, X, n_epochs = 1, neural_dynamic_iterations = 250, neural_lr_start = 0.9, neural_lr_stop = 1e-3, shuffle = False, debug_iteration_point = 1000, plot_in_jupyter = False):
        
        lambday, lambdae, muW, gamy, game, W, By, Be = self.lambday, self.lambdae, self.muW, self.gamy, self.game, self.W, self.By, self.Be
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        h = 1 / gamy # Hopefield parameter

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            plt.figure(figsize = (25, 10), dpi = 80)
        # Y = np.zeros((self.s_dim, samples))
        Y = np.random.randn(self.s_dim, samples)
        
        
        if shuffle:
            idx = np.random.permutation(samples) # random permutation
        else:
            idx = np.arange(samples)
            
        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:,idx[i_sample]]
                y = np.zeros(self.s_dim)
                # y = np.random.uniform(0,1, size = (self.s_dim,))
                # Output recurrent weights
                My = By + h * np.eye(self.s_dim)
                y = self.run_neural_dynamics_nnantisparse(x_current, y, W, My, Be, gamy, game, 
                                                             neural_lr_start, neural_lr_stop, neural_dynamic_iterations, 
                                                             neural_dynamic_tol)
                        
                e = y - W @ x_current

                W = (1 - 1e-6) * W + muW * np.outer(e, x_current)
                
                z = By @ y
                By = (1/lambday) * (By - gamy * np.outer(z, z))

                # ee = np.dot(Be,e)
                # Be = 1 / lambdae * (Be - game * np.outer(ee,ee))
                # Record the seperated signal
                Y[:,idx[i_sample]] = y

                if debugging:
                    if (((i_sample % debug_iteration_point) == 0) | (i_sample == samples - 1)) & (i_sample >= debug_iteration_point):
                        self.W = W
                        self.By = By
                        self.Be = Be
                        Wf = self.compute_overall_mapping(return_mapping = True)
                        Y_ = Wf @ X
                        Y_ = self.signed_and_permutation_corrected_sources(S.T,Y_.T)
                        coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                        Y_ = coef_ * Y_
                        SIR = self.CalculateSIR(A, Wf)[0]
                        SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                        SIRlist.append(SIR)
                        SINRlist.append(SINR)
                        SNRlist.append(self.snr(S.T,Y_))
                        self.SIR_list = SIRlist
                        self.SINR_list = SINRlist
                        self.SNR_list = SNRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.subplot(1,2,1)
                            pl.plot(np.arange(1,len(self.SINR_list)+1), np.array(SIRlist), linewidth = 3, label = "SIR")
                            pl.plot(np.arange(1,len(self.SINR_list)+1), np.array(SINRlist), linewidth = 3, label = "SINR")
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                            pl.ylabel("SINR (dB)", fontsize = 35)
                            pl.title("SINR Behaviour", fontsize = 35)
                            pl.xticks(fontsize=45)
                            pl.yticks(fontsize=45)
                            pl.legend(fontsize=25)
                            pl.grid()
                            pl.subplot(1,2,2)
                            pl.plot(np.arange(1,len(self.SINR_list)+1), np.array(SNRlist), linewidth = 3)
                            pl.grid()
                            pl.title("Component SNR Check", fontsize = 35)
                            pl.ylabel("SNR (dB)", fontsize = 35)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                            pl.xticks(fontsize=45)
                            pl.yticks(fontsize=45)
                            clear_output(wait=True)
                            display(pl.gcf())   
        self.W = W
        self.By = By
        self.Be = Be

    def fit_batch_mixedantisparse(self, X, nn_components, n_epochs = 1, neural_dynamic_iterations = 250, neural_lr_start = 0.9, neural_lr_stop = 1e-3, shuffle = False, debug_iteration_point = 1000, plot_in_jupyter = False):
        
        lambday, lambdae, muW, gamy, game, W, By, Be = self.lambday, self.lambdae, self.muW, self.gamy, self.game, self.W, self.By, self.Be
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
        
        source_indices = [j for j in range(self.s_dim)]
        signed_components = source_indices.copy()
        for a in nn_components:
            signed_components.remove(a)
        nn_components = np.array(nn_components)
        signed_components = np.array(signed_components)

        if shuffle:
            idx = np.random.permutation(samples) # random permutation
        else:
            idx = np.arange(samples)
            
        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:,idx[i_sample]]
                y = np.zeros(self.s_dim)

                # Output recurrent weights
                My = By + h * np.eye(self.s_dim)
                y = self.run_neural_dynamics_mixedantisparse(x_current, y, nn_components, signed_components, W, My, Be, gamy, game, 
                                                             lr_start = neural_lr_start, lr_stop = neural_lr_stop, 
                                                             neural_dynamic_iterations = neural_dynamic_iterations, 
                                                             neural_OUTPUT_COMP_TOL = neural_dynamic_tol)
                        
                e = y - W @ x_current

                W = W + muW * np.outer(e, x_current)
                
                z = By @ y
                By = (1/lambday) * (By - gamy * np.outer(z, z))

                ee = np.dot(Be,e)
                Be = 1 / lambdae * (Be - game * np.outer(ee,ee))
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
       
    def fit_batch_sparse(self, X, n_epochs = 1, neural_dynamic_iterations = 250, neural_lr_start = 0.9, neural_lr_stop = 1e-3, shuffle = False, debug_iteration_point = 1000, plot_in_jupyter = False):
        
        lambday, lambdae, muW, gamy, game, W, By, Be = self.lambday, self.lambdae, self.muW, self.gamy, self.game, self.W, self.By, self.Be
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        h = 1 / gamy # Hopefield parameter

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            plt.figure(figsize = (25, 10), dpi = 80)

        # Y = np.zeros((self.s_dim, samples))
        Y = np.random.randn(self.s_dim, samples)
        
        
        if shuffle:
            idx = np.random.permutation(samples) # random permutation
        else:
            idx = np.arange(samples)
            
        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:,idx[i_sample]]
                y = np.zeros(self.s_dim)
                # y = np.random.uniform(0,1, size = (self.s_dim,))
                # Output recurrent weights
                My = By + h * np.eye(self.s_dim)
                y = self.run_neural_dynamics_sparse(x_current, y, W, My, Be, gamy, game, 
                                                    neural_lr_start, neural_lr_stop, neural_dynamic_iterations, 
                                                    neural_dynamic_tol)
                        
                e = y - W @ x_current

                W = (1 - 1e-6) * W + muW * np.outer(e, x_current)
                
                z = By @ y
                By = (1/lambday) * (By - gamy * np.outer(z, z))

                # ee = np.dot(Be,e)
                # Be = 1 / lambdae * (Be - game * np.outer(ee,ee))
                # Record the seperated signal
                Y[:,idx[i_sample]] = y

                if debugging:
                    if (((i_sample % debug_iteration_point) == 0) | (i_sample == samples - 1)):
                        self.W = W
                        self.By = By
                        self.Be = Be
                        Wf = self.compute_overall_mapping(return_mapping = True)
                        Y_ = Wf @ X
                        Y_ = self.signed_and_permutation_corrected_sources(S.T,Y_.T)
                        coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                        Y_ = coef_ * Y_
                        SIR = self.CalculateSIR(A, Wf)[0]
                        SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                        SIRlist.append(SIR)
                        SINRlist.append(SINR)
                        SNRlist.append(self.snr(S.T,Y_))
                        self.SIR_list = SIRlist
                        self.SINR_list = SINRlist
                        self.SNR_list = SNRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.subplot(1,2,1)
                            pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                            pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                            pl.ylabel("SINR (dB)", fontsize = 35)
                            pl.title("SINR Behaviour", fontsize = 35)
                            pl.xticks(fontsize=45)
                            pl.yticks(fontsize=45)
                            pl.legend(fontsize=25)
                            pl.grid()
                            pl.subplot(1,2,2)
                            pl.plot(np.array(SNRlist), linewidth = 3)
                            pl.grid()
                            pl.title("Component SNR Check", fontsize = 35)
                            pl.ylabel("SNR (dB)", fontsize = 35)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                            pl.xticks(fontsize=45)
                            pl.yticks(fontsize=45)
                            clear_output(wait=True)
                            display(pl.gcf())  
        self.W = W
        self.By = By
        self.Be = Be

    def fit_batch_nnsparse(self, X, n_epochs = 1, neural_dynamic_iterations = 250, neural_lr_start = 0.9, neural_lr_stop = 1e-3, shuffle = False, debug_iteration_point = 1000, plot_in_jupyter = False):
        
        lambday, lambdae, muW, gamy, game, W, By, Be = self.lambday, self.lambdae, self.muW, self.gamy, self.game, self.W, self.By, self.Be
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        h = 1 / gamy # Hopefield parameter

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            plt.figure(figsize = (25, 10), dpi = 80)

        # Y = np.zeros((self.s_dim, samples))
        Y = np.random.randn(self.s_dim, samples)
        
        
        if shuffle:
            idx = np.random.permutation(samples) # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:,idx[i_sample]]
                y = np.zeros(self.s_dim)
                # y = np.random.uniform(0,1, size = (self.s_dim,))
                # Output recurrent weights
                My = By + h * np.eye(self.s_dim)
                y = self.run_neural_dynamics_nnsparse(x_current, y, W, My, Be, gamy, game, 
                                                    neural_lr_start, neural_lr_stop, neural_dynamic_iterations, 
                                                    neural_dynamic_tol)
                        
                e = y - W @ x_current

                W = (1 - 1e-6) * W + muW * np.outer(e, x_current)
                
                z = By @ y
                By = (1/lambday) * (By - gamy * np.outer(z, z))

                # ee = np.dot(Be,e)
                # Be = 1 / lambdae * (Be - game * np.outer(ee,ee))
                # Record the seperated signal
                Y[:,idx[i_sample]] = y

                if debugging:
                    if (((i_sample % debug_iteration_point) == 0) | (i_sample == samples - 1)):
                        self.W = W
                        self.By = By
                        self.Be = Be
                        Wf = self.compute_overall_mapping(return_mapping = True)
                        Y_ = Wf @ X
                        Y_ = self.signed_and_permutation_corrected_sources(S.T,Y_.T)
                        coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                        Y_ = coef_ * Y_
                        SIR = self.CalculateSIR(A, Wf)[0]
                        SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                        SIRlist.append(SIR)
                        SINRlist.append(SINR)
                        SNRlist.append(self.snr(S.T,Y_))
                        self.SIR_list = SIRlist
                        self.SINR_list = SINRlist
                        self.SNR_list = SNRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.subplot(1,2,1)
                            pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                            pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                            pl.ylabel("SINR (dB)", fontsize = 35)
                            pl.title("SINR Behaviour", fontsize = 35)
                            pl.xticks(fontsize=45)
                            pl.yticks(fontsize=45)
                            pl.legend(fontsize=25)
                            pl.grid()
                            pl.subplot(1,2,2)
                            pl.plot(np.array(SNRlist), linewidth = 3)
                            pl.grid()
                            pl.title("Component SNR Check", fontsize = 35)
                            pl.ylabel("SNR (dB)", fontsize = 35)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                            pl.xticks(fontsize=45)
                            pl.yticks(fontsize=45)
                            clear_output(wait=True)
                            display(pl.gcf())   
        self.W = W
        self.By = By
        self.Be = Be

    def fit_batch_simplex(self, X, n_epochs = 1, neural_dynamic_iterations = 250, neural_lr_start = 0.9, neural_lr_stop = 1e-3, shuffle = False, debug_iteration_point = 1000, plot_in_jupyter = False):
        
        lambday, lambdae, muW, gamy, game, W, By, Be = self.lambday, self.lambdae, self.muW, self.gamy, self.game, self.W, self.By, self.Be
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        h = 1 / gamy # Hopefield parameter

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            Szeromean = S - S.mean(axis = 1).reshape(-1,1)
            plt.figure(figsize = (25, 10), dpi = 80)

        # Y = np.zeros((self.s_dim, samples))
        Y = np.random.randn(self.s_dim, samples)
        
        if shuffle:
            idx = np.random.permutation(samples) # random permutation
        else:
            idx = np.arange(samples)
            
        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:,idx[i_sample]]
                y = np.zeros(self.s_dim)
                # y = np.random.uniform(0,1, size = (self.s_dim,))
                # Output recurrent weights
                My = By + h * np.eye(self.s_dim)
                y = self.run_neural_dynamics_simplex(x_current, y, W, My, Be, gamy, game, 
                                                    neural_lr_start, neural_lr_stop, neural_dynamic_iterations, 
                                                    neural_dynamic_tol)
                        
                e = y - W @ x_current

                W = (1 - 1e-6) * W + muW * np.outer(e, x_current)
                
                z = By @ y
                By = (1/lambday) * (By - gamy * np.outer(z, z))

                # ee = np.dot(Be,e)
                # # Be = 1 / lambdae * (Be - game * np.outer(ee,ee))
                # Be = 1 / lambdae * (Be - game * np.diag(np.diag(np.outer(ee,ee))))
                # Record the seperated signal
                Y[:,idx[i_sample]] = y

                if debugging:
                    if (((i_sample % debug_iteration_point) == 0) | (i_sample == samples - 1)) & (i_sample > debug_iteration_point):
                        self.W = W
                        self.By = By
                        self.Be = Be
                        Wf = self.compute_overall_mapping(return_mapping = True)
                        Y = Wf @ X
                        Yzeromean = Y - Y.mean(axis = 1).reshape(-1,1)
                        Y_ = self.signed_and_permutation_corrected_sources(Szeromean.T,Yzeromean.T)
                        coef_ = (Y_ * Szeromean.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                        Y_ = coef_ * Y_
                        SIR = self.CalculateSIR(A, Wf)[0]
                        SINR = 10*np.log10(self.CalculateSINR(Y_.T, Szeromean)[0])
                        SIRlist.append(SIR)
                        SINRlist.append(SINR)
                        SNRlist.append(self.snr(Szeromean.T,Y_))
                        self.SIR_list = SIRlist
                        self.SINR_list = SINRlist
                        self.SNR_list = SNRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.subplot(1,2,1)
                            pl.plot(np.arange(2,len(self.SINR_list)+2), np.array(SIRlist), linewidth = 3, label = "SIR")
                            pl.plot(np.arange(2,len(self.SINR_list)+2), np.array(SINRlist), linewidth = 3, label = "SINR")
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                            pl.ylabel("SINR (dB)", fontsize = 35)
                            pl.title("SINR Behaviour", fontsize = 35)
                            pl.xticks(fontsize=45)
                            pl.yticks(fontsize=45)
                            pl.legend(fontsize=25)
                            pl.grid()
                            pl.subplot(1,2,2)
                            pl.plot(np.arange(2,len(self.SINR_list)+2), np.array(SNRlist), linewidth = 3)
                            pl.grid()
                            pl.title("Component SNR Check", fontsize = 35)
                            pl.ylabel("SNR (dB)", fontsize = 35)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                            pl.xticks(fontsize=45)
                            pl.yticks(fontsize=45)
                            clear_output(wait=True)
                            display(pl.gcf())  
        self.W = W
        self.By = By
        self.Be = Be

    def fit_batch_nnwsubsparse(self, X, sparse_components, n_epochs = 1, neural_dynamic_iterations = 250, neural_lr_start = 0.9, neural_lr_stop = 1e-3, shuffle = False, debug_iteration_point = 1000, plot_in_jupyter = False):
        
        lambday, lambdae, muW, gamy, game, W, By, Be = self.lambday, self.lambdae, self.muW, self.gamy, self.game, self.W, self.By, self.Be
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
        
        source_indices = [j for j in range(self.s_dim)]
        nn_components = source_indices.copy()
        for a in sparse_components:
            nn_components.remove(a)
        sparse_components = np.array(sparse_components)
        nn_components = np.array(nn_components)

        if shuffle:
            idx = np.random.permutation(samples) # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:,idx[i_sample]]
                y = np.zeros(self.s_dim)
                # y = np.random.uniform(0,1, size = (self.s_dim,))
                # Output recurrent weights
                My = By + h * np.eye(self.s_dim)
                y = self.run_neural_dynamics_nnwsubsparse(x_current, y, nn_components, sparse_components, W, My, Be, gamy, game, 
                                                    neural_lr_start, neural_lr_stop, neural_dynamic_iterations, 
                                                    neural_dynamic_tol)
                        
                e = y - W @ x_current

                W = (1 - 1e-6) * W + muW * np.outer(e, x_current)
                
                z = By @ y
                By = (1/lambday) * (By - gamy * np.outer(z, z))

                # ee = np.dot(Be,e)
                # Be = 1 / lambdae * (Be - game * np.outer(ee,ee))
                # # Record the seperated signal
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

    def fit_batch_nnwsubnnsparse(self, X, nnsparse_components, n_epochs = 1, neural_dynamic_iterations = 250, neural_lr_start = 0.9, neural_lr_stop = 1e-3, shuffle = False, debug_iteration_point = 1000, plot_in_jupyter = False):
        
        lambday, lambdae, muW, gamy, game, W, By, Be = self.lambday, self.lambdae, self.muW, self.gamy, self.game, self.W, self.By, self.Be
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
        
        source_indices = [j for j in range(self.s_dim)]
        nn_components = source_indices.copy()
        for a in nnsparse_components:
            nn_components.remove(a)
        nnsparse_components = np.array(nnsparse_components)
        nn_components = np.array(nn_components)

        if shuffle:
            idx = np.random.permutation(samples) # random permutation
        else:
            idx = np.arange(samples)
            
        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:,idx[i_sample]]
                y = np.zeros(self.s_dim)
                # y = np.random.uniform(0,1, size = (self.s_dim,))
                # Output recurrent weights
                My = By + h * np.eye(self.s_dim)
                y = self.run_neural_dynamics_nnwsubnnsparse(x_current, y, nn_components, nnsparse_components, W, My, Be, gamy, game, 
                                                    neural_lr_start, neural_lr_stop, neural_dynamic_iterations, 
                                                    neural_dynamic_tol)
                        
                e = y - W @ x_current

                W = (1 - 1e-6) * W + muW * np.outer(e, x_current)
                
                z = By @ y
                By = (1/lambday) * (By - gamy * np.outer(z, z))

                # ee = np.dot(Be,e)
                # Be = 1 / lambdae * (Be - game * np.outer(ee,ee))
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

class LDMIBSS:

    """
    Implementation of batch Log-Det Mutual Information Based Blind Source Separation Framework
    Parameters:
    =================================
    s_dim          -- Dimension of the sources
    x_dim          -- Dimension of the mixtures
    W              -- Feedforward Synapses

    Methods:
    ==================================
    fit_batch_antisparse
    fit_batch_nnantisparse

    """
    
    def __init__(self, s_dim, x_dim, W = None, set_ground_truth = False, S = None, A = None):
        if W is not None:
            assert W.shape == (s_dim, x_dim), "The shape of the initial guess W must be (s_dim, x_dim) = (%d,%d)" % (s_dim, x_dim)
            W = W
        else:
            W = np.random.randn(s_dim, x_dim)
            
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.W = W # Trainable separator matrix, i.e., W@X \approx S
        ### Ground Truth Sources and Mixing Matrix For Debugging
        self.set_ground_truth = set_ground_truth
        self.S = S # Sources
        self.A = A # Mixing Matrix
        self.SIR_list = []
        self.SINR_list = []
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

    def CalculateSINR(self, Out, S, compute_permutation = True):
        r=S.shape[0]
        if compute_permutation:
            G=np.dot(Out-np.reshape(np.mean(Out,1),(r,1)),np.linalg.pinv(S-np.reshape(np.mean(S,1),(r,1))))
            indmax=np.argmax(np.abs(G),1)
        else:
            G=np.dot(Out-np.reshape(np.mean(Out,1),(r,1)),np.linalg.pinv(S-np.reshape(np.mean(S,1),(r,1))))
            indmax = np.arange(0,r)
        GG=np.zeros((r,r))
        for kk in range(r):
            GG[kk,indmax[kk]]=np.dot(Out[kk,:]-np.mean(Out[kk,:]),S[indmax[kk],:].T-np.mean(S[indmax[kk],:]))/np.dot(S[indmax[kk],:]-np.mean(S[indmax[kk],:]),S[indmax[kk],:].T-np.mean(S[indmax[kk],:]))#(G[kk,indmax[kk]])
        ZZ=GG@(S-np.reshape(np.mean(S,1),(r,1)))+np.reshape(np.mean(Out,1),(r,1))
        E=Out-ZZ
        MSE=np.linalg.norm(E,'fro')**2
        SigPow=np.linalg.norm(ZZ,'fro')**2
        SINR=(SigPow/MSE)
        return SINR,SigPow,MSE,G

    def snr(self, S_original, S_noisy):
        N_hat = S_original - S_noisy
        N_P = (N_hat ** 2).sum(axis = 0)
        S_P = (S_original ** 2).sum(axis = 0)
        snr = 10 * np.log10(S_P / N_P)
        return snr

    def outer_prod_broadcasting(self, A, B):
        """Broadcasting trick"""
        return A[...,None]*B[:,None]

    def find_permutation_between_source_and_estimation(self, S, Y):
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

    def signed_and_permutation_corrected_sources(self, S, Y):
        perm = self.find_permutation_between_source_and_estimation(S,Y)
        return np.sign((Y[:,perm] * S).sum(axis = 0)) * Y[:,perm]

    @staticmethod
    @njit
    def update_Y_corr_based(Y, X, W, epsilon, step_size):
        s_dim, samples = Y.shape[0], Y.shape[1]
        Identity_like_Y = np.eye(s_dim)
        RY = (1/samples) * np.dot(Y, Y.T) + epsilon * Identity_like_Y
        E = Y - np.dot(W, X)
        RE = (1/samples) * np.dot(E, E.T) + epsilon * Identity_like_Y
        gradY = (1/samples) * (np.dot(np.linalg.pinv(RY), Y) - np.dot(np.linalg.pinv(RE), E))
        Y = Y + (step_size) * gradY
        return Y

    # @njit(parallel=True)
    # def mean_numba(a):

    #     res = []
    #     for i in range(a.shape[0]):
    #         res.append(a[i, :].mean())

    #     return np.array(res)

    @staticmethod
    @njit
    def update_Y_cov_based(Y, X, muX, W, epsilon, step_size):
        def mean_numba(a):

            res = []
            for i in range(a.shape[0]):
                res.append(a[i, :].mean())

            return np.array(res)
        s_dim, samples = Y.shape[0], Y.shape[1]
        muY = mean_numba(Y).reshape(-1,1)
        Identity_like_Y = np.eye(s_dim)
        RY = (1/samples) * (np.dot(Y, Y.T) - np.dot(muY, muY.T)) + epsilon * Identity_like_Y
        E = (Y - muY) - np.dot(W, (X - muX.reshape(-1,1)))
        muE = mean_numba(E).reshape(-1,1)
        RE = (1/samples) * (np.dot(E, E.T) - np.dot(muE, muE.T)) + epsilon * Identity_like_Y
        gradY = (1/samples) * (np.dot(np.linalg.pinv(RY), Y - muY) - np.dot(np.linalg.pinv(RE), E - muE))
        Y = Y + (step_size) * gradY
        return Y

    @staticmethod
    @njit
    def ProjectOntoLInfty(X):
        return X*(X>=-1.0)*(X<=1.0)+(X>1.0)*1.0-1.0*(X<-1.0)
    
    @staticmethod
    @njit
    def ProjectOntoNNLInfty(X):
        return X*(X>=0.0)*(X<=1.0)+(X>1.0)*1.0#-0.0*(X<0.0)
        
    def ProjectRowstoL1NormBall(self, H):
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

    def ProjectColstoSimplex(self, v, z=1):
        """v array of shape (n_features, n_samples)."""
        p, n = v.shape
        u = np.sort(v, axis=0)[::-1, ...]
        pi = np.cumsum(u, axis=0) - z
        ind = (np.arange(p) + 1).reshape(-1, 1)
        mask = (u - pi / ind) > 0
        rho = p - 1 - np.argmax(mask[::-1, ...], axis=0)
        theta = pi[tuple([rho, np.arange(n)])] / (rho + 1)
        w = np.maximum(v - theta, 0)
        return w

    def fit_batch_antisparse(self, X, n_iterations = 1000, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            plt.figure(figsize = (25, 10), dpi = 80)

        if method == "correlation":
            RX = (1/samples) * np.dot(X, X.T)
            RXinv = np.linalg.pinv(RX)
        elif method == "covariance":
            muX = np.mean(X, axis = 1)
            RX = (1/samples) * (np.dot(X, X.T) - np.outer(muX, muX))
            RXinv = np.linalg.pinv(RX)
        Y = np.zeros((self.s_dim, samples))
        # Y = (np.random.rand(self.s_dim, samples) - 0.5)/2
        for k in range(n_iterations):
            if method == "correlation":
                Y = self.update_Y_corr_based(Y, X, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectOntoLInfty(Y)
                RYX = (1/samples) * np.dot(Y, X.T)
            elif method == "covariance":
                Y = self.update_Y_cov_based(Y, X, muX, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectOntoLInfty(Y)
                muY = np.mean(Y, axis = 1)
                RYX = (1/samples) * (np.dot(Y, X.T) - np.outer(muY, muX))
            W = np.dot(RYX, RXinv)

            if debugging:
                if ((k % debug_iteration_point) == 0)  | (k == n_iterations - 1):
                    self.W = W
                    Y_ = self.signed_and_permutation_corrected_sources(S.T,Y.T)
                    coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                    Y_ = coef_ * Y_
                    SIR = self.CalculateSIR(A, W)[0]
                    SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                    SINRlist.append(SINR)
                    SNRlist.append(self.snr(S.T,Y_))
                    SIRlist.append(SIR)
                    self.SIR_list = SIRlist
                    self.SINR_list = SINRlist
                    self.SNR_list = SNRlist
                    if plot_in_jupyter:
                        pl.clf()
                        pl.subplot(1,2,1)
                        pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                        pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.ylabel("SINR (dB)", fontsize = 35)
                        pl.title("SINR Behaviour", fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        pl.legend(fontsize=25)
                        pl.grid()
                        pl.subplot(1,2,2)
                        pl.plot(np.array(SNRlist), linewidth = 3)
                        pl.grid()
                        pl.title("Component SNR Check", fontsize = 35)
                        pl.ylabel("SNR (dB)", fontsize = 35)
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        clear_output(wait=True)
                        display(pl.gcf())  
        self.W = W

    def fit_batch_nnantisparse(self, X, n_iterations = 1000, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            plt.figure(figsize = (25, 10), dpi = 80)

        if method == "correlation":
            RX = (1/samples) * np.dot(X, X.T)
            RXinv = np.linalg.pinv(RX)
        elif method == "covariance":
            muX = np.mean(X, axis = 1)
            RX = (1/samples) * (np.dot(X, X.T) - np.outer(muX, muX))
            RXinv = np.linalg.pinv(RX)
        Y = np.zeros((self.s_dim, samples))
        Y = np.random.rand(self.s_dim, samples)/2
        for k in tqdm(range(n_iterations)):
            if method == "correlation":
                Y = self.update_Y_corr_based(Y, X, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectOntoNNLInfty(Y)
                RYX = (1/samples) * np.dot(Y, X.T)
            elif method == "covariance":
                Y = self.update_Y_cov_based(Y, X, muX, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectOntoNNLInfty(Y)
                muY = np.mean(Y, axis = 1)
                RYX = (1/samples) * (np.dot(Y, X.T) - np.outer(muY, muX))
            W = np.dot(RYX, RXinv)

            if debugging:
                if ((k % debug_iteration_point) == 0)  | (k == n_iterations - 1):
                    self.W = W
                    Y_ = self.signed_and_permutation_corrected_sources(S.T,Y.T)
                    coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                    Y_ = coef_ * Y_
                    SIR = self.CalculateSIR(A, W)[0]
                    SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                    SINRlist.append(SINR)
                    SNRlist.append(self.snr(S.T,Y_))
                    SIRlist.append(SIR)
                    self.SIR_list = SIRlist
                    self.SINR_list = SINRlist
                    self.SNR_list = SNRlist
                    if plot_in_jupyter:
                        pl.clf()
                        pl.subplot(1,2,1)
                        pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                        pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.ylabel("SINR (dB)", fontsize = 35)
                        pl.title("SINR Behaviour", fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        pl.legend(fontsize=25)
                        pl.grid()
                        pl.subplot(1,2,2)
                        pl.plot(np.array(SNRlist), linewidth = 3)
                        pl.grid()
                        pl.title("Component SNR Check", fontsize = 35)
                        pl.ylabel("SNR (dB)", fontsize = 35)
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        clear_output(wait=True)
                        display(pl.gcf())  
        
    def fit_batch_sparse(self, X, n_iterations = 1000, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            plt.figure(figsize = (25, 10), dpi = 80)

        if method == "correlation":
            RX = (1/samples) * np.dot(X, X.T)
            RXinv = np.linalg.pinv(RX)
        elif method == "covariance":
            muX = np.mean(X, axis = 1)
            RX = (1/samples) * (np.dot(X, X.T) - np.outer(muX, muX))
            RXinv = np.linalg.pinv(RX)
        Y = np.zeros((self.s_dim, samples))
        # Y = np.random.rand(self.s_dim, samples)/2
        for k in range(n_iterations):
            if method == "correlation":
                Y = self.update_Y_corr_based(Y, X, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectRowstoL1NormBall(Y.T).T
                RYX = (1/samples) * np.dot(Y, X.T)
            elif method == "covariance":
                Y = self.update_Y_cov_based(Y, X, muX, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectRowstoL1NormBall(Y.T).T
                muY = np.mean(Y, axis = 1)
                RYX = (1/samples) * (np.dot(Y, X.T) - np.outer(muY, muX))
            W = np.dot(RYX, RXinv)

            if debugging:
                if ((k % debug_iteration_point) == 0)  | (k == n_iterations - 1):
                    self.W = W
                    Y_ = self.signed_and_permutation_corrected_sources(S.T,Y.T)
                    coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                    Y_ = coef_ * Y_
                    SIR = self.CalculateSIR(A, W)[0]
                    SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                    SINRlist.append(SINR)
                    SNRlist.append(self.snr(S.T,Y_))
                    SIRlist.append(SIR)
                    self.SIR_list = SIRlist
                    self.SINR_list = SINRlist
                    self.SNR_list = SNRlist
                    if plot_in_jupyter:
                        pl.clf()
                        pl.subplot(1,2,1)
                        pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                        pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.ylabel("SINR (dB)", fontsize = 35)
                        pl.title("SINR Behaviour", fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        pl.legend(fontsize=25)
                        pl.grid()
                        pl.subplot(1,2,2)
                        pl.plot(np.array(SNRlist), linewidth = 3)
                        pl.grid()
                        pl.title("Component SNR Check", fontsize = 35)
                        pl.ylabel("SNR (dB)", fontsize = 35)
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        clear_output(wait=True)
                        display(pl.gcf())  
        self.W = W

    def fit_batch_nnsparse(self, X, n_iterations = 1000, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            plt.figure(figsize = (25, 10), dpi = 80)

        if method == "correlation":
            RX = (1/samples) * np.dot(X, X.T)
            RXinv = np.linalg.pinv(RX)
        elif method == "covariance":
            muX = np.mean(X, axis = 1)
            RX = (1/samples) * (np.dot(X, X.T) - np.outer(muX, muX))
            RXinv = np.linalg.pinv(RX)
        Y = np.zeros((self.s_dim, samples))
        # Y = np.random.rand(self.s_dim, samples)/2
        for k in range(n_iterations):
            if method == "correlation":
                Y = self.update_Y_corr_based(Y, X, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectRowstoL1NormBall((Y * (Y>= 0)).T).T
                RYX = (1/samples) * np.dot(Y, X.T)
            elif method == "covariance":
                Y = self.update_Y_cov_based(Y, X, muX, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectRowstoL1NormBall((Y * (Y>= 0)).T).T
                muY = np.mean(Y, axis = 1)
                RYX = (1/samples) * (np.dot(Y, X.T) - np.outer(muY, muX))
            W = np.dot(RYX, RXinv)

            if debugging:
                if ((k % debug_iteration_point) == 0)  | (k == n_iterations - 1):
                    self.W = W
                    Y_ = self.signed_and_permutation_corrected_sources(S.T,Y.T)
                    coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                    Y_ = coef_ * Y_
                    SIR = self.CalculateSIR(A, W)[0]
                    SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                    SINRlist.append(SINR)
                    SNRlist.append(self.snr(S.T,Y_))
                    SIRlist.append(SIR)
                    self.SIR_list = SIRlist
                    self.SINR_list = SINRlist
                    self.SNR_list = SNRlist
                    if plot_in_jupyter:
                        pl.clf()
                        pl.subplot(1,2,1)
                        pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                        pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.ylabel("SINR (dB)", fontsize = 35)
                        pl.title("SINR Behaviour", fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        pl.legend(fontsize=25)
                        pl.grid()
                        pl.subplot(1,2,2)
                        pl.plot(np.array(SNRlist), linewidth = 3)
                        pl.grid()
                        pl.title("Component SNR Check", fontsize = 35)
                        pl.ylabel("SNR (dB)", fontsize = 35)
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        clear_output(wait=True)
                        display(pl.gcf())   
        self.W = W

    def fit_batch_simplex(self, X, n_iterations = 1000, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            plt.figure(figsize = (25, 10), dpi = 80)

        if method == "correlation":
            RX = (1/samples) * np.dot(X, X.T)
            RXinv = np.linalg.pinv(RX)
        elif method == "covariance":
            muX = np.mean(X, axis = 1)
            RX = (1/samples) * (np.dot(X, X.T) - np.outer(muX, muX))
            RXinv = np.linalg.pinv(RX)
        Y = np.zeros((self.s_dim, samples))
        # Y = np.random.rand(self.s_dim, samples)/2
        for k in range(n_iterations):
            if method == "correlation":
                Y = self.update_Y_corr_based(Y, X, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectColstoSimplex(Y)
                RYX = (1/samples) * np.dot(Y, X.T)
            elif method == "covariance":
                Y = self.update_Y_cov_based(Y, X, muX, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectColstoSimplex(Y)
                muY = np.mean(Y, axis = 1)
                RYX = (1/samples) * (np.dot(Y, X.T) - np.outer(muY, muX))
            W = np.dot(RYX, RXinv)

            if debugging:
                if ((k % debug_iteration_point) == 0)  | (k == n_iterations - 1):
                    self.W = W
                    Y_ = self.signed_and_permutation_corrected_sources(S.T,Y.T)
                    coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                    Y_ = coef_ * Y_
                    SIR = self.CalculateSIR(A, W)[0]
                    SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                    SINRlist.append(SINR)
                    SNRlist.append(self.snr(S.T,Y_))
                    SIRlist.append(SIR)
                    self.SIR_list = SIRlist
                    self.SINR_list = SINRlist
                    self.SNR_list = SNRlist
                    if plot_in_jupyter:
                        pl.clf()
                        pl.subplot(1,2,1)
                        pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                        pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.ylabel("SINR (dB)", fontsize = 35)
                        pl.title("SINR Behaviour", fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        pl.legend(fontsize=25)
                        pl.grid()
                        pl.subplot(1,2,2)
                        pl.plot(np.array(SNRlist), linewidth = 3)
                        pl.grid()
                        pl.title("Component SNR Check", fontsize = 35)
                        pl.ylabel("SNR (dB)", fontsize = 35)
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        clear_output(wait=True)
                        display(pl.gcf())   
        self.W = W

class BatchLDMIBSS:

    """
    Implementation of batch Log-Det Mutual Information Based Blind Source Separation Framework

    ALMOST THE SAME IMPLEMENTATION WITH THE ABOVE LDMIBSS CLASS. THE ONLY DIFFERENCE IS THAT 
    THIS ALGORITHM UPDATES ARE PERFORMED BASED ON THE MINIBATCHES. THE ABOVE LDMIBSS CLASS IS 
    WORKING SLOW WHEN THE NUMBER OF DATA IS BIG (FOR EXAMPLE THE MIXTURE SIZE IS (Nmixtures, Nsamples) = (10, 500000)).
    THEREFORE, IN EACH ITERATION, WE TAKE A MINIBATCH OF MIXTURES TO RUN THE ALGORITHM. IN THE DEBUGGING
    PART FOR SNR ANS SINR CALCULATION, THE WHOLE DATA IS USED (NOT THE MINIBATCHES).

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
    
    def __init__(self, s_dim, x_dim, W = None, set_ground_truth = False, S = None, A = None):
        if W is not None:
            assert W.shape == (s_dim, x_dim), "The shape of the initial guess W must be (s_dim, x_dim) = (%d,%d)" % (s_dim, x_dim)
            W = W
        else:
            W = np.random.randn(s_dim, x_dim)
            
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.W = W # Trainable separator matrix, i.e., W@X \approx S
        ### Ground Truth Sources and Mixing Matrix For Debugging
        self.set_ground_truth = set_ground_truth
        self.S = S # Sources
        self.A = A # Mixing Matrix
        self.SIR_list = []
        self.SINR_list = []
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

    def CalculateSINR(self, Out, S, compute_permutation = True):
        r=S.shape[0]
        if compute_permutation:
            G=np.dot(Out-np.reshape(np.mean(Out,1),(r,1)),np.linalg.pinv(S-np.reshape(np.mean(S,1),(r,1))))
            indmax=np.argmax(np.abs(G),1)
        else:
            G=np.dot(Out-np.reshape(np.mean(Out,1),(r,1)),np.linalg.pinv(S-np.reshape(np.mean(S,1),(r,1))))
            indmax = np.arange(0,r)
        GG=np.zeros((r,r))
        for kk in range(r):
            GG[kk,indmax[kk]]=np.dot(Out[kk,:]-np.mean(Out[kk,:]),S[indmax[kk],:].T-np.mean(S[indmax[kk],:]))/np.dot(S[indmax[kk],:]-np.mean(S[indmax[kk],:]),S[indmax[kk],:].T-np.mean(S[indmax[kk],:]))#(G[kk,indmax[kk]])
        ZZ=GG@(S-np.reshape(np.mean(S,1),(r,1)))+np.reshape(np.mean(Out,1),(r,1))
        E=Out-ZZ
        MSE=np.linalg.norm(E,'fro')**2
        SigPow=np.linalg.norm(ZZ,'fro')**2
        SINR=(SigPow/MSE)
        return SINR,SigPow,MSE,G

    def snr(self, S_original, S_noisy):
        N_hat = S_original - S_noisy
        N_P = (N_hat ** 2).sum(axis = 0)
        S_P = (S_original ** 2).sum(axis = 0)
        snr = 10 * np.log10(S_P / N_P)
        return snr

    def outer_prod_broadcasting(self, A, B):
        """Broadcasting trick"""
        return A[...,None]*B[:,None]

    def find_permutation_between_source_and_estimation(self, S, Y):
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

    def signed_and_permutation_corrected_sources(self, S, Y):
        perm = self.find_permutation_between_source_and_estimation(S,Y)
        return np.sign((Y[:,perm] * S).sum(axis = 0)) * Y[:,perm]

    @staticmethod
    @njit
    def update_Y_corr_based(Y, X, W, epsilon, step_size):
        s_dim, samples = Y.shape[0], Y.shape[1]
        Identity_like_Y = np.eye(s_dim)
        RY = (1/samples) * np.dot(Y, Y.T) + epsilon * Identity_like_Y
        E = Y - np.dot(W, X)
        RE = (1/samples) * np.dot(E, E.T) + epsilon * Identity_like_Y
        gradY = (1/samples) * (np.dot(np.linalg.pinv(RY), Y) - np.dot(np.linalg.pinv(RE), E))
        Y = Y + (step_size) * gradY
        return Y

    # @njit(parallel=True)
    # def mean_numba(a):

    #     res = []
    #     for i in range(a.shape[0]):
    #         res.append(a[i, :].mean())

    #     return np.array(res)

    @staticmethod
    @njit
    def update_Y_cov_based(Y, X, muX, W, epsilon, step_size):
        def mean_numba(a):

            res = []
            for i in range(a.shape[0]):
                res.append(a[i, :].mean())

            return np.array(res)
        s_dim, samples = Y.shape[0], Y.shape[1]
        muY = mean_numba(Y).reshape(-1,1)
        Identity_like_Y = np.eye(s_dim)
        RY = (1/samples) * (np.dot(Y, Y.T) - np.dot(muY, muY.T)) + epsilon * Identity_like_Y
        E = (Y - muY) - np.dot(W, (X - muX.reshape(-1,1)))
        muE = mean_numba(E).reshape(-1,1)
        RE = (1/samples) * (np.dot(E, E.T) - np.dot(muE, muE.T)) + epsilon * Identity_like_Y
        gradY = (1/samples) * (np.dot(np.linalg.pinv(RY), Y - muY) - np.dot(np.linalg.pinv(RE), E - muE))
        Y = Y + (step_size) * gradY
        return Y

    def ProjectOntoLInfty(self, X):
        return X*(X>=-1.0)*(X<=1.0)+(X>1.0)*1.0-1.0*(X<-1.0)
    
    def ProjectOntoNNLInfty(self, X):
        return X*(X>=0.0)*(X<=1.0)+(X>1.0)*1.0#-0.0*(X<0.0)
        
    def ProjectRowstoL1NormBall(self, H):
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

    def ProjectColstoSimplex(self, v, z=1):
        """v array of shape (n_features, n_samples)."""
        p, n = v.shape
        u = np.sort(v, axis=0)[::-1, ...]
        pi = np.cumsum(u, axis=0) - z
        ind = (np.arange(p) + 1).reshape(-1, 1)
        mask = (u - pi / ind) > 0
        rho = p - 1 - np.argmax(mask[::-1, ...], axis=0)
        theta = pi[tuple([rho, np.arange(n)])] / (rho + 1)
        w = np.maximum(v - theta, 0)
        return w

    def fit_batch_antisparse(self, X, n_iterations = 1000, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            plt.figure(figsize = (25, 10), dpi = 80)

        if method == "correlation":
            RX = (1/samples) * np.dot(X, X.T)
            RXinv = np.linalg.pinv(RX)
        elif method == "covariance":
            muX = np.mean(X, axis = 1)
            RX = (1/samples) * (np.dot(X, X.T) - np.outer(muX, muX))
            RXinv = np.linalg.pinv(RX)
        Y = np.zeros((self.s_dim, samples))
        # Y = (np.random.rand(self.s_dim, samples) - 0.5)/2
        for k in range(n_iterations):
            if method == "correlation":
                Y = self.update_Y_corr_based(Y, X, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectOntoLInfty(Y)
                RYX = (1/samples) * np.dot(Y, X.T)
            elif method == "covariance":
                Y = self.update_Y_cov_based(Y, X, muX, W, epsilon, (mu_start/np.sqrt(k+1)))
                Y = self.ProjectOntoLInfty(Y)
                muY = np.mean(Y, axis = 1)
                RYX = (1/samples) * (np.dot(Y, X.T) - np.outer(muY, muX))
            W = np.dot(RYX, RXinv)

            if debugging:
                if ((k % debug_iteration_point) == 0)  | (k == n_iterations - 1):
                    self.W = W
                    Y_ = self.signed_and_permutation_corrected_sources(S.T,Y.T)
                    coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                    Y_ = coef_ * Y_
                    SIR = self.CalculateSIR(A, W)[0]
                    SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                    SINRlist.append(SINR)
                    SNRlist.append(self.snr(S.T,Y_))
                    SIRlist.append(SIR)
                    self.SIR_list = SIRlist
                    self.SINR_list = SINRlist
                    self.SNR_list = SNRlist
                    if plot_in_jupyter:
                        pl.clf()
                        pl.subplot(1,2,1)
                        pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                        pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.ylabel("SINR (dB)", fontsize = 35)
                        pl.title("SINR Behaviour", fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        pl.legend(fontsize=25)
                        pl.grid()
                        pl.subplot(1,2,2)
                        pl.plot(np.array(SNRlist), linewidth = 3)
                        pl.grid()
                        pl.title("Component SNR Check", fontsize = 35)
                        pl.ylabel("SNR (dB)", fontsize = 35)
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        clear_output(wait=True)
                        display(pl.gcf())  
        self.W = W

    def fit_batch_nnantisparse(self, X, batch_size = 10000, n_epochs = 1, n_iterations_per_batch = 500, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, drop_last_batch = True, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            plt.figure(figsize = (25, 10), dpi = 80)
        total_iteration = 1
        if drop_last_batch:
            m = (X.shape[1])//batch_size
        else:
            m = int(np.ceil((X.shape[1])/batch_size))
        for epoch_ in range(n_epochs):
            for kk in (range(m)):
                Xbatch = X[:,kk*batch_size:(kk+1)*batch_size]
                sample_batch_size = Xbatch.shape[1]
                if kk == 0:
                    Ybatch = np.random.rand(self.s_dim, Xbatch.shape[1])/2
                else:
                    Ybatch = self.W @ Xbatch
                if method == "correlation":
                    RX = (1/sample_batch_size) * np.dot(Xbatch, Xbatch.T)
                    RXinv = np.linalg.pinv(RX)
                elif method == "covariance":
                    muX = np.mean(Xbatch, axis = 1)
                    RX = (1/sample_batch_size) * (np.dot(Xbatch, Xbatch.T) - np.outer(muX, muX))
                    RXinv = np.linalg.pinv(RX)

                for k in tqdm(range(n_iterations_per_batch)):
                    if method == "correlation":
                        Ybatch = self.update_Y_corr_based(Ybatch, Xbatch, W, epsilon, (mu_start/np.sqrt((k+1)*kk+1)))
                        Ybatch = self.ProjectOntoNNLInfty(Ybatch)
                        RYX = (1/sample_batch_size) * np.dot(Ybatch, Xbatch.T)
                    elif method == "covariance":
                        Ybatch = self.update_Y_cov_based(Ybatch, Xbatch, muX, W, epsilon, (mu_start/np.sqrt(total_iteration+1)))
                        Ybatch = self.ProjectOntoNNLInfty(Ybatch)
                        muY = np.mean(Ybatch, axis = 1)
                        RYX = (1/sample_batch_size) * (np.dot(Ybatch, Xbatch.T) - np.outer(muY, muX))
                    W = np.dot(RYX, RXinv)
                    self.W = W
                    total_iteration += 1
                    if debugging:
                        if ((k % debug_iteration_point) == 0)  | (k == n_iterations_per_batch - 1):
                        # if (((k % debug_iteration_point) == 0) | (k == n_iterations_per_batch - 1)) & (k >= debug_iteration_point):
                            Y = W @ X
                            Y_ = self.signed_and_permutation_corrected_sources(S.T,Y.T)
                            coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                            Y_ = coef_ * Y_
                            SIR = self.CalculateSIR(A, W)[0]
                            SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                            SINRlist.append(SINR)
                            SNRlist.append(self.snr(S.T,Y_))
                            SIRlist.append(SIR)
                            self.SIR_list = SIRlist
                            self.SINR_list = SINRlist
                            self.SNR_list = SNRlist
                            if plot_in_jupyter:
                                pl.clf()
                                pl.subplot(1,2,1)
                                pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                                pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                                pl.ylabel("SINR (dB)", fontsize = 35)
                                pl.title("SINR Behaviour", fontsize = 35)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)
                                pl.legend(fontsize=25)
                                pl.grid()
                                pl.subplot(1,2,2)
                                pl.plot(np.array(SNRlist), linewidth = 3)
                                pl.grid()
                                pl.title("Component SNR Check", fontsize = 35)
                                pl.ylabel("SNR (dB)", fontsize = 35)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)
                                clear_output(wait=True)
                                display(pl.gcf())  
        
    def fit_batch_sparse(self, X, batch_size = 10000, n_epochs = 1, n_iterations_per_batch = 500, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, drop_last_batch = True, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            plt.figure(figsize = (25, 10), dpi = 80)
        total_iteration = 1
        if drop_last_batch:
            m = (X.shape[1])//batch_size
        else:
            m = int(np.ceil((X.shape[1])/batch_size))
        for epoch_ in range(n_epochs):
            for kk in (range(m)):
                Xbatch = X[:,kk*batch_size:(kk+1)*batch_size]
                sample_batch_size = Xbatch.shape[1]
                if kk == 0:
                    Ybatch = np.zeros((self.s_dim, Xbatch.shape[1]))
                else:
                    Ybatch = self.W @ Xbatch
                if method == "correlation":
                    RX = (1/sample_batch_size) * np.dot(Xbatch, Xbatch.T)
                    RXinv = np.linalg.pinv(RX)
                elif method == "covariance":
                    muX = np.mean(Xbatch, axis = 1)
                    RX = (1/sample_batch_size) * (np.dot(Xbatch, Xbatch.T) - np.outer(muX, muX))
                    RXinv = np.linalg.pinv(RX)

                for k in tqdm(range(n_iterations_per_batch)):
                    if method == "correlation":
                        Ybatch = self.update_Y_corr_based(Ybatch, Xbatch, W, epsilon, (mu_start/np.sqrt((k+1)*kk+1)))
                        Ybatch = self.ProjectRowstoL1NormBall(Ybatch.T).T
                        RYX = (1/sample_batch_size) * np.dot(Ybatch, Xbatch.T)
                    elif method == "covariance":
                        Ybatch = self.update_Y_cov_based(Ybatch, Xbatch, muX, W, epsilon, (mu_start/np.sqrt(total_iteration+1)))
                        Ybatch = self.ProjectRowstoL1NormBall(Ybatch.T).T
                        muY = np.mean(Ybatch, axis = 1)
                        RYX = (1/sample_batch_size) * (np.dot(Ybatch, Xbatch.T) - np.outer(muY, muX))
                    W = np.dot(RYX, RXinv)
                    self.W = W
                    total_iteration += 1
                    if debugging:
                        if ((k % debug_iteration_point) == 0)  | (k == n_iterations_per_batch - 1):
                            Y = W @ X
                            Y_ = self.signed_and_permutation_corrected_sources(S.T,Y.T)
                            coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                            Y_ = coef_ * Y_
                            SIR = self.CalculateSIR(A, W)[0]
                            SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                            SINRlist.append(SINR)
                            SNRlist.append(self.snr(S.T,Y_))
                            SIRlist.append(SIR)
                            self.SIR_list = SIRlist
                            self.SINR_list = SINRlist
                            self.SNR_list = SNRlist
                            if plot_in_jupyter:
                                pl.clf()
                                pl.subplot(1,2,1)
                                pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                                pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                                pl.ylabel("SINR (dB)", fontsize = 35)
                                pl.title("SINR Behaviour", fontsize = 35)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)
                                pl.legend(fontsize=25)
                                pl.grid()
                                pl.subplot(1,2,2)
                                pl.plot(np.array(SNRlist), linewidth = 3)
                                pl.grid()
                                pl.title("Component SNR Check", fontsize = 35)
                                pl.ylabel("SNR (dB)", fontsize = 35)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)
                                clear_output(wait=True)
                                display(pl.gcf())  

    def fit_batch_nnsparse(self, X, batch_size = 10000, n_epochs = 1, n_iterations_per_batch = 500, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, drop_last_batch = True, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            plt.figure(figsize = (25, 10), dpi = 80)
        total_iteration = 1
        if drop_last_batch:
            m = (X.shape[1])//batch_size
        else:
            m = int(np.ceil((X.shape[1])/batch_size))
        for epoch_ in range(n_epochs):
            for kk in (range(m)):
                Xbatch = X[:,kk*batch_size:(kk+1)*batch_size]
                sample_batch_size = Xbatch.shape[1]
                if kk == 0:
                    Ybatch = np.zeros((self.s_dim, Xbatch.shape[1]))
                else:
                    Ybatch = self.W @ Xbatch
                if method == "correlation":
                    RX = (1/sample_batch_size) * np.dot(Xbatch, Xbatch.T)
                    RXinv = np.linalg.pinv(RX)
                elif method == "covariance":
                    muX = np.mean(Xbatch, axis = 1)
                    RX = (1/sample_batch_size) * (np.dot(Xbatch, Xbatch.T) - np.outer(muX, muX))
                    RXinv = np.linalg.pinv(RX)

                for k in tqdm(range(n_iterations_per_batch)):
                    if method == "correlation":
                        Ybatch = self.update_Y_corr_based(Ybatch, Xbatch, W, epsilon, (mu_start/np.sqrt((k+1)*kk+1)))
                        Ybatch = self.ProjectRowstoL1NormBall((Ybatch * (Ybatch >= 0)).T).T
                        RYX = (1/sample_batch_size) * np.dot(Ybatch, Xbatch.T)
                    elif method == "covariance":
                        Ybatch = self.update_Y_cov_based(Ybatch, Xbatch, muX, W, epsilon, (mu_start/np.sqrt(total_iteration+1)))
                        Ybatch = self.ProjectRowstoL1NormBall((Ybatch * (Ybatch >= 0)).T).T
                        muY = np.mean(Ybatch, axis = 1)
                        RYX = (1/sample_batch_size) * (np.dot(Ybatch, Xbatch.T) - np.outer(muY, muX))
                    W = np.dot(RYX, RXinv)
                    self.W = W
                    total_iteration += 1
                    if debugging:
                        if ((k % debug_iteration_point) == 0)  | (k == n_iterations_per_batch - 1):
                            Y = W @ X
                            Y_ = self.signed_and_permutation_corrected_sources(S.T,Y.T)
                            coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                            Y_ = coef_ * Y_
                            SIR = self.CalculateSIR(A, W)[0]
                            SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                            SINRlist.append(SINR)
                            SNRlist.append(self.snr(S.T,Y_))
                            SIRlist.append(SIR)
                            self.SIR_list = SIRlist
                            self.SINR_list = SINRlist
                            self.SNR_list = SNRlist
                            if plot_in_jupyter:
                                pl.clf()
                                pl.subplot(1,2,1)
                                pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                                pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                                pl.ylabel("SINR (dB)", fontsize = 35)
                                pl.title("SINR Behaviour", fontsize = 35)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)
                                pl.legend(fontsize=25)
                                pl.grid()
                                pl.subplot(1,2,2)
                                pl.plot(np.array(SNRlist), linewidth = 3)
                                pl.grid()
                                pl.title("Component SNR Check", fontsize = 35)
                                pl.ylabel("SNR (dB)", fontsize = 35)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)
                                clear_output(wait=True)
                                display(pl.gcf())  

    def fit_batch_simplex(self, X, batch_size = 10000, n_epochs = 1, n_iterations_per_batch = 500, epsilon = 1e-3, mu_start = 100, method = "correlation", debug_iteration_point = 1, drop_last_batch = True, plot_in_jupyter = False):
        
        W = self.W
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            plt.figure(figsize = (25, 10), dpi = 80)
        total_iteration = 1
        if drop_last_batch:
            m = (X.shape[1])//batch_size
        else:
            m = int(np.ceil((X.shape[1])/batch_size))
        for epoch_ in range(n_epochs):
            for kk in (range(m)):
                Xbatch = X[:,kk*batch_size:(kk+1)*batch_size]
                sample_batch_size = Xbatch.shape[1]
                if kk == 0:
                    Ybatch = np.zeros((self.s_dim, Xbatch.shape[1]))
                else:
                    Ybatch = self.W @ Xbatch
                if method == "correlation":
                    RX = (1/sample_batch_size) * np.dot(Xbatch, Xbatch.T)
                    RXinv = np.linalg.pinv(RX)
                elif method == "covariance":
                    muX = np.mean(Xbatch, axis = 1)
                    RX = (1/sample_batch_size) * (np.dot(Xbatch, Xbatch.T) - np.outer(muX, muX))
                    RXinv = np.linalg.pinv(RX)

                for k in tqdm(range(n_iterations_per_batch)):
                    if method == "correlation":
                        Ybatch = self.update_Y_corr_based(Ybatch, Xbatch, W, epsilon, (mu_start/np.sqrt((k+1)*kk+1)))
                        Ybatch = self.ProjectColstoSimplex(Ybatch)
                        RYX = (1/sample_batch_size) * np.dot(Ybatch, Xbatch.T)
                    elif method == "covariance":
                        Ybatch = self.update_Y_cov_based(Ybatch, Xbatch, muX, W, epsilon, (mu_start/np.sqrt(total_iteration+1)))
                        Ybatch = self.ProjectColstoSimplex(Ybatch)
                        muY = np.mean(Ybatch, axis = 1)
                        RYX = (1/sample_batch_size) * (np.dot(Ybatch, Xbatch.T) - np.outer(muY, muX))
                    W = np.dot(RYX, RXinv)
                    self.W = W
                    total_iteration += 1
                    if debugging:
                        if ((k % debug_iteration_point) == 0)  | (k == n_iterations_per_batch - 1):
                            Y = W @ X
                            Y_ = self.signed_and_permutation_corrected_sources(S.T,Y.T)
                            coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                            Y_ = coef_ * Y_
                            SIR = self.CalculateSIR(A, W)[0]
                            SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                            SINRlist.append(SINR)
                            SNRlist.append(self.snr(S.T,Y_))
                            SIRlist.append(SIR)
                            self.SIR_list = SIRlist
                            self.SINR_list = SINRlist
                            self.SNR_list = SNRlist
                            if plot_in_jupyter:
                                pl.clf()
                                pl.subplot(1,2,1)
                                pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                                pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                                pl.ylabel("SINR (dB)", fontsize = 35)
                                pl.title("SINR Behaviour", fontsize = 35)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)
                                pl.legend(fontsize=25)
                                pl.grid()
                                pl.subplot(1,2,2)
                                pl.plot(np.array(SNRlist), linewidth = 3)
                                pl.grid()
                                pl.title("Component SNR Check", fontsize = 35)
                                pl.ylabel("SNR (dB)", fontsize = 35)
                                pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                                pl.xticks(fontsize=45)
                                pl.yticks(fontsize=45)
                                clear_output(wait=True)
                                display(pl.gcf())  

class PMF:

    """
    Implementation of batch Polytopic Matrix Factorization
    Parameters:
    =================================
    s_dim          -- Dimension of the sources
    x_dim          -- Dimension of the mixtures
    H              -- Feedforward Synapses
    
    Methods:
    ==================================
    fit_batch_antisparse
    fit_batch_nnantisparse

    """
    
    def __init__(self, s_dim, x_dim, H = None, set_ground_truth = False, Sgt = None, Agt = None):
        if H is not None:
            assert H.shape == (x_dim, s_dim), "The shape of the initial guess H must be (s_dim, x_dim) = (%d,%d)" % (s_dim, x_dim)
            H = H
        else:
            H = np.random.randn(x_dim, s_dim)
            
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.H = H
        ### Ground Truth Sources and Mixing Matrix For Debugging
        self.set_ground_truth = set_ground_truth
        self.Sgt = Sgt # Ground Truth Sources
        self.Agt = Agt # Ground Truth Mixing Matrix
        self.SIR_list = []
        self.SINR_list = []
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

    def CalculateSINR(self, Out, S, compute_permutation = True):
        r=S.shape[0]
        if compute_permutation:
            G=np.dot(Out-np.reshape(np.mean(Out,1),(r,1)),np.linalg.pinv(S-np.reshape(np.mean(S,1),(r,1))))
            indmax=np.argmax(np.abs(G),1)
        else:
            G=np.dot(Out-np.reshape(np.mean(Out,1),(r,1)),np.linalg.pinv(S-np.reshape(np.mean(S,1),(r,1))))
            indmax = np.arange(0,r)
        GG=np.zeros((r,r))
        for kk in range(r):
            GG[kk,indmax[kk]]=np.dot(Out[kk,:]-np.mean(Out[kk,:]),S[indmax[kk],:].T-np.mean(S[indmax[kk],:]))/np.dot(S[indmax[kk],:]-np.mean(S[indmax[kk],:]),S[indmax[kk],:].T-np.mean(S[indmax[kk],:]))#(G[kk,indmax[kk]])
        ZZ=GG@(S-np.reshape(np.mean(S,1),(r,1)))+np.reshape(np.mean(Out,1),(r,1))
        E=Out-ZZ
        MSE=np.linalg.norm(E,'fro')**2
        SigPow=np.linalg.norm(ZZ,'fro')**2
        SINR=(SigPow/MSE)
        return SINR,SigPow,MSE,G

    def snr(self, S_original, S_noisy):
        N_hat = S_original - S_noisy
        N_P = (N_hat ** 2).sum(axis = 0)
        S_P = (S_original ** 2).sum(axis = 0)
        snr = 10 * np.log10(S_P / N_P)
        return snr

    def outer_prod_broadcasting(self, A, B):
        """Broadcasting trick"""
        return A[...,None]*B[:,None]

    def find_permutation_between_source_and_estimation(self, S, Y):
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

    def signed_and_permutation_corrected_sources(self, S, Y):
        perm = self.find_permutation_between_source_and_estimation(S,Y)
        return np.sign((Y[:,perm] * S).sum(axis = 0)) * Y[:,perm]

    @staticmethod
    @njit
    def ProjectOntoLInfty(X):
        return X*(X>=-1.0)*(X<=1.0)+(X>1.0)*1.0-1.0*(X<-1.0)
    
    @staticmethod
    @njit
    def ProjectOntoNNLInfty(X):
        return X*(X>=0.0)*(X<=1.0)+(X>1.0)*1.0#-0.0*(X<0.0)
        
    def ProjectRowstoL1NormBall(self, H):
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

    def ProjectColstoSimplex(self, v, z=1):
        """v array of shape (n_features, n_samples)."""
        p, n = v.shape
        u = np.sort(v, axis=0)[::-1, ...]
        pi = np.cumsum(u, axis=0) - z
        ind = (np.arange(p) + 1).reshape(-1, 1)
        mask = (u - pi / ind) > 0
        rho = p - 1 - np.argmax(mask[::-1, ...], axis=0)
        theta = pi[tuple([rho, np.arange(n)])] / (rho + 1)
        w = np.maximum(v - theta, 0)
        return w

    def fit_batch_nnantisparse(self, Y, n_iterations = 1000, lambda_ = 0.01, tau = 1e-8, debug_iteration_point = 1, plot_in_jupyter = False):
        
        H = self.H
        debugging = self.set_ground_truth
        Identity = np.eye(self.s_dim)
        F = Identity.copy()
        q = 1
        assert Y.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = Y.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            Sgt = self.Sgt
            Agt = self.Agt
            plt.figure(figsize = (25, 10), dpi = 80)

        S = np.zeros((self.s_dim, samples))
        X = S.copy()
        for k in range(n_iterations):
            #### PMF Algorithm #################
            Sprev = S.copy()
            qprev = q + 0.0
            # print(S.shape)
            # print(H.shape)
            # print(Y.shape)
            # print(X.shape)
            S = self.ProjectOntoNNLInfty(X - H.T @ (Y - H @ X))
            q = (1 + np.sqrt(1 + q ** 2))/2.0
            X = S + ((qprev - 1) / q) * (S - Sprev)
            H = Y @ S.T @ (S @ S.T + lambda_ * F)
            F = np.linalg.pinv(H.T @ H + tau * Identity)
            if debugging:
                if ((k % debug_iteration_point) == 0)  | (k == n_iterations - 1):
                    self.H = H
                    S_ = self.signed_and_permutation_corrected_sources(Sgt.T,S.T)
                    coef_ = (S_ * Sgt.T).sum(axis = 0) / (S_ * S_).sum(axis = 0)
                    S_ = coef_ * S_
                    self.S_ = S_
                    # SIR = self.CalculateSIR(Agt, H)[0]
                    SINR = 10*np.log10(self.CalculateSINR(S_.T, Sgt)[0])
                    SINRlist.append(SINR)
                    SNRlist.append(self.snr(S.T,S_))
                    # SIRlist.append(SIR)
                    self.SIR_list = SIRlist
                    self.SINR_list = SINRlist
                    self.SNR_list = SNRlist
                    if plot_in_jupyter:
                        pl.clf()
                        pl.subplot(1,2,1)
                        pl.plot(np.array(SIRlist), linewidth = 3, label = "SIR")
                        pl.plot(np.array(SINRlist), linewidth = 3, label = "SINR")
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.ylabel("SINR (dB)", fontsize = 35)
                        pl.title("SINR Behaviour", fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        pl.legend(fontsize=25)
                        pl.grid()
                        pl.subplot(1,2,2)
                        pl.plot(np.array(SNRlist), linewidth = 3)
                        pl.grid()
                        pl.title("Component SNR Check", fontsize = 35)
                        pl.ylabel("SNR (dB)", fontsize = 35)
                        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                        pl.xticks(fontsize=45)
                        pl.yticks(fontsize=45)
                        clear_output(wait=True)
                        display(pl.gcf())  
        self.H = H

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
                if i_sample < ZERO_CHECK_INTERVAL:
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

    def CalculateSINR(self, Out, S, compute_permutation = True):
        r=S.shape[0]
        if compute_permutation:
            G=np.dot(Out-np.reshape(np.mean(Out,1),(r,1)),np.linalg.pinv(S-np.reshape(np.mean(S,1),(r,1))))
            indmax=np.argmax(np.abs(G),1)
        else:
            G=np.dot(Out-np.reshape(np.mean(Out,1),(r,1)),np.linalg.pinv(S-np.reshape(np.mean(S,1),(r,1))))
            indmax = np.arange(0,r)
        GG=np.zeros((r,r))
        for kk in range(r):
            GG[kk,indmax[kk]]=np.dot(Out[kk,:]-np.mean(Out[kk,:]),S[indmax[kk],:].T-np.mean(S[indmax[kk],:]))/np.dot(S[indmax[kk],:]-np.mean(S[indmax[kk],:]),S[indmax[kk],:].T-np.mean(S[indmax[kk],:]))#(G[kk,indmax[kk]])
        ZZ=GG@(S-np.reshape(np.mean(S,1),(r,1)))+np.reshape(np.mean(Out,1),(r,1))
        E=Out-ZZ
        MSE=np.linalg.norm(E,'fro')**2
        SigPow=np.linalg.norm(ZZ,'fro')**2
        SINR=(SigPow/MSE)
        return SINR,SigPow,MSE,G

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
    
    def snr(self, S_original, S_noisy):
        N_hat = S_original - S_noisy
        N_P = (N_hat ** 2).sum(axis = 0)
        S_P = (S_original ** 2).sum(axis = 0)
        snr = 10 * np.log10(S_P / N_P)
        return snr

    def outer_prod_broadcasting(self, A, B):
        """Broadcasting trick"""
        return A[...,None]*B[:,None]

    def find_permutation_between_source_and_estimation(self, S, Y):
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

    def signed_and_permutation_corrected_sources(self, S, Y):
        perm = self.find_permutation_between_source_and_estimation(S,Y)
        return np.sign((Y[:,perm] * S).sum(axis = 0)) * Y[:,perm]

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
         
    def fit_batch_antisparse(self, X, n_epochs = 2, neural_dynamic_iterations = 250, lr_start = 0.9, whiten = False, whiten_type = 2, shuffle = False, verbose = True, debug_iteration_point = 1000, plot_in_jupyter = False):
        
        lambda_, beta, mu_F, gamma_hat, F, B = self.lambda_, self.beta, self.mu_F, self.gamma_hat, self.F, self.B
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        if debugging:
            SIRlist = []
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            plt.figure(figsize = (25, 10), dpi = 80)

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
                        Y_ = W @ X
                        Y_ = self.signed_and_permutation_corrected_sources(S.T,Y_.T)
                        coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                        Y_ = coef_ * Y_
                        SIR = self.CalculateSIR(A, W)[0]
                        SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                        SIRlist.append(SIR)
                        SINRlist.append(SINR)
                        SNRlist.append(self.snr(S.T,Y_))
                        self.SIR_list = SIRlist
                        self.SINR_list = SINRlist
                        self.SNR_list = SNRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.subplot(1,2,1)
                            pl.plot(np.arange(1,len(self.SINR_list)+1), np.array(SIRlist), linewidth = 3, label = "SIR")
                            pl.plot(np.arange(1,len(self.SINR_list)+1), np.array(SINRlist), linewidth = 3, label = "SINR")
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                            pl.ylabel("SINR (dB)", fontsize = 35)
                            pl.title("SINR Behaviour", fontsize = 35)
                            pl.xticks(fontsize=45)
                            pl.yticks(fontsize=45)
                            pl.legend(fontsize=25)
                            pl.grid()
                            pl.subplot(1,2,2)
                            pl.plot(np.arange(1,len(self.SINR_list)+1), np.array(SNRlist), linewidth = 3)
                            pl.grid()
                            pl.title("Component SNR Check", fontsize = 35)
                            pl.ylabel("SNR (dB)", fontsize = 35)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                            pl.xticks(fontsize=45)
                            pl.yticks(fontsize=45)
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
            SINRlist = []
            SNRlist = []
            S = self.S
            A = self.A
            plt.figure(figsize = (25, 10), dpi = 80)

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
                        Y_ = W @ X
                        Y_ = self.signed_and_permutation_corrected_sources(S.T,Y_.T)
                        coef_ = (Y_ * S.T).sum(axis = 0) / (Y_ * Y_).sum(axis = 0)
                        Y_ = coef_ * Y_
                        SIR = self.CalculateSIR(A, W)[0]
                        SINR = 10*np.log10(self.CalculateSINR(Y_.T, S)[0])
                        SIRlist.append(SIR)
                        SINRlist.append(SINR)
                        SNRlist.append(self.snr(S.T,Y_))
                        self.SIR_list = SIRlist
                        self.SINR_list = SINRlist
                        self.SNR_list = SNRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.subplot(1,2,1)
                            pl.plot(np.arange(1,len(self.SINR_list)+1), np.array(SIRlist), linewidth = 3, label = "SIR")
                            pl.plot(np.arange(1,len(self.SINR_list)+1), np.array(SINRlist), linewidth = 3, label = "SINR")
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                            pl.ylabel("SINR (dB)", fontsize = 35)
                            pl.title("SINR Behaviour", fontsize = 35)
                            pl.xticks(fontsize=45)
                            pl.yticks(fontsize=45)
                            pl.legend(fontsize=25)
                            pl.grid()
                            pl.subplot(1,2,2)
                            pl.plot(np.arange(1,len(self.SINR_list)+1), np.array(SNRlist), linewidth = 3)
                            pl.grid()
                            pl.title("Component SNR Check", fontsize = 35)
                            pl.ylabel("SNR (dB)", fontsize = 35)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 35)
                            pl.xticks(fontsize=45)
                            pl.yticks(fontsize=45)
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

def ProjectOntoLInfty(X, thresh = 1.0):
    return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)

def ProjectOntoNNLInfty(X, thresh = 1.0):
    return X*(X>=0)*(X<=thresh)+(X>thresh)*thresh

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

def ProjectColstoSimplex(v, z=1):
    """v array of shape (n_features, n_samples)."""
    p, n = v.shape
    u = np.sort(v, axis=0)[::-1, ...]
    pi = np.cumsum(u, axis=0) - z
    ind = (np.arange(p) + 1).reshape(-1, 1)
    mask = (u - pi / ind) > 0
    rho = p - 1 - np.argmax(mask[::-1, ...], axis=0)
    theta = pi[tuple([rho, np.arange(n)])] / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w

def projection_simplex(V, z=1, axis=None):
    """
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    """
    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()
       
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

def CalculateSINR(Out, S, compute_permutation = True):
    r=S.shape[0]
    if compute_permutation:
        G=np.dot(Out-np.reshape(np.mean(Out,1),(r,1)),np.linalg.pinv(S-np.reshape(np.mean(S,1),(r,1))))
        indmax=np.argmax(np.abs(G),1)
    else:
        G=np.dot(Out-np.reshape(np.mean(Out,1),(r,1)),np.linalg.pinv(S-np.reshape(np.mean(S,1),(r,1))))
        indmax = np.arange(0,r)
    GG=np.zeros((r,r))
    for kk in range(r):
        GG[kk,indmax[kk]]=np.dot(Out[kk,:]-np.mean(Out[kk,:]),S[indmax[kk],:].T-np.mean(S[indmax[kk],:]))/np.dot(S[indmax[kk],:]-np.mean(S[indmax[kk],:]),S[indmax[kk],:].T-np.mean(S[indmax[kk],:]))#(G[kk,indmax[kk]])
    ZZ=GG@(S-np.reshape(np.mean(S,1),(r,1)))+np.reshape(np.mean(Out,1),(r,1))
    E=Out-ZZ
    MSE=np.linalg.norm(E,'fro')**2
    SigPow=np.linalg.norm(ZZ,'fro')**2
    SINR=(SigPow/MSE)
    return SINR,SigPow,MSE,G

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

def addWGN(signal, SNR, return_noise = False, print_resulting_SNR = False):
    """
    Adding white Gaussian Noise to the input signal
    signal              : Input signal, numpy array of shape (number of sources, number of samples)
                          If your signal is a 1D numpy array of shape (number of samples, ), then reshape it 
                          by signal.reshape(1,-1) before giving it as input to this function
    SNR                 : Desired input signal to noise ratio
    print_resulting_SNR : If you want to print the numerically calculated SNR, pass it as True
    
    Returns
    ============================
    signal_noisy        : Output signal which is the sum of input signal and additive noise
    noise               : Returns the added noise
    """
    sigpow = np.mean(signal**2, axis = 1)
    noisepow = 10 **(-SNR/10) * sigpow
    noise =  np.sqrt(noisepow)[:,np.newaxis] * np.random.randn(signal.shape[0], signal.shape[1])
    signal_noisy = signal + noise
    if print_resulting_SNR:
        SNRinp = 10 * np.log10(np.sum(np.mean(signal**2, axis = 1)) / np.sum(np.mean(noise**2, axis = 1)))
        print("Input SNR is : {}".format(SNRinp))
    if return_noise:
        return signal_noisy, noise
    else:
        return signal_noisy

def WSM_Mixing_Scenario(S, NumberofMixtures = None, INPUT_STD = None):
    NumberofSources = S.shape[0]
    if INPUT_STD is None:
        INPUT_STD = S.std()
    if NumberofMixtures is None:
        NumberofMixtures = NumberofSources
    A = np.random.standard_normal(size=(NumberofMixtures,NumberofSources))
    X = A @ S
    for M in range(A.shape[0]):
        stdx = np.std(X[M,:])
        A[M,:] = A[M,:]/stdx * INPUT_STD
        
    return A, X

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

def generate_uniform_points_in_polytope(polytope_vertices, size):
    """"
    polytope_vertices : vertex matrix of shape (n_dim, n_vertices)

    return:
        Samples of shape (n_dim, size)
    """
    polytope_vertices = polytope_vertices.T
    dims = polytope_vertices.shape[-1]
    hull = polytope_vertices[ConvexHull(polytope_vertices).vertices]
    deln = hull[Delaunay(hull).simplices]

    vols = np.abs(det(deln[:, :dims, :] - deln[:, dims:, :])) / np.math.factorial(dims)    
    sample = np.random.choice(len(vols), size = size, p = vols / vols.sum())

    return np.einsum('ijk, ij -> ik', deln[sample], dirichlet.rvs([1]*(dims + 1), size = size)).T

def generate_practical_polytope(dim, antisparse_dims, nonnegative_dims, relative_sparse_dims_list):
    A = []
    b = []
    for j in antisparse_dims:
        row1 = [0 for _ in range(dim)]
        row2 = row1.copy()
        row1[j] = 1
        A.append(row1)
        b.append(1)
        row2[j] = -1
        A.append(row2)
        b.append(1)

    for j in nonnegative_dims:
        row1 = [0 for _ in range(dim)]
        row2 = row1.copy()
        row1[j] = 1
        A.append(row1)
        b.append(1)
        row2[j] = -1
        A.append(row2)
        b.append(0)

    for relative_sparse_dims in relative_sparse_dims_list:
        row = np.zeros(dim)
        pm_one = [[1,-1] for _ in range(relative_sparse_dims.shape[0])]
        for i in itertools.product(*pm_one):
            row_copy = row.copy()
            row_copy[relative_sparse_dims] = i
            A.append(list(row_copy))
            b.append(1)
    A = np.array(A)
    b = np.array(b)
    vertices = pypoman.compute_polytope_vertices(A, b)
    V = np.array([list(v) for v in vertices]).T
    return (A,b), V
