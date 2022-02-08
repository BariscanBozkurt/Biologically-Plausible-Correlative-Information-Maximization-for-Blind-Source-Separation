import numpy as np
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
            STLAMBD = max(STLAMBD + 1* dval, 0)

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
                NumberofOutputIterations_ = np.int64(5 + np.min([np.ceil(i_sample / 50000.0),neural_dynamic_iterations]))
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
                        SIR,_ = CalculateSIR(A, Wf)
                        SIR_list.append(SIR)

                        if plot_convergence_plot:
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

