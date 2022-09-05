"""
Title: PMFBSS.py

Code Writer: Bariscan Bozkurt (Koç University - EEE & Mathematics)

Date: 06.06.2022
"""

import itertools
import math
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import pypoman
import scipy
from IPython import display as display1
from IPython.display import Latex, Math, clear_output, display
from matplotlib.pyplot import draw, plot, show
from numba import jit, njit
from numpy.linalg import det
from scipy import linalg
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import chi2, dirichlet, invgamma, t
from tqdm import tqdm

warnings.filterwarnings("ignore")


class OnlinePMF:
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

    fit_next_antisparse(y_online)     -- Updates the network parameters for one data point y_online

    fit_batch_antisparse(Y_batch)     -- Updates the network parameters for given batch data Y_batch (but in online manner)

    """

    def __init__(
        self,
        s_dim,
        y_dim,
        lambda_=0.999,
        muW=0.03,
        beta=5,
        W=None,
        B=None,
        neural_OUTPUT_COMP_TOL=1e-6,
        set_ground_truth=False,
        Sgt=None,
        A=None,
    ):
        if W is not None:
            assert W.shape == (
                s_dim,
                y_dim,
            ), "The shape of the initial guess F must be (s_dim, x_dim) = (%d,%d)" % (
                s_dim,
                y_dim,
            )
            W = W
        else:
            W = np.random.randn(s_dim, y_dim)
            W = W / np.sqrt(np.sum(np.abs(W) ** 2, axis=1)).reshape(s_dim, 1)
            W = np.eye(s_dim, y_dim)

        if B is not None:
            assert B.shape == (
                s_dim,
                s_dim,
            ), "The shape of the initial guess B must be (s_dim, s_dim) = (%d,%d)" % (
                s_dim,
                s_dim,
            )
            B = B
        else:
            B = 5 * np.eye(s_dim)

        self.s_dim = s_dim
        self.y_dim = y_dim
        self.lambda_ = lambda_
        self.beta = beta
        self.muW = muW
        self.gamma_hat = (1 - lambda_) / lambda_
        self.W = W
        self.B = B
        self.neural_OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL
        self.set_ground_truth = set_ground_truth
        self.SIRlist = []
        self.Sgt = Sgt  # Ground truth sources for debugging
        self.A = A  # Ground truth mixing matrix

    # Calculate SIR Function
    def CalculateSIR(self, H, pH, return_db=True):
        G = pH @ H
        Gmax = np.diag(np.max(abs(G), axis=1))
        P = 1.0 * ((np.linalg.inv(Gmax) @ np.abs(G)) > 0.99)
        T = G @ P.T
        rankP = np.linalg.matrix_rank(P)
        diagT = np.diag(T)
        # Signal Power
        sigpow = np.linalg.norm(diagT, 2) ** 2
        # Interference Power
        intpow = np.linalg.norm(T, "fro") ** 2 - sigpow
        SIRV = sigpow / intpow
        # SIRV=np.linalg.norm((np.diag(T)))**2/(np.linalg.norm(T,'fro')**2-np.linalg.norm(np.diag(T))**2)
        if return_db:
            SIRV = 10 * np.log10(sigpow / intpow)

        return SIRV, rankP

    def whiten_signal(self, X, mean_normalize=True, type_=3):
        """
        Input : X  ---> Input signal to be whitened
        type_ : Defines the type for preprocesing matrix. type_ = 1 and 2 uses eigenvalue decomposition whereas type_ = 3 uses SVD.
        Output: X_white  ---> Whitened signal, i.e., X_white = W_pre @ X where W_pre = (R_x^0.5)^+ (square root of sample correlation matrix)
        """
        if mean_normalize:
            X = X - np.mean(X, axis=0, keepdims=True)

        cov = np.cov(X.T)

        if type_ == 3:  # Whitening using singular value decomposition
            U, S, V = np.linalg.svd(cov)
            d = np.diag(1.0 / np.sqrt(S))
            W_pre = np.dot(U, np.dot(d, U.T))

        else:  # Whitening using eigenvalue decomposition
            d, S = np.linalg.eigh(cov)
            D = np.diag(d)

            D_sqrt = np.sqrt(D * (D > 0))

            if type_ == 1:  # Type defines how you want W_pre matrix to be
                W_pre = np.linalg.pinv(S @ D_sqrt)
            elif type_ == 2:
                W_pre = np.linalg.pinv(S @ D_sqrt @ S.T)

        X_white = (W_pre @ X.T).T

        return X_white, W_pre

    def ProjectOntoLInfty(self, X):

        return X * (X >= -1.0) * (X <= 1.0) + (X > 1.0) * 1.0 - 1.0 * (X < -1.0)

    def compute_overall_mapping(self, return_mapping=True):
        W, B, gamma_hat, beta = self.W, self.B, self.gamma_hat, self.beta
        Wf = np.linalg.pinv(gamma_hat * B - beta * np.eye(self.s_dim)) @ (beta * W)
        self.Wf = Wf
        if return_mapping:
            return Wf
        else:
            return None

    @staticmethod
    @njit
    def run_neural_dynamics_antisparse(
        y,
        s,
        W,
        B,
        beta,
        gamma_hat,
        lr_start=0.9,
        lr_stop=1e-15,
        neural_dynamic_iterations=100,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):
        def ProjectOntoLInfty(X, thresh=1.0):
            return (
                X * (X >= -thresh) * (X <= thresh)
                + (X > thresh) * thresh
                - thresh * (X < -thresh)
            )

        ske = np.dot(W, y)
        v = gamma_hat * s
        M = B + (1 / gamma_hat) * np.eye(s.shape[0])
        for j in range(neural_dynamic_iterations):
            mu_s = max(lr_start / (j + 1), lr_stop)
            s_old = s.copy()
            e = ske - s
            grads = -s + gamma_hat * M @ s + beta * e
            s = s + mu_s * grads
            s = ProjectOntoLInfty(s)

            if np.linalg.norm(s - s_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(s):
                break
        return s

    @staticmethod
    @njit
    def run_neural_dynamics_nnantisparse(
        y,
        s,
        W,
        B,
        beta,
        gamma_hat,
        lr_start=0.9,
        lr_stop=1e-15,
        neural_dynamic_iterations=100,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):
        def ProjectOntoNNLInfty(X, thresh=1.0):
            return (
                X * (X >= 0.0) * (X <= thresh) + (X > thresh) * thresh
            )  # -thresh*(X<-thresh)

        ske = np.dot(W, y)
        v = gamma_hat * s
        M = B + (1 / gamma_hat) * np.eye(s.shape[0])
        for j in range(neural_dynamic_iterations):
            mu_s = max(lr_start / (j + 1), lr_stop)
            s_old = s.copy()
            e = ske - s
            grads = -s + gamma_hat * M @ s + beta * e
            s = s + mu_s * grads
            s = ProjectOntoNNLInfty(s)

            if np.linalg.norm(s - s_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(s):
                break
        return s

    @staticmethod
    @njit
    def run_neural_dynamics_sparse(
        y,
        s,
        W,
        B,
        beta,
        gamma_hat,
        lr_start=0.9,
        lr_stop=1e-15,
        neural_dynamic_iterations=100,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):
        STLAMBD = 0
        dval = 0
        ske = np.dot(W, y)
        v = gamma_hat * s
        M = B + (1 / gamma_hat) * np.eye(s.shape[0])
        for j in range(neural_dynamic_iterations):
            mu_s = max(lr_start / (j + 1), lr_stop)
            s_old = s.copy()
            e = ske - s
            grads = -s + gamma_hat * M @ s + beta * e
            s = s + mu_s * grads
            # SOFT THRESHOLDING
            s_absolute = np.abs(s)
            s_sign = np.sign(s)

            s = (s_absolute > STLAMBD) * (s_absolute - STLAMBD) * s_sign
            dval = np.linalg.norm(s, 1) - 1
            STLAMBD = max(STLAMBD + 1 * dval, 0)

            if np.linalg.norm(s - s_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(s):
                break
        return s

    @staticmethod
    @njit
    def run_neural_dynamics_nnsparse(
        y,
        s,
        W,
        B,
        beta,
        gamma_hat,
        lr_start=0.9,
        lr_stop=1e-15,
        neural_dynamic_iterations=100,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):
        STLAMBD = 0
        dval = 0
        ske = np.dot(W, y)
        v = gamma_hat * s
        M = B + (1 / gamma_hat) * np.eye(s.shape[0])
        for j in range(neural_dynamic_iterations):
            mu_s = max(lr_start / (j + 1), lr_stop)
            s_old = s.copy()
            e = ske - s
            grads = -s + gamma_hat * M @ s + beta * e
            s = s + mu_s * grads
            s = np.maximum(s - STLAMBD, 0)

            dval = np.sum(s) - 1
            STLAMBD = max(STLAMBD + 1 * dval, 0)

            if np.linalg.norm(s - s_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(s):
                break
        return s

    @staticmethod
    @njit
    def run_neural_dynamics_simplex(
        y,
        s,
        W,
        B,
        beta,
        gamma_hat,
        lr_start=0.9,
        lr_stop=1e-15,
        neural_dynamic_iterations=100,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):
        STLAMBD = 0
        dval = 0
        ske = np.dot(W, y)
        v = gamma_hat * s
        M = B + (1 / gamma_hat) * np.eye(s.shape[0])
        for j in range(neural_dynamic_iterations):
            mu_s = max(lr_start / (j + 1), lr_stop)
            s_old = s.copy()
            e = ske - s
            grads = -s + gamma_hat * M @ s + beta * e
            s = s + mu_s * grads
            s = np.maximum(s - STLAMBD, 0)

            dval = np.sum(s) - 1
            STLAMBD = STLAMBD + 0.1 * dval

            if np.linalg.norm(s - s_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(s):
                break
        return s

    @staticmethod
    @njit
    def run_neural_dynamics_general_polytope(
        y,
        s,
        signed_dims,
        nn_dims,
        sparse_dims_list,
        W,
        B,
        beta,
        gamma_hat,
        lr_start=0.9,
        lr_stop=1e-15,
        neural_dynamic_iterations=100,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):
        def ProjectOntoLInfty(X, thresh=1.0):
            return (
                X * (X >= -thresh) * (X <= thresh)
                + (X > thresh) * thresh
                - thresh * (X < -thresh)
            )

        def ProjectOntoNNLInfty(X, thresh=1.0):
            return X * (X >= 0.0) * (X <= thresh) + (X > thresh) * thresh

        def SoftThresholding(X, thresh):
            X_absolute = np.abs(X)
            X_sign = np.sign(X)
            X_thresholded = (X_absolute > thresh) * (X_absolute - thresh) * X_sign
            return X_thresholded

        def ReLU(X):
            return np.maximum(X, 0)

        def loop_intersection(lst1, lst2):
            result = []
            for element1 in lst1:
                for element2 in lst2:
                    if element1 == element2:
                        result.append(element1)
            return result

        STLAMBD_list = np.zeros(len(sparse_dims_list))
        ske = np.dot(W, y)
        v = gamma_hat * s
        M = B + (1 / gamma_hat) * np.eye(s.shape[0])
        for j in range(neural_dynamic_iterations):
            mu_s = max(lr_start / (j + 1), lr_stop)
            s_old = s.copy()
            e = ske - s
            grads = -s + gamma_hat * M @ s + beta * e
            s = s + mu_s * grads
            if sparse_dims_list[0][0] != -1:
                for ss, sparse_dim in enumerate(sparse_dims_list):
                    # s[sparse_dim] = SoftThresholding(s[sparse_dim], STLAMBD_list[ss])
                    # STLAMBD_list[ss] = max(STLAMBD_list[ss] + (np.linalg.norm(s[sparse_dim],1) - 1), 0)
                    if signed_dims[0] != -1:
                        s[
                            np.array(loop_intersection(sparse_dim, signed_dims))
                        ] = SoftThresholding(
                            s[np.array(loop_intersection(sparse_dim, signed_dims))],
                            STLAMBD_list[ss],
                        )
                    if nn_dims[0] != -1:
                        s[np.array(loop_intersection(sparse_dim, nn_dims))] = ReLU(
                            s[np.array(loop_intersection(sparse_dim, nn_dims))]
                            - STLAMBD_list[ss]
                        )
                    STLAMBD_list[ss] = max(
                        STLAMBD_list[ss] + (np.linalg.norm(s[sparse_dim], 1) - 1), 0
                    )
            if signed_dims[0] != -1:
                s[signed_dims] = ProjectOntoLInfty(s[signed_dims])
            if nn_dims[0] != -1:
                s[nn_dims] = ProjectOntoNNLInfty(s[nn_dims])

            if np.linalg.norm(s - s_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(s):
                break
        return s

    def fit_batch_general_polytope(
        self,
        Y,
        signed_dims,
        nn_dims,
        sparse_dims_list,
        n_epochs=1,
        neural_dynamic_iterations=250,
        neural_lr_start=0.9,
        neural_lr_stop=1e-3,
        shuffle=False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):

        lambda_, beta, muW, gamma_hat, W, B = (
            self.lambda_,
            self.beta,
            self.muW,
            self.gamma_hat,
            self.W,
            self.B,
        )
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
        neural_dynamics_dummy = False
        assert Y.shape[0] == self.y_dim, "You must input the transpose"

        samples = Y.shape[1]

        if debugging:
            SIRlist = []
            Sgt = self.Sgt
            A = self.A

        S = np.random.randn(self.s_dim, samples)  # Initial source estimates

        if shuffle:
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        if signed_dims.size == 0:
            signed_dims = np.array([-1])
        if nn_dims.size == 0:
            nn_dims = np.array([-1])
        if not sparse_dims_list:
            sparse_dims_list = [np.array([-1])]

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                y_current = Y[:, idx[i_sample]]
                s = np.zeros(self.s_dim)

                s = self.run_neural_dynamics_general_polytope(
                    y_current,
                    s,
                    signed_dims,
                    nn_dims,
                    sparse_dims_list,
                    W,
                    B,
                    beta,
                    gamma_hat,
                    lr_start=neural_lr_start,
                    lr_stop=neural_lr_stop,
                    neural_dynamic_iterations=neural_dynamic_iterations,
                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                )

                e = W @ y_current - s

                W = W - muW * beta * np.outer(e, y_current)

                z = B @ s
                B = (1 / lambda_) * (B - gamma_hat * np.outer(z, z))
                # Record the seperated signal
                S[:, idx[i_sample]] = s

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.B = B
                        Wf = self.compute_overall_mapping(return_mapping=True)
                        SIR = self.CalculateSIR(A, Wf)[0]
                        SIRlist.append(SIR)
                        self.SIR_list = SIRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth=3)
                            pl.xlabel(
                                "Number of Iterations / {}".format(
                                    debug_iteration_point
                                ),
                                fontsize=15,
                            )
                            pl.ylabel("SIR (dB)", fontsize=15)
                            pl.title("SIR Behaviour", fontsize=15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())
        self.W = W
        self.B = B

    def fit_batch_antisparse(
        self,
        Y,
        n_epochs=1,
        neural_dynamic_iterations=250,
        neural_lr_start=0.9,
        neural_lr_stop=1e-3,
        shuffle=False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):

        lambda_, beta, muW, gamma_hat, W, B = (
            self.lambda_,
            self.beta,
            self.muW,
            self.gamma_hat,
            self.W,
            self.B,
        )
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        assert Y.shape[0] == self.y_dim, "You must input the transpose"

        samples = Y.shape[1]

        if debugging:
            SIRlist = []
            Sgt = self.Sgt
            A = self.A

        S = np.random.randn(self.s_dim, samples)  # Initial source estimates

        if shuffle:
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                y_current = Y[:, idx[i_sample]]
                s = np.zeros(self.s_dim)

                s = self.run_neural_dynamics_antisparse(
                    y_current,
                    s,
                    W,
                    B,
                    beta,
                    gamma_hat,
                    lr_start=neural_lr_start,
                    lr_stop=neural_lr_stop,
                    neural_dynamic_iterations=neural_dynamic_iterations,
                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                )

                e = W @ y_current - s

                W = W - muW * beta * np.outer(e, y_current)

                z = B @ s
                B = (1 / lambda_) * (B - gamma_hat * np.outer(z, z))
                # Record the seperated signal
                S[:, idx[i_sample]] = s

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.B = B
                        Wf = self.compute_overall_mapping(return_mapping=True)
                        SIR = self.CalculateSIR(A, Wf)[0]
                        SIRlist.append(SIR)
                        self.SIR_list = SIRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth=3)
                            pl.xlabel(
                                "Number of Iterations / {}".format(
                                    debug_iteration_point
                                ),
                                fontsize=15,
                            )
                            pl.ylabel("SIR (dB)", fontsize=15)
                            pl.title("SIR Behaviour", fontsize=15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())
        self.W = W
        self.B = B

    def fit_batch_nnantisparse(
        self,
        Y,
        n_epochs=1,
        neural_dynamic_iterations=250,
        neural_lr_start=0.9,
        neural_lr_stop=1e-3,
        shuffle=False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):

        lambda_, beta, muW, gamma_hat, W, B = (
            self.lambda_,
            self.beta,
            self.muW,
            self.gamma_hat,
            self.W,
            self.B,
        )
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        assert Y.shape[0] == self.y_dim, "You must input the transpose"

        samples = Y.shape[1]

        if debugging:
            SIRlist = []
            Sgt = self.Sgt
            A = self.A

        S = np.random.randn(self.s_dim, samples)  # Initial source estimates

        if shuffle:
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                y_current = Y[:, idx[i_sample]]
                s = np.zeros(self.s_dim)

                s = self.run_neural_dynamics_nnantisparse(
                    y_current,
                    s,
                    W,
                    B,
                    beta,
                    gamma_hat,
                    lr_start=neural_lr_start,
                    lr_stop=neural_lr_stop,
                    neural_dynamic_iterations=neural_dynamic_iterations,
                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                )

                e = W @ y_current - s

                W = W - muW * beta * np.outer(e, y_current)

                z = B @ s
                B = (1 / lambda_) * (B - gamma_hat * np.outer(z, z))
                # Record the seperated signal
                S[:, idx[i_sample]] = s

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.B = B
                        Wf = self.compute_overall_mapping(return_mapping=True)
                        SIR = self.CalculateSIR(A, Wf)[0]
                        SIRlist.append(SIR)
                        self.SIR_list = SIRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth=3)
                            pl.xlabel(
                                "Number of Iterations / {}".format(
                                    debug_iteration_point
                                ),
                                fontsize=15,
                            )
                            pl.ylabel("SIR (dB)", fontsize=15)
                            pl.title("SIR Behaviour", fontsize=15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())
        self.W = W
        self.B = B

    def fit_batch_sparse(
        self,
        Y,
        n_epochs=1,
        neural_dynamic_iterations=250,
        neural_lr_start=0.9,
        neural_lr_stop=1e-3,
        shuffle=False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):

        lambda_, beta, muW, gamma_hat, W, B = (
            self.lambda_,
            self.beta,
            self.muW,
            self.gamma_hat,
            self.W,
            self.B,
        )
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        assert Y.shape[0] == self.y_dim, "You must input the transpose"

        samples = Y.shape[1]

        if debugging:
            SIRlist = []
            Sgt = self.Sgt
            A = self.A

        S = np.random.randn(self.s_dim, samples)  # Initial source estimates

        if shuffle:
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                y_current = Y[:, idx[i_sample]]
                s = np.zeros(self.s_dim)

                s = self.run_neural_dynamics_sparse(
                    y_current,
                    s,
                    W,
                    B,
                    beta,
                    gamma_hat,
                    lr_start=neural_lr_start,
                    lr_stop=neural_lr_stop,
                    neural_dynamic_iterations=neural_dynamic_iterations,
                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                )

                e = W @ y_current - s

                W = W - muW * beta * np.outer(e, y_current)

                z = B @ s
                B = (1 / lambda_) * (B - gamma_hat * np.outer(z, z))
                # Record the seperated signal
                S[:, idx[i_sample]] = s

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.B = B
                        Wf = self.compute_overall_mapping(return_mapping=True)
                        SIR = self.CalculateSIR(A, Wf)[0]
                        SIRlist.append(SIR)
                        self.SIR_list = SIRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth=3)
                            pl.xlabel(
                                "Number of Iterations / {}".format(
                                    debug_iteration_point
                                ),
                                fontsize=15,
                            )
                            pl.ylabel("SIR (dB)", fontsize=15)
                            pl.title("SIR Behaviour", fontsize=15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())
        self.W = W
        self.B = B

    def fit_batch_nnsparse(
        self,
        Y,
        n_epochs=1,
        neural_dynamic_iterations=250,
        neural_lr_start=0.9,
        neural_lr_stop=1e-3,
        shuffle=False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):

        lambda_, beta, muW, gamma_hat, W, B = (
            self.lambda_,
            self.beta,
            self.muW,
            self.gamma_hat,
            self.W,
            self.B,
        )
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        assert Y.shape[0] == self.y_dim, "You must input the transpose"

        samples = Y.shape[1]

        if debugging:
            SIRlist = []
            Sgt = self.Sgt
            A = self.A

        S = np.random.randn(self.s_dim, samples)  # Initial source estimates

        if shuffle:
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                y_current = Y[:, idx[i_sample]]
                s = np.zeros(self.s_dim)

                s = self.run_neural_dynamics_nnsparse(
                    y_current,
                    s,
                    W,
                    B,
                    beta,
                    gamma_hat,
                    lr_start=neural_lr_start,
                    lr_stop=neural_lr_stop,
                    neural_dynamic_iterations=neural_dynamic_iterations,
                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                )

                e = W @ y_current - s

                W = W - muW * beta * np.outer(e, y_current)

                z = B @ s
                B = (1 / lambda_) * (B - gamma_hat * np.outer(z, z))
                # Record the seperated signal
                S[:, idx[i_sample]] = s

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.B = B
                        Wf = self.compute_overall_mapping(return_mapping=True)
                        SIR = self.CalculateSIR(A, Wf)[0]
                        SIRlist.append(SIR)
                        self.SIR_list = SIRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth=3)
                            pl.xlabel(
                                "Number of Iterations / {}".format(
                                    debug_iteration_point
                                ),
                                fontsize=15,
                            )
                            pl.ylabel("SIR (dB)", fontsize=15)
                            pl.title("SIR Behaviour", fontsize=15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())
        self.W = W
        self.B = B

    def fit_batch_simplex(
        self,
        Y,
        n_epochs=1,
        neural_dynamic_iterations=250,
        neural_lr_start=0.9,
        neural_lr_stop=1e-3,
        shuffle=False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):

        lambda_, beta, muW, gamma_hat, W, B = (
            self.lambda_,
            self.beta,
            self.muW,
            self.gamma_hat,
            self.W,
            self.B,
        )
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        assert Y.shape[0] == self.y_dim, "You must input the transpose"

        samples = Y.shape[1]

        if debugging:
            SIRlist = []
            Sgt = self.Sgt
            A = self.A

        S = np.random.randn(self.s_dim, samples)  # Initial source estimates

        if shuffle:
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                y_current = Y[:, idx[i_sample]]
                s = np.zeros(self.s_dim)

                s = self.run_neural_dynamics_simplex(
                    y_current,
                    s,
                    W,
                    B,
                    beta,
                    gamma_hat,
                    lr_start=neural_lr_start,
                    lr_stop=neural_lr_stop,
                    neural_dynamic_iterations=neural_dynamic_iterations,
                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                )

                e = W @ y_current - s

                W = W - muW * beta * np.outer(e, y_current)

                z = B @ s
                B = (1 / lambda_) * (B - gamma_hat * np.outer(z, z))
                # Record the seperated signal
                S[:, idx[i_sample]] = s

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.B = B
                        Wf = self.compute_overall_mapping(return_mapping=True)
                        SIR = self.CalculateSIR(A, Wf)[0]
                        SIRlist.append(SIR)
                        self.SIR_list = SIRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth=3)
                            pl.xlabel(
                                "Number of Iterations / {}".format(
                                    debug_iteration_point
                                ),
                                fontsize=15,
                            )
                            pl.ylabel("SIR (dB)", fontsize=15)
                            pl.title("SIR Behaviour", fontsize=15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())
        self.W = W
        self.B = B

    def fit_batch_olshaussen(
        self,
        Y,
        n_epochs=1,
        neural_dynamic_iterations=250,
        neural_lr_start=0.9,
        neural_lr_stop=1e-3,
        shuffle=False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):

        lambda_, beta, muW, gamma_hat, W, B = (
            self.lambda_,
            self.beta,
            self.muW,
            self.gamma_hat,
            self.W,
            self.B,
        )
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        assert Y.shape[0] == self.y_dim, "You must input the transpose"

        samples = Y.shape[1]

        S = np.random.randn(self.s_dim, samples)  # Initial source estimates

        if shuffle:
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                y_current = Y[:, idx[i_sample]]
                s = np.zeros(self.s_dim)

                s = self.run_neural_dynamics_sparse(
                    y_current,
                    s,
                    W,
                    B,
                    beta,
                    gamma_hat,
                    lr_start=neural_lr_start,
                    lr_stop=neural_lr_stop,
                    neural_dynamic_iterations=neural_dynamic_iterations,
                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                )

                e = W @ y_current - s

                W = W - muW * beta * np.outer(e, y_current)

                z = B @ s
                B = (1 / lambda_) * (B - gamma_hat * np.outer(z, z))
                # Record the seperated signal
                S[:, idx[i_sample]] = s

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.B = B
                        Wf = self.compute_overall_mapping(return_mapping=True)
                        if plot_in_jupyter:
                            pl.clf()
                            n_iterations = k * samples + i_sample
                            fig, ax = pl.subplots(12, 12, figsize=(20, 20))
                            for l in range(144):
                                rf = -np.reshape(Wf[l, :], (12, 12))
                                # rf = ZeroOneNormalizeData(rf)
                                ax[l // 12, l % 12].imshow(rf, cmap="gray")
                                ax[l // 12, l % 12].axes.xaxis.set_visible(False)
                                ax[l // 12, l % 12].axes.yaxis.set_visible(False)
                            pl.subplots_adjust(
                                right=0.97,
                                left=0.03,
                                bottom=0.03,
                                top=0.97,
                                wspace=0.1,
                                hspace=0.1,
                            )
                            pl.suptitle(
                                f"The receptive fields after {n_iterations}",
                                fontsize=25,
                            )
                            clear_output(wait=True)
                            display(pl.gcf())
        self.W = W
        self.B = B


def whiten_signal(X, mean_normalize=True, type_=3):
    """
    Input : X  ---> Input signal to be whitened

    type_ : Defines the type for preprocesing matrix. type_ = 1 and 2 uses eigenvalue decomposition whereas type_ = 3 uses SVD.

    Output: X_white  ---> Whitened signal, i.e., X_white = W_pre @ X where W_pre = (R_x^0.5)^+ (square root of sample correlation matrix)
    """
    if mean_normalize:
        X = X - np.mean(X, axis=0, keepdims=True)

    cov = np.cov(X.T)

    if type_ == 3:  # Whitening using singular value decomposition
        U, S, V = np.linalg.svd(cov)
        d = np.diag(1.0 / np.sqrt(S))
        W_pre = np.dot(U, np.dot(d, U.T))

    else:  # Whitening using eigenvalue decomposition
        d, S = np.linalg.eigh(cov)
        D = np.diag(d)

        D_sqrt = np.sqrt(D * (D > 0))

        if type_ == 1:  # Type defines how you want W_pre matrix to be
            W_pre = np.linalg.pinv(S @ D_sqrt)
        elif type_ == 2:
            W_pre = np.linalg.pinv(S @ D_sqrt @ S.T)

    X_white = (W_pre @ X.T).T

    return X_white, W_pre


def whiten_input(X, n_components=None, return_prewhitening_matrix=False):
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
    mX = np.mean(X, axis=1).reshape((x_dim, 1))
    # Covariance of Mixtures
    Rxx = np.dot(X, X.T) / N - np.dot(mX, mX.T)
    # Eigenvalue Decomposition
    d, V = np.linalg.eig(Rxx)
    D = np.diag(d)
    # Sorting indexis for eigenvalues from large to small
    ie = np.argsort(-d)
    # Inverse square root of eigenvalues
    ddinv = 1 / np.sqrt(d[ie[:s_dim]])
    # Pre-whitening matrix
    Wpre = np.dot(np.diag(ddinv), V[:, ie[:s_dim]].T)  # *np.sqrt(12)
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
        X_normalized[:, i] = ZeroOneNormalizeData(X[:, i])

    return X_normalized


def ProjectOntoLInfty(X, thresh=1.0):
    return (
        X * (X >= -thresh) * (X <= thresh)
        + (X > thresh) * thresh
        - thresh * (X < -thresh)
    )


def ProjectOntoNNLInfty(X, thresh=1.0):
    return X * (X >= 0) * (X <= thresh) + (X > thresh) * thresh


def Subplot_gray_images(I, image_shape=[512, 512], height=15, width=15, title=""):
    n_images = I.shape[1]
    fig, ax = plt.subplots(1, n_images)
    fig.suptitle(title)
    fig.set_figheight(height)
    fig.set_figwidth(width)
    for i in range(n_images):
        ax[i].imshow(I[:, i].reshape(image_shape[0], image_shape[1]), cmap="gray")

    plt.show()


def subplot_1D_signals(
    X, title="", title_fontsize=20, figsize=(10, 5), linewidth=1, colorcode="#050C12"
):
    """Plot the 1D signals (each column from the given matrix)"""
    n = X.shape[1]  # Number of signals

    fig, ax = plt.subplots(n, 1, figsize=figsize)

    for i in range(n):
        ax[i].plot(X[:, i], linewidth=linewidth, color=colorcode)
        ax[i].grid()

    plt.suptitle(title, fontsize=title_fontsize)
    # plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
    plt.draw()


def plot_convergence_plot(
    metric,
    xlabel="",
    ylabel="",
    title="",
    figsize=(12, 8),
    fontsize=15,
    linewidth=3,
    colorcode="#050C12",
):

    plt.figure(figsize=figsize)
    plt.plot(metric, linewidth=linewidth, color=colorcode)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    # plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
    plt.grid()
    plt.draw()


def find_permutation_between_source_and_estimation(S, Y):
    """
    S    : Original source matrix
    Y    : Matrix of estimations of sources (after BSS or ICA algorithm)

    return the permutation of the source seperation algorithm
    """

    # perm = np.argmax(np.abs(np.corrcoef(S.T,Y.T) - np.eye(2*S.shape[1])),axis = 0)[S.shape[1]:]
    perm = np.argmax(np.abs(np.corrcoef(Y.T, S.T) - np.eye(2 * S.shape[1])), axis=0)[
        S.shape[1] :
    ]
    return perm


def signed_and_permutation_corrected_sources(S, Y):
    perm = find_permutation_between_source_and_estimation(S, Y)
    return np.sign((Y[:, perm] * S).sum(axis=0)) * Y[:, perm]


def psnr(img1, img2, pixel_max=1):
    """Return peak-signal-to-noise-ratio between given two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    else:
        return 20 * np.log10(pixel_max / np.sqrt(mse))


def snr(S_original, S_noisy):
    N_hat = S_original - S_noisy
    N_P = (N_hat**2).sum(axis=0)
    S_P = (S_original**2).sum(axis=0)
    snr = 10 * np.log10(S_P / N_P)
    return snr


def ProjectRowstoL1NormBall(H):
    Hshape = H.shape
    # lr=np.ones((Hshape[0],1))@np.reshape((1/np.linspace(1,Hshape[1],Hshape[1])),(1,Hshape[1]))
    lr = np.tile(
        np.reshape((1 / np.linspace(1, Hshape[1], Hshape[1])), (1, Hshape[1])),
        (Hshape[0], 1),
    )
    # Hnorm1=np.reshape(np.sum(np.abs(self.H),axis=1),(Hshape[0],1))

    u = -np.sort(-np.abs(H), axis=1)
    sv = np.cumsum(u, axis=1)
    q = np.where(
        u > ((sv - 1) * lr),
        np.tile(
            np.reshape((np.linspace(1, Hshape[1], Hshape[1]) - 1), (1, Hshape[1])),
            (Hshape[0], 1),
        ),
        np.zeros((Hshape[0], Hshape[1])),
    )
    rho = np.max(q, axis=1)
    rho = rho.astype(int)
    lindex = np.linspace(1, Hshape[0], Hshape[0]) - 1
    lindex = lindex.astype(int)
    theta = np.maximum(
        0, np.reshape((sv[tuple([lindex, rho])] - 1) / (rho + 1), (Hshape[0], 1))
    )
    ww = np.abs(H) - theta
    H = np.sign(H) * (ww > 0) * ww
    return H


def display_matrix(array):
    data = ""
    for line in array:
        if len(line) == 1:
            data += " %.3f &" % line + r" \\\n"
            continue
        for element in line:
            data += " %.3f &" % element
        data += r" \\" + "\n"
    display(Math("\\begin{bmatrix} \n%s\\end{bmatrix}" % data))


# Calculate SIR Function
def CalculateSIR(H, pH, return_db=True):
    G = pH @ H
    Gmax = np.diag(np.max(abs(G), axis=1))
    P = 1.0 * ((np.linalg.inv(Gmax) @ np.abs(G)) > 0.99)
    T = G @ P.T
    rankP = np.linalg.matrix_rank(P)
    diagT = np.diag(T)
    # Signal Power
    sigpow = np.linalg.norm(diagT, 2) ** 2
    # Interference Power
    intpow = np.linalg.norm(T, "fro") ** 2 - sigpow
    SIRV = sigpow / intpow
    # SIRV=np.linalg.norm((np.diag(T)))**2/(np.linalg.norm(T,'fro')**2-np.linalg.norm(np.diag(T))**2)
    if return_db:
        SIRV = 10 * np.log10(sigpow / intpow)

    return SIRV, rankP


@njit(fastmath=True)
def accumu(lis):
    """Cumulative Sum.

    Same as np.cumsum()
    """
    result = np.zeros_like(lis)
    for i in range(lis.shape[1]):
        result[:, i] = np.sum(lis[:, : i + 1])

    return result


@njit(fastmath=True)
def merge_sort(list_):
    """Sorts a list in ascending order. Returns a new sorted list.

    Divide : Find the midpoint of the list and divide into sublist
    Conquer : Recursively sort the sublists created in previous step
    Combine : Merge the sorted sublists created in previous step

    Takes O(n log n) time.
    """

    def merge(left, right):
        """Merges two lists (arrays), sorting them in the process. Returns a new merged
        list.

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
        """Divide the unsorted list at midpoint into sublists.

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


def generate_correlated_uniform_sources(
    R, range_=[-1, 1], n_sources=5, size_sources=500000
):
    """
    R : correlation matrix
    """
    assert R.shape[0] == n_sources, (
        "The shape of correlation matrix must be equal to the number of sources, which is entered as (%d)"
        % (n_sources)
    )
    S = np.random.uniform(range_[0], range_[1], size=(n_sources, size_sources))
    L = np.linalg.cholesky(R)
    S_ = L @ S
    return S_


def generate_correlated_copula_sources(
    rho=0.0, df=4, n_sources=5, size_sources=500000, decreasing_correlation=True
):
    """
    rho     : correlation parameter
    df      : degrees for freedom

    required libraries:
    from scipy.stats import invgamma, chi2, t
    from scipy import linalg
    import numpy as np
    """
    if decreasing_correlation:
        first_row = np.array([rho**j for j in range(n_sources)])
        calib_correl_matrix = linalg.toeplitz(first_row, first_row)
    else:
        calib_correl_matrix = (
            np.eye(n_sources) * (1 - rho) + np.ones((n_sources, n_sources)) * rho
        )

    mu = np.zeros(len(calib_correl_matrix))
    s = chi2.rvs(df, size=size_sources)[:, np.newaxis]
    Z = np.random.multivariate_normal(mu, calib_correl_matrix, size_sources)
    X = np.sqrt(df / s) * Z  # chi-square method
    S = t.cdf(X, df).T
    return S


def generate_uniform_points_in_polytope(polytope_vertices, size):
    """ "
    polytope_vertices : vertex matrix of shape (n_dim, n_vertices)

    return:
        Samples of shape (n_dim, size)
    """
    polytope_vertices = polytope_vertices.T
    dims = polytope_vertices.shape[-1]
    hull = polytope_vertices[ConvexHull(polytope_vertices).vertices]
    deln = hull[Delaunay(hull).simplices]

    vols = np.abs(det(deln[:, :dims, :] - deln[:, dims:, :])) / np.math.factorial(dims)
    sample = np.random.choice(len(vols), size=size, p=vols / vols.sum())

    return np.einsum(
        "ijk, ij -> ik", deln[sample], dirichlet.rvs([1] * (dims + 1), size=size)
    ).T


def generate_practical_polytope(
    dim, antisparse_dims, nonnegative_dims, relative_sparse_dims_list
):
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
        pm_one = [[1, -1] for _ in range(relative_sparse_dims.shape[0])]
        for i in itertools.product(*pm_one):
            row_copy = row.copy()
            row_copy[relative_sparse_dims] = i
            A.append(list(row_copy))
            b.append(1)
    A = np.array(A)
    b = np.array(b)
    vertices = pypoman.compute_polytope_vertices(A, b)
    V = np.array([list(v) for v in vertices]).T
    return (A, b), V
