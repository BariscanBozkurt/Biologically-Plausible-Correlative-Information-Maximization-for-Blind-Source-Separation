import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from IPython.display import Latex, Math, clear_output, display
from numba import njit
from tqdm import tqdm

from bss_utils import *
##### IMPORT MY UTILITY SCRIPTS #######
from BSSbase import *
from dsp_utils import *
# from general_utils import *
from numba_utils import *

# from visualization_utils import *

mpl.rcParams["xtick.labelsize"] = 15
mpl.rcParams["ytick.labelsize"] = 15


class OnlineBCA(BSSBaseClass):
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
    """

    def __init__(
        self,
        s_dim,
        x_dim,
        lambda_=0.999,
        mu_F=0.03,
        beta=5,
        F=None,
        B=None,
        neural_OUTPUT_COMP_TOL=1e-6,
        set_ground_truth=False,
        S=None,
        A=None,
    ):
        if F is not None:
            assert F.shape == (
                s_dim,
                x_dim,
            ), "The shape of the initial guess F must be (s_dim, x_dim) = (%d,%d)" % (
                s_dim,
                x_dim,
            )
            F = F
        else:
            F = np.random.randn(s_dim, x_dim)
            F = F / np.sqrt(np.sum(np.abs(F) ** 2, axis=1)).reshape(s_dim, 1)
            F = np.eye(s_dim, x_dim)

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
        self.x_dim = x_dim
        self.lambda_ = lambda_
        self.beta = beta
        self.mu_F = mu_F
        self.gamma_hat = (1 - lambda_) / lambda_
        self.F = F
        self.B = B
        self.neural_OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL
        self.set_ground_truth = set_ground_truth
        self.S = S
        self.A = A
        self.SIR_list = []
        self.SNR_list = []

    ############################################################################################
    ############### REQUIRED FUNCTIONS FOR DEBUGGING ###########################################
    ############################################################################################
    def evaluate_for_debug(self, W, A, S, X, mean_normalize_estimation=False):
        s_dim = self.s_dim
        Y_ = W @ X
        if mean_normalize_estimation:
            Y_ = Y_ - Y_.mean(axis=1).reshape(-1, 1)
        Y_ = self.signed_and_permutation_corrected_sources(S, Y_)
        coef_ = ((Y_ * S).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1)
        Y_ = coef_ * Y_

        SINR = 10 * np.log10(self.CalculateSINRjit(Y_, S, False)[0])
        SNR = self.snr_jit(S, Y_)

        T = W @ A
        Tabs = np.abs(T)
        P = np.zeros((s_dim, s_dim))

        for SourceIndex in range(s_dim):
            Tmax = np.max(Tabs[SourceIndex, :])
            Tabs[SourceIndex, :] = Tabs[SourceIndex, :] / Tmax
            P[SourceIndex, :] = Tabs[SourceIndex, :] > 0.999

        GG = P.T @ T
        _, SGG, _ = np.linalg.svd(
            GG
        )  # SGG is the singular values of overall matrix Wf @ A

        return SINR, SNR, SGG, Y_, P

    def plot_for_debug(self, SIR_list, SNR_list, P, debug_iteration_point, YforPlot):
        pl.clf()
        pl.subplot(2, 2, 1)
        pl.plot(np.array(SIR_list), linewidth=5)
        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize=45)
        pl.ylabel("SIR (dB)", fontsize=45)
        pl.title("SIR Behaviour", fontsize=45)
        pl.grid()
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        pl.subplot(2, 2, 2)
        pl.plot(np.array(SNR_list), linewidth=5)
        pl.grid()
        pl.title("Component SNR Check", fontsize=45)
        pl.ylabel("SNR (dB)", fontsize=45)
        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize=45)
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        pl.subplot(2, 2, 3)
        pl.plot(np.array(self.SV_list), linewidth=5)
        pl.grid()
        pl.title(
            "Singular Value Check, Overall Matrix Rank: "
            + str(np.linalg.matrix_rank(P)),
            fontsize=45,
        )
        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize=45)
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        pl.subplot(2, 2, 4)
        pl.plot(YforPlot, linewidth=5)
        pl.title("Y last 25", fontsize=45)
        pl.grid()
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        clear_output(wait=True)
        display(pl.gcf())

    def compute_overall_mapping(self, return_mapping=False):
        F, B, gamma_hat, beta = self.F, self.B, self.gamma_hat, self.beta
        # W = np.linalg.pinv((gamma_hat/beta) * B - np.eye(self.s_dim)) @ F
        W = np.linalg.solve((gamma_hat / beta) * B - np.eye(self.s_dim), F)
        if return_mapping:
            return W
        else:
            return None

    def predict(self, X):
        W = self.compute_overall_mapping(return_mapping=True)
        return W @ X

    ###############################################################
    ############### NEURAL DYNAMICS ALGORITHMS ####################
    ###############################################################
    @staticmethod
    @njit
    def run_neural_dynamics_antisparse(
        x,
        y,
        F,
        B,
        beta,
        gamma_hat,
        mu_y_start=0.9,
        neural_dynamic_iterations=100,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):

        yke = np.dot(F, x)
        for j in range(neural_dynamic_iterations):
            mu_y = mu_y_start / (j + 1)
            y_old = y.copy()
            e = yke - y
            y = y + mu_y * (gamma_hat * B @ y + beta * e)
            y = np.clip(y, -1, 1)

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    @staticmethod
    @njit
    def run_neural_dynamics_nnantisparse(
        x,
        y,
        F,
        B,
        beta,
        gamma_hat,
        mu_y_start=0.9,
        neural_dynamic_iterations=100,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):

        yke = np.dot(F, x)
        for j in range(neural_dynamic_iterations):
            mu_y = mu_y_start / (j + 1)
            y_old = y.copy()
            e = yke - y
            y = y + mu_y * (gamma_hat * B @ y + beta * e)
            y = np.clip(y, 0, 1)

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    def fit_batch_antisparse(
        self,
        X,
        n_epochs=2,
        neural_dynamic_iterations=250,
        neural_lr_start=0.9,
        shuffle=False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):

        lambda_, beta, mu_F, gamma_hat, F, B = (
            self.lambda_,
            self.beta,
            self.mu_F,
            self.gamma_hat,
            self.F,
            self.B,
        )
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
        SIR_list = self.SIR_list
        SNR_list = self.SNR_list
        self.SV_list = []

        assert X.shape[0] == self.x_dim, "You must input the transpose"

        samples = X.shape[1]

        if debugging:
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize=(45, 30), dpi=80)

        # Y = np.zeros((self.s_dim, samples))
        Y = np.random.randn(self.s_dim, samples) * 0.05

        if shuffle:
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:, idx[i_sample]]
                y = np.zeros(self.s_dim)

                y = self.run_neural_dynamics_antisparse(
                    x_current,
                    y,
                    F,
                    B,
                    beta,
                    gamma_hat,
                    mu_y_start=neural_lr_start,
                    neural_dynamic_iterations=neural_dynamic_iterations,
                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                )

                e = F @ x_current - y

                F = F - mu_F * beta * np.outer(e, x_current)

                z = B @ y
                B = (1 / lambda_) * (B - gamma_hat * np.outer(z, z))

                # Record the seperated signal
                Y[:, idx[i_sample]] = y

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.F = F
                        self.B = B
                        try:
                            Wf = self.compute_overall_mapping(return_mapping=True)

                            SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(
                                Wf, A, S, X, False
                            )
                            self.SV_list.append(abs(SGG))

                            SIR_list.append(SINR)
                            SNR_list.append(SNR)

                            self.SNR_list = SNR_list
                            self.SIR_list = SIR_list

                            if plot_in_jupyter:
                                YforPlot = Y[:, idx[i_sample - 25 : i_sample]].T
                                self.plot_for_debug(
                                    SIR_list,
                                    SNR_list,
                                    P,
                                    debug_iteration_point,
                                    YforPlot,
                                )
                        except Exception as e:
                            print(str(e))
        self.F = F
        self.B = B

    def fit_batch_nnantisparse(
        self,
        X,
        n_epochs=2,
        neural_dynamic_iterations=250,
        neural_lr_start=0.9,
        shuffle=False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):

        lambda_, beta, mu_F, gamma_hat, F, B = (
            self.lambda_,
            self.beta,
            self.mu_F,
            self.gamma_hat,
            self.F,
            self.B,
        )
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
        SIR_list = self.SIR_list
        SNR_list = self.SNR_list
        self.SV_list = []

        assert X.shape[0] == self.x_dim, "You must input the transpose"

        samples = X.shape[1]

        if debugging:
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize=(45, 30), dpi=80)

        # Y = np.zeros((self.s_dim, samples))
        Y = np.abs(np.random.randn(self.s_dim, samples) * 0.05)

        if shuffle:
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:, idx[i_sample]]
                y = np.zeros(self.s_dim)

                y = self.run_neural_dynamics_nnantisparse(
                    x_current,
                    y,
                    F,
                    B,
                    beta,
                    gamma_hat,
                    mu_y_start=neural_lr_start,
                    neural_dynamic_iterations=neural_dynamic_iterations,
                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                )

                e = F @ x_current - y

                F = F - mu_F * beta * np.outer(e, x_current)

                z = B @ y
                B = (1 / lambda_) * (B - gamma_hat * np.outer(z, z))
                # Record the seperated signal
                Y[:, idx[i_sample]] = y

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.F = F
                        self.B = B
                        try:
                            Wf = self.compute_overall_mapping(return_mapping=True)

                            SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(
                                Wf, A, S, X, False
                            )
                            self.SV_list.append(abs(SGG))

                            SIR_list.append(SINR)
                            SNR_list.append(SNR)

                            self.SNR_list = SNR_list
                            self.SIR_list = SIR_list

                            if plot_in_jupyter:
                                YforPlot = Y[:, idx[i_sample - 25 : i_sample]].T
                                self.plot_for_debug(
                                    SIR_list,
                                    SNR_list,
                                    P,
                                    debug_iteration_point,
                                    YforPlot,
                                )
                        except Exception as e:
                            print(str(e))
        self.F = F
        self.B = B
