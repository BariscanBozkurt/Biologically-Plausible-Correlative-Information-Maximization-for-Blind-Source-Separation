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
from polytope_utils import ProjectOntoPolytopeCanonical
# from visualization_utils import *

mpl.rcParams["xtick.labelsize"] = 15
mpl.rcParams["ytick.labelsize"] = 15

############# Log-Det Mutual Information Based Blind Source Separation Neural Network ####################################
class OnlineCorInfoMax(BSSBaseClass):

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

    """

    def __init__(
        self,
        s_dim,
        x_dim,
        lambday=0.999,
        lambdae=1,
        muW=1e-3,
        epsilon=1e-3,
        W=None,
        By=None,
        Be=None,
        neural_OUTPUT_COMP_TOL=1e-6,
        set_ground_truth=False,
        S=None,
        A=None,
    ):
        if W is not None:
            assert W.shape == (
                s_dim,
                x_dim,
            ), "The shape of the initial guess W must be (s_dim, x_dim) = (%d,%d)" % (
                s_dim,
                x_dim,
            )
            W = W
        else:
            W = np.eye(s_dim, x_dim)

        if By is not None:
            assert By.shape == (
                s_dim,
                s_dim,
            ), "The shape of the initial guess By must be (s_dim, s_dim) = (%d,%d)" % (
                s_dim,
                s_dim,
            )
            By = By
        else:
            By = 5 * np.eye(s_dim)

        if Be is not None:
            assert Be.shape == (
                s_dim,
                s_dim,
            ), "The shape of the initial guess Be must be (s_dim, s_dim) = (%d,%d)" % (
                s_dim,
                s_dim,
            )
            Be = Be
        else:
            Be = 1 * np.eye(s_dim)

        self.s_dim = s_dim
        self.x_dim = x_dim
        self.lambday = lambday
        self.lambdae = lambdae
        self.muW = muW
        self.gamy = (1 - lambday) / lambday
        self.game = (1 - lambdae) / lambdae
        self.epsilon = epsilon
        self.W = W
        self.By = By
        self.Be = Be
        self.neural_OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL
        ### Ground Truth Sources and Mixing Matrix For Debugging
        self.set_ground_truth = set_ground_truth
        self.S = S  # Sources
        self.A = A  # Mixing Matrix
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
        W, By, Be, gamy, game = self.W, self.By, self.Be, self.gamy, self.game
        # Wf = np.linalg.pinv(gamy * By - game * Be - beta * np.eye(self.s_dim)) @ (game * Be @ W + beta * W)
        # Wf = np.linalg.pinv(gamy * By - game * Be) @ (game * Be @ W)
        Wf = np.linalg.solve(gamy * By - game * Be, game * Be @ W)
        if return_mapping:
            return Wf
        else:
            return None

    ###############################################################
    ############### NEURAL DYNAMICS ALGORITHMS ####################
    ###############################################################
    @staticmethod
    @njit
    def run_neural_dynamics_antisparse(
        x,
        y,
        W,
        By,
        Be,
        gamy,
        game,
        lr_start=0.9,
        lr_stop=1e-15,
        lr_rule="divide_by_loop_index",
        lr_decay_multiplier=0.01,
        neural_dynamic_iterations=100,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):
        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            # mu_y = max(lr_start / (j + 1), lr_stop)
            if lr_rule == "constant":
                mu_y = lr_start
            elif lr_rule == "divide_by_loop_index":
                mu_y = max(lr_start / (j + 1), lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                mu_y = max(lr_start / (j * lr_decay_multiplier + 1), lr_stop)
            y_old = y.copy()
            e = yke - y
            grady = gamy * By @ y + game * Be @ e
            y = y + mu_y * (grady)
            y = np.clip(y, -1, 1)

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    @staticmethod
    @njit
    def run_neural_dynamics_nnantisparse(
        x,
        y,
        W,
        By,
        Be,
        gamy,
        game,
        lr_start=0.9,
        lr_stop=1e-15,
        lr_rule="divide_by_loop_index",
        lr_decay_multiplier=0.01,
        neural_dynamic_iterations=100,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):

        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            # mu_y = max(lr_start / (j + 1), lr_stop)
            if lr_rule == "constant":
                mu_y = lr_start
            elif lr_rule == "divide_by_loop_index":
                mu_y = max(lr_start / (j + 1), lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                mu_y = max(lr_start / (j * lr_decay_multiplier + 1), lr_stop)
            y_old = y.copy()
            e = yke - y
            grady = gamy * By @ y + game * Be @ e
            y = y + mu_y * (grady)
            y = np.clip(y, 0, 1)

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    @staticmethod
    @njit
    def run_neural_dynamics_mixedantisparse(
        x,
        y,
        nn_components,
        signed_components,
        W,
        By,
        Be,
        gamy,
        game,
        lr_start=0.9,
        lr_stop=1e-15,
        lr_rule="divide_by_loop_index",
        lr_decay_multiplier=0.01,
        neural_dynamic_iterations=100,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):

        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            # mu_y = max(lr_start / (j + 1), lr_stop)
            if lr_rule == "constant":
                mu_y = lr_start
            elif lr_rule == "divide_by_loop_index":
                mu_y = max(lr_start / (j + 1), lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                mu_y = max(lr_start / (j * lr_decay_multiplier + 1), lr_stop)
            y_old = y.copy()
            e = yke - y
            grady = gamy * By @ y + game * Be @ e
            y = y + mu_y * (grady)
            y[signed_components] = np.clip(y[signed_components], -1, 1)
            y[nn_components] = np.clip(y[nn_components], 0, 1)

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    @staticmethod
    @njit
    def run_neural_dynamics_sparse(
        x,
        y,
        W,
        By,
        Be,
        gamy,
        game,
        lr_start=0.9,
        lr_stop=1e-15,
        lr_rule="divide_by_loop_index",
        lr_decay_multiplier=0.01,
        stlambd_lr=1,
        neural_dynamic_iterations=100,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):

        STLAMBD = 0
        dval = 0
        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            # mu_y = max(lr_start / (j + 1), lr_stop)
            if lr_rule == "constant":
                mu_y = lr_start
            elif lr_rule == "divide_by_loop_index":
                mu_y = max(lr_start / (j + 1), lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                mu_y = max(lr_start / (j * lr_decay_multiplier + 1), lr_stop)
            y_old = y.copy()
            e = yke - y
            grady = gamy * By @ y + game * Be @ e
            y = y + mu_y * (grady)

            # SOFT THRESHOLDING
            y_absolute = np.abs(y)
            y_sign = np.sign(y)

            y = (y_absolute > STLAMBD) * (y_absolute - STLAMBD) * y_sign
            dval = np.linalg.norm(y, 1) - 1
            STLAMBD = max(STLAMBD + stlambd_lr * dval, 0)

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    @staticmethod
    @njit
    def run_neural_dynamics_nnsparse(
        x,
        y,
        W,
        By,
        Be,
        gamy,
        game,
        lr_start=0.9,
        lr_stop=1e-15,
        lr_rule="divide_by_loop_index",
        lr_decay_multiplier=0.01,
        stlambd_lr=1,
        neural_dynamic_iterations=100,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):

        STLAMBD = 0
        dval = 0
        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            # mu_y = max(lr_start / (j + 1), lr_stop)
            if lr_rule == "constant":
                mu_y = lr_start
            elif lr_rule == "divide_by_loop_index":
                mu_y = max(lr_start / (j + 1), lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                mu_y = max(lr_start / (j * lr_decay_multiplier + 1), lr_stop)

            y_old = y.copy()
            e = yke - y
            grady = gamy * By @ y + game * Be @ e
            y = y + mu_y * (grady)

            y = np.maximum(y - STLAMBD, 0)

            dval = np.sum(y) - 1
            STLAMBD = max(STLAMBD + stlambd_lr * dval, 0)

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    @staticmethod
    @njit
    def run_neural_dynamics_simplex(
        x,
        y,
        W,
        By,
        Be,
        gamy,
        game,
        lr_start=0.9,
        lr_stop=1e-15,
        lr_rule="divide_by_loop_index",
        lr_decay_multiplier=0.01,
        stlambd_lr=0.05,
        neural_dynamic_iterations=100,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):

        STLAMBD = 0
        dval = 0
        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            # mu_y = max(lr_start / (j + 1), lr_stop)
            if lr_rule == "constant":
                mu_y = lr_start
            elif lr_rule == "divide_by_loop_index":
                mu_y = max(lr_start / (j + 1), lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                mu_y = max(lr_start / (j * lr_decay_multiplier + 1), lr_stop)
                
            y_old = y.copy()
            e = yke - y
            grady = gamy * By @ y + game * Be @ e
            y = y + mu_y * (grady)

            y = np.maximum(y - STLAMBD, 0)

            dval = np.sum(y) - 1
            STLAMBD = STLAMBD + stlambd_lr * dval

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    @staticmethod
    @njit
    def run_neural_dynamics_nnwsubsparse(
        x,
        y,
        nn_components,
        sparse_components,
        W,
        My,
        Be,
        gamy,
        game,
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
            return (
                X * (X >= 0.0) * (X <= thresh) + (X > thresh) * thresh
            )  # -thresh*(X<-thresh)

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

            y[sparse_components] = (
                (y_sparse_absolute > STLAMBD)
                * (y_sparse_absolute - STLAMBD)
                * y_sparse_sign
            )
            y = ProjectOntoLInfty(y)
            dval = np.linalg.norm(y[sparse_components], 1) - 1
            STLAMBD = max(STLAMBD + 1 * dval, 0)

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    @staticmethod
    @njit
    def run_neural_dynamics_nnwsubnnsparse(
        x,
        y,
        nn_components,
        nnsparse_components,
        W,
        My,
        Be,
        gamy,
        game,
        lr_start=0.9,
        lr_stop=1e-15,
        neural_dynamic_iterations=100,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):
        def ProjectOntoNNLInfty(X, thresh=1.0):
            return X * (X >= 0.0) * (X <= thresh) + (X > thresh) * thresh

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
    def run_neural_dynamics_general_polytope(
        x,
        y,
        signed_dims,
        nn_dims,
        sparse_dims_list,
        W,
        By,
        Be,
        gamy,
        game,
        lr_start=0.9,
        lr_stop=1e-15,
        lr_rule="divide_by_loop_index",
        lr_decay_multiplier=0.01,
        stlambd_lr=1,
        neural_dynamic_iterations=100,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):
        def loop_intersection(lst1, lst2):
            result = []
            for element1 in lst1:
                for element2 in lst2:
                    if element1 == element2:
                        result.append(element1)
            return result

        def SoftThresholding(X, thresh):
            X_absolute = np.abs(X)
            X_sign = np.sign(X)
            X_thresholded = (X_absolute > thresh) * (X_absolute - thresh) * X_sign
            return X_thresholded

        STLAMBD_list = np.zeros(len(sparse_dims_list))
        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            # mu_y = max(lr_start / (j + 1), lr_stop)
            if lr_rule == "constant":
                mu_y = lr_start
            elif lr_rule == "divide_by_loop_index":
                mu_y = max(lr_start / (j + 1), lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                mu_y = max(lr_start / (j * lr_decay_multiplier + 1), lr_stop)
            y_old = y.copy()
            e = yke - y
            grady = gamy * By @ y + game * Be @ e
            y = y + mu_y * (grady)
            if sparse_dims_list[0][0] != -1:
                for ss, sparse_dim in enumerate(sparse_dims_list):
                    # y[sparse_dim] = SoftThresholding(y[sparse_dim], STLAMBD_list[ss])
                    # STLAMBD_list[ss] = max(STLAMBD_list[ss] + (np.linalg.norm(y[sparse_dim],1) - 1), 0)
                    if signed_dims[0] != -1:
                        y[
                            np.array(loop_intersection(sparse_dim, signed_dims))
                        ] = SoftThresholding(
                            y[np.array(loop_intersection(sparse_dim, signed_dims))],
                            STLAMBD_list[ss],
                        )
                    if nn_dims[0] != -1:
                        y[
                            np.array(loop_intersection(sparse_dim, nn_dims))
                        ] = np.maximum(
                            y[np.array(loop_intersection(sparse_dim, nn_dims))]
                            - STLAMBD_list[ss],
                            0,
                        )
                    STLAMBD_list[ss] = max(
                        STLAMBD_list[ss] + stlambd_lr * (np.linalg.norm(y[sparse_dim], 1) - 1), 0
                    )
            if signed_dims[0] != -1:
                y[signed_dims] = np.clip(y[signed_dims], -1, 1)
            if nn_dims[0] != -1:
                y[nn_dims] = np.clip(y[nn_dims], 0, 1)

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    ###############################################################
    ######FIT NEXT FUNCTIONS FOR ONLINE LEARNING SETTING ##########
    ###############################################################
    def fit_next_antisparse(
        self, x_current, neural_dynamic_iterations=250, lr_start=0.9
    ):

        lambday, lambdae, muW, gamy, game, W, By, Be = (
            self.lambday,
            self.lambdae,
            self.muW,
            self.gamy,
            self.game,
            self.W,
            self.By,
            self.Be,
        )
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        h = 1 / gamy  # Hopefield parameter

        y = np.zeros(self.s_dim)

        # Output recurrent weights
        My = By + h * np.eye(self.s_dim)

        y = self.run_neural_dynamics_antisparse(
            x_current,
            y,
            W,
            My,
            Be,
            gamy,
            game,
            lr_start=lr_start,
            neural_dynamic_iterations=neural_dynamic_iterations,
            neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
        )

        e = y - W @ x_current

        W = W + muW * np.outer(e, x_current)

        By = (1 / lambday) * (By - gamy * np.outer(By @ y, By @ y))

        ee = np.dot(Be, e)
        Be = 1 / lambdae * (Be - game * np.outer(ee, ee))

        self.W = W
        self.By = By
        self.Be = Be

    ####################################################################
    ## FIT BATCH FUNCTIONS IF ALL THE OBSERVATIONS ARE AVAILABLE      ##
    ## THESE FUNCTIONS ALSO FIT IN ONLINE MANNER. YOU CAN CONSIDER    ##
    ## THEM AS EXTENSIONS OF FIT NEXT FUNCTIONS ABOVE (FOR DEBUGGING) ##
    ####################################################################
    def fit_batch_antisparse(
        self,
        X,
        n_epochs=1,
        neural_dynamic_iterations=250,
        neural_lr_start=0.9,
        neural_lr_stop=1e-15,
        synaptic_lr_rule="constant",
        synaptic_lr_decay_divider=5000,
        neural_loop_lr_rule="divide_by_loop_index",
        neural_lr_decay_multiplier=0.005,
        use_error_corr_structured_connectivity=False,
        shuffle=False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):

        lambday, lambdae, muW, gamy, game, W, By, Be = (
            self.lambday,
            self.lambdae,
            self.muW,
            self.gamy,
            self.game,
            self.W,
            self.By,
            self.Be,
        )
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
        SIR_list = self.SIR_list
        SNR_list = self.SNR_list
        self.SV_list = []

        h = 1 / gamy  # Hopefield parameter

        assert X.shape[0] == self.x_dim, "You must input the transpose"

        samples = X.shape[1]

        if debugging:
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize=(45, 30), dpi=80)

        Y = np.random.randn(self.s_dim, samples)

        if shuffle:
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:, idx[i_sample]]
                y = np.zeros(self.s_dim)

                # Output recurrent weights
                y = self.run_neural_dynamics_antisparse(
                    x_current,
                    y,
                    W,
                    By,
                    Be,
                    gamy,
                    game,
                    lr_start=neural_lr_start,
                    lr_stop=neural_lr_stop,
                    lr_rule=neural_loop_lr_rule,
                    lr_decay_multiplier=neural_lr_decay_multiplier,
                    neural_dynamic_iterations=neural_dynamic_iterations,
                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                )

                e = y - W @ x_current

                if synaptic_lr_rule == "constant":
                    muW_ = muW
                elif synaptic_lr_rule == "divide_by_log_index":
                    muW_ = np.max(
                        [
                            muW
                            / (1 + np.log(2 + (i_sample // synaptic_lr_decay_divider))),
                            1e-3,
                        ]
                    )
                elif synaptic_lr_rule == "divide_by_index":
                    muW_ = np.max([muW / (i_sample // synaptic_lr_decay_divider), 1e-3])

                W = W + muW_ * np.outer(e, x_current)

                z = By @ y
                By = (1 / lambday) * (By - gamy * np.outer(z, z))

                if use_error_corr_structured_connectivity:
                    ee = np.dot(Be, e)
                    Be = 1 / lambdae * (Be - game * np.diag(ee * ee))
                    # Be = 1 / lambdae * (Be - game * ((np.outer(ee, ee))))
                    # Be = 1 / lambdae * (Be - game * np.diag(np.diag(np.outer(ee,ee))))

                # Record the seperated signal
                Y[:, idx[i_sample]] = y

                if debugging:
                    if ((i_sample % debug_iteration_point) == 0) | (
                        i_sample == samples - 1
                    ):  # & (i_sample >= debug_iteration_point):
                        self.W = W
                        self.By = By
                        self.Be = Be
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
        self.W = W
        self.By = By
        self.Be = Be

    def fit_batch_nnantisparse(
        self,
        X,
        n_epochs=1,
        neural_dynamic_iterations=250,
        neural_lr_start=0.9,
        neural_lr_stop=1e-3,
        synaptic_lr_rule="constant",
        synaptic_lr_decay_divider=5000,
        neural_loop_lr_rule="divide_by_loop_index",
        neural_lr_decay_multiplier=0.005,
        use_error_corr_structured_connectivity=False,
        shuffle=False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):

        lambday, lambdae, muW, gamy, game, W, By, Be = (
            self.lambday,
            self.lambdae,
            self.muW,
            self.gamy,
            self.game,
            self.W,
            self.By,
            self.Be,
        )
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
        SIR_list = self.SIR_list
        SNR_list = self.SNR_list
        self.SV_list = []

        # h = 1 / gamy # Hopefield parameter

        assert X.shape[0] == self.x_dim, "You must input the transpose"

        samples = X.shape[1]

        if debugging:
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize=(45, 30), dpi=80)
        # Y = np.zeros((self.s_dim, samples))
        Y = np.random.randn(self.s_dim, samples)

        if shuffle:
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:, idx[i_sample]]
                y = np.zeros(self.s_dim)
                # y = np.random.uniform(0,1, size = (self.s_dim,))
                # Output recurrent weights
                # My = By + h * np.eye(self.s_dim)
                y = self.run_neural_dynamics_nnantisparse(
                    x_current,
                    y,
                    W,
                    By,
                    Be,
                    gamy,
                    game,
                    lr_start=neural_lr_start,
                    lr_stop=neural_lr_stop,
                    lr_rule=neural_loop_lr_rule,
                    lr_decay_multiplier=neural_lr_decay_multiplier,
                    neural_dynamic_iterations=neural_dynamic_iterations,
                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                )

                e = y - W @ x_current

                if synaptic_lr_rule == "constant":
                    muW_ = muW
                elif synaptic_lr_rule == "divide_by_log_index":
                    muW_ = np.max(
                        [
                            muW
                            / (1 + np.log(2 + (i_sample // synaptic_lr_decay_divider))),
                            1e-3,
                        ]
                    )
                elif synaptic_lr_rule == "divide_by_index":
                    muW_ = np.max([muW / (i_sample // synaptic_lr_decay_divider), 1e-3])

                W = W + muW_ * np.outer(e, x_current)

                z = By @ y
                By = (1 / lambday) * (By - gamy * np.outer(z, z))

                if use_error_corr_structured_connectivity:
                    ee = np.dot(Be, e)
                    Be = 1 / lambdae * (Be - game * np.diag(ee * ee))

                Y[:, idx[i_sample]] = y

                if debugging:
                    if ((i_sample % debug_iteration_point) == 0) | (
                        i_sample == samples - 1
                    ):  # & (i_sample >= debug_iteration_point):
                        self.W = W
                        self.By = By
                        self.Be = Be
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
        self.W = W
        self.By = By
        self.Be = Be

    def fit_batch_sparse(
        self,
        X,
        n_epochs=1,
        neural_dynamic_iterations=250,
        neural_lr_start=0.9,
        neural_lr_stop=1e-3,
        stlambd_lr=1,
        synaptic_lr_rule="constant",
        synaptic_lr_decay_divider=5000,
        neural_loop_lr_rule="divide_by_loop_index",
        neural_lr_decay_multiplier=0.005,
        use_error_corr_structured_connectivity=False,
        shuffle=False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):

        lambday, lambdae, muW, gamy, game, W, By, Be = (
            self.lambday,
            self.lambdae,
            self.muW,
            self.gamy,
            self.game,
            self.W,
            self.By,
            self.Be,
        )
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
        SIR_list = self.SIR_list
        SNR_list = self.SNR_list
        self.SV_list = []

        # h = 1 / gamy  # Hopefield parameter

        assert X.shape[0] == self.x_dim, "You must input the transpose"

        samples = X.shape[1]

        if debugging:
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize=(45, 30), dpi=80)

        # Y = np.zeros((self.s_dim, samples))
        Y = np.random.randn(self.s_dim, samples)

        if shuffle:
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:, idx[i_sample]]
                y = np.zeros(self.s_dim)
                # y = np.random.uniform(0,1, size = (self.s_dim,))
                # Output recurrent weights
                # My = By + h * np.eye(self.s_dim)
                y = self.run_neural_dynamics_sparse(
                    x_current,
                    y,
                    W,
                    By,
                    Be,
                    gamy,
                    game,
                    lr_start=neural_lr_start,
                    lr_stop=neural_lr_stop,
                    lr_rule=neural_loop_lr_rule,
                    lr_decay_multiplier=neural_lr_decay_multiplier,
                    stlambd_lr=stlambd_lr,
                    neural_dynamic_iterations=neural_dynamic_iterations,
                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                )

                e = y - W @ x_current

                if synaptic_lr_rule == "constant":
                    muW_ = muW
                elif synaptic_lr_rule == "divide_by_log_index":
                    muW_ = np.max(
                        [
                            muW
                            / (1 + np.log(2 + (i_sample // synaptic_lr_decay_divider))),
                            1e-3,
                        ]
                    )
                elif synaptic_lr_rule == "divide_by_index":
                    muW_ = np.max([muW / (i_sample // synaptic_lr_decay_divider), 1e-3])

                W = (1 - 1e-6) * W + muW_ * np.outer(e, x_current)

                z = By @ y
                By = (1 / lambday) * (By - gamy * np.outer(z, z))

                if use_error_corr_structured_connectivity:
                    ee = np.dot(Be, e)
                    Be = 1 / lambdae * (Be - game * np.diag(ee * ee))

                Y[:, idx[i_sample]] = y

                if debugging:
                    if ((i_sample % debug_iteration_point) == 0) | (
                        i_sample == samples - 1
                    ):
                        self.W = W
                        self.By = By
                        self.Be = Be
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
        self.W = W
        self.By = By
        self.Be = Be

    def fit_batch_nnsparse(
        self,
        X,
        n_epochs=1,
        neural_dynamic_iterations=250,
        neural_lr_start=0.9,
        neural_lr_stop=1e-3,
        stlambd_lr=1,
        synaptic_lr_rule="constant",
        synaptic_lr_decay_divider=5000,
        neural_loop_lr_rule="divide_by_loop_index",
        neural_lr_decay_multiplier=0.005,
        use_error_corr_structured_connectivity=False,
        shuffle=False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):

        lambday, lambdae, muW, gamy, game, W, By, Be = (
            self.lambday,
            self.lambdae,
            self.muW,
            self.gamy,
            self.game,
            self.W,
            self.By,
            self.Be,
        )
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
        SIR_list = self.SIR_list
        SNR_list = self.SNR_list
        self.SV_list = []

        # h = 1 / gamy  # Hopefield parameter

        assert X.shape[0] == self.x_dim, "You must input the transpose"

        samples = X.shape[1]

        if debugging:
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize=(45, 30), dpi=80)

        # Y = np.zeros((self.s_dim, samples))
        Y = np.random.randn(self.s_dim, samples)

        if shuffle:
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:, idx[i_sample]]
                y = np.zeros(self.s_dim)
                # y = np.random.uniform(0,1, size = (self.s_dim,))
                # Output recurrent weights
                # My = By + h * np.eye(self.s_dim)
                y = self.run_neural_dynamics_nnsparse(
                    x_current,
                    y,
                    W,
                    By,
                    Be,
                    gamy,
                    game,
                    lr_start=neural_lr_start,
                    lr_stop=neural_lr_stop,
                    lr_rule=neural_loop_lr_rule,
                    lr_decay_multiplier=neural_lr_decay_multiplier,
                    stlambd_lr=stlambd_lr,
                    neural_dynamic_iterations=neural_dynamic_iterations,
                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                )

                e = y - W @ x_current

                if synaptic_lr_rule == "constant":
                    muW_ = muW
                elif synaptic_lr_rule == "divide_by_log_index":
                    muW_ = np.max(
                        [
                            muW
                            / (1 + np.log(2 + (i_sample // synaptic_lr_decay_divider))),
                            1e-3,
                        ]
                    )
                elif synaptic_lr_rule == "divide_by_index":
                    muW_ = np.max([muW / (i_sample // synaptic_lr_decay_divider), 1e-3])

                W = W + muW_ * np.outer(e, x_current)

                z = By @ y
                By = (1 / lambday) * (By - gamy * np.outer(z, z))

                if use_error_corr_structured_connectivity:
                    ee = np.dot(Be, e)
                    Be = 1 / lambdae * (Be - game * np.diag(ee * ee))

                Y[:, idx[i_sample]] = y

                if debugging:
                    if ((i_sample % debug_iteration_point) == 0) | (
                        i_sample == samples - 1
                    ):
                        self.W = W
                        self.By = By
                        self.Be = Be
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

        self.W = W
        self.By = By
        self.Be = Be

    def fit_batch_simplex(
        self,
        X,
        n_epochs=1,
        neural_lr_start=0.9,
        neural_lr_stop=1e-3,
        stlambd_lr=0.05,
        neural_dynamic_iterations=250,
        synaptic_lr_rule="constant",
        synaptic_lr_decay_divider=5000,
        neural_loop_lr_rule="divide_by_loop_index",
        neural_lr_decay_multiplier=0.005,
        use_error_corr_structured_connectivity=False,
        shuffle=False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):

        lambday, lambdae, muW, gamy, game, W, By, Be = (
            self.lambday,
            self.lambdae,
            self.muW,
            self.gamy,
            self.game,
            self.W,
            self.By,
            self.Be,
        )
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
        SIR_list = self.SIR_list
        SNR_list = self.SNR_list
        self.SV_list = []

        # h = 1 / gamy # Hopefield parameter

        assert X.shape[0] == self.x_dim, "You must input the transpose"

        samples = X.shape[1]

        if debugging:
            S = self.S
            A = self.A
            Szeromean = S - S.mean(axis=1).reshape(-1, 1)
            if plot_in_jupyter:
                plt.figure(figsize=(45, 30), dpi=80)

        # Y = np.zeros((self.s_dim, samples))
        Y = np.random.randn(self.s_dim, samples)

        if shuffle:
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:, idx[i_sample]]
                y = np.zeros(self.s_dim)
                # y = np.random.uniform(0,1, size = (self.s_dim,))
                # Output recurrent weights
                # My = By + h * np.eye(self.s_dim)
                y = self.run_neural_dynamics_simplex(
                    x_current,
                    y,
                    W,
                    By,
                    Be,
                    gamy,
                    game,
                    lr_start=neural_lr_start,
                    lr_stop=neural_lr_stop,
                    lr_rule=neural_loop_lr_rule,
                    lr_decay_multiplier=neural_lr_decay_multiplier,
                    stlambd_lr=stlambd_lr,
                    neural_dynamic_iterations=neural_dynamic_iterations,
                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                )

                e = y - W @ x_current

                if synaptic_lr_rule == "constant":
                    muW_ = muW
                elif synaptic_lr_rule == "divide_by_log_index":
                    muW_ = np.max(
                        [
                            muW
                            / (1 + np.log(2 + (i_sample // synaptic_lr_decay_divider))),
                            1e-3,
                        ]
                    )
                elif synaptic_lr_rule == "divide_by_index":
                    muW_ = np.max([muW / (i_sample // synaptic_lr_decay_divider), 1e-3])

                W = W + muW_ * np.outer(e, x_current)

                z = By @ y
                By = (1 / lambday) * (By - gamy * np.outer(z, z))

                if use_error_corr_structured_connectivity:
                    ee = np.dot(Be, e)
                    Be = 1 / lambdae * (Be - game * np.diag(ee * ee))
                # Record the seperated signal
                Y[:, idx[i_sample]] = y

                if debugging:
                    if ((i_sample % debug_iteration_point) == 0) | (
                        i_sample == samples - 1
                    ):  # & (i_sample > debug_iteration_point):
                        self.W = W
                        self.By = By
                        self.Be = Be
                        try:
                            Wf = self.compute_overall_mapping(return_mapping=True)

                            SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(
                                Wf, A, Szeromean, X, True
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

        self.W = W
        self.By = By
        self.Be = Be

    def fit_batch_general_polytope(
        self,
        X,
        signed_dims,
        nn_dims,
        sparse_dims_list,
        n_epochs=1,
        neural_lr_start=0.9,
        neural_lr_stop=1e-15,
        stlambd_lr=0.05,
        neural_dynamic_iterations=250,
        synaptic_lr_rule="constant",
        synaptic_lr_decay_divider=5000,
        neural_loop_lr_rule="divide_by_loop_index",
        neural_lr_decay_multiplier=0.005,
        use_error_corr_structured_connectivity=False,
        shuffle=False,
        mean_center_for_debug_evaluation = False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):
        lambday, lambdae, muW, gamy, game, W, By, Be = (
            self.lambday,
            self.lambdae,
            self.muW,
            self.gamy,
            self.game,
            self.W,
            self.By,
            self.Be,
        )
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
        SIR_list = self.SIR_list
        SNR_list = self.SNR_list
        self.SV_list = []

        # h = 1 / gamy  # Hopefield parameter

        assert X.shape[0] == self.x_dim, "You must input the transpose"

        samples = X.shape[1]

        if debugging:
            S = self.S
            A = self.A
            Szeromean = S - S.mean(axis=1).reshape(-1, 1)
            if plot_in_jupyter:
                plt.figure(figsize=(45, 30), dpi=80)

        # Y = np.zeros((self.s_dim, samples))
        Y = np.random.randn(self.s_dim, samples)

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
                x_current = X[:, idx[i_sample]]
                y = np.zeros(self.s_dim)

                # Output recurrent weights
                # My = By + h * np.eye(self.s_dim)
                y = self.run_neural_dynamics_general_polytope(
                    x_current,
                    y,
                    signed_dims,
                    nn_dims,
                    sparse_dims_list,
                    W,
                    By,
                    Be,
                    gamy,
                    game,
                    lr_start=neural_lr_start,
                    lr_stop=neural_lr_stop,
                    lr_rule=neural_loop_lr_rule,
                    lr_decay_multiplier=neural_lr_decay_multiplier,
                    stlambd_lr=stlambd_lr,
                    neural_dynamic_iterations=neural_dynamic_iterations,
                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                )

                e = y - W @ x_current

                if synaptic_lr_rule == "constant":
                    muW_ = muW
                elif synaptic_lr_rule == "divide_by_log_index":
                    muW_ = np.max(
                        [
                            muW
                            / (1 + np.log(2 + (i_sample // synaptic_lr_decay_divider))),
                            1e-3,
                        ]
                    )
                elif synaptic_lr_rule == "divide_by_index":
                    muW_ = np.max([muW / (i_sample // synaptic_lr_decay_divider), 1e-3])

                W = W + muW_ * np.outer(e, x_current)

                z = By @ y
                By = (1 / lambday) * (By - gamy * np.outer(z, z))

                if use_error_corr_structured_connectivity:
                    ee = np.dot(Be, e)
                    Be = 1 / lambdae * (Be - game * np.diag(ee * ee))
                # Record the seperated signal
                Y[:, idx[i_sample]] = y

                if debugging:
                    if ((i_sample % debug_iteration_point) == 0) | (
                        i_sample == samples - 1
                    ):
                        self.W = W
                        self.By = By
                        self.Be = Be
                        try:
                            Wf = self.compute_overall_mapping(return_mapping=True)
                            if mean_center_for_debug_evaluation:
                                SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(
                                    Wf, A, Szeromean, X, True
                                )
                            else:
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
        self.W = W
        self.By = By
        self.Be = Be

    def fit_batch_mixedantisparse(
        self,
        X,
        nn_components,
        n_epochs=1,
        neural_dynamic_iterations=250,
        neural_lr_start=0.9,
        neural_lr_stop=1e-3,
        synaptic_lr_rule="constant",
        synaptic_lr_decay_divider=5000,
        neural_loop_lr_rule="divide_by_loop_index",
        neural_lr_decay_multiplier=0.005,
        shuffle=False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):

        lambday, lambdae, muW, gamy, game, W, By, Be = (
            self.lambday,
            self.lambdae,
            self.muW,
            self.gamy,
            self.game,
            self.W,
            self.By,
            self.Be,
        )
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        # h = 1 / gamy  # Hopefield parameter

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
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:, idx[i_sample]]
                y = np.zeros(self.s_dim)

                # Output recurrent weights
                # My = By + h * np.eye(self.s_dim)
                y = self.run_neural_dynamics_mixedantisparse(
                    x_current,
                    y,
                    nn_components,
                    signed_components,
                    W,
                    By,
                    Be,
                    gamy,
                    game,
                    lr_start=neural_lr_start,
                    lr_stop=neural_lr_stop,
                    lr_rule=neural_loop_lr_rule,
                    lr_decay_multiplier=neural_lr_decay_multiplier,
                    neural_dynamic_iterations=neural_dynamic_iterations,
                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                )

                e = y - W @ x_current

                if synaptic_lr_rule == "constant":
                    muW_ = muW
                elif synaptic_lr_rule == "divide_by_log_index":
                    muW_ = np.max(
                        [
                            muW
                            / (1 + np.log(2 + (i_sample // synaptic_lr_decay_divider))),
                            1e-3,
                        ]
                    )
                elif synaptic_lr_rule == "divide_by_index":
                    muW_ = np.max([muW / (i_sample // synaptic_lr_decay_divider), 1e-3])

                W = W + muW_ * np.outer(e, x_current)

                z = By @ y
                By = (1 / lambday) * (By - gamy * np.outer(z, z))

                ee = np.dot(Be, e)
                Be = 1 / lambdae * (Be - game * np.outer(ee, ee))
                # Record the seperated signal
                Y[:, idx[i_sample]] = y

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.By = By
                        self.Be = Be
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
        self.By = By
        self.Be = Be

    def fit_batch_nnwsubsparse(
        self,
        X,
        sparse_components,
        n_epochs=1,
        neural_dynamic_iterations=250,
        neural_lr_start=0.9,
        neural_lr_stop=1e-3,
        shuffle=False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):

        lambday, lambdae, muW, gamy, game, W, By, Be = (
            self.lambday,
            self.lambdae,
            self.muW,
            self.gamy,
            self.game,
            self.W,
            self.By,
            self.Be,
        )
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        h = 1 / gamy  # Hopefield parameter

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
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:, idx[i_sample]]
                y = np.zeros(self.s_dim)
                # y = np.random.uniform(0,1, size = (self.s_dim,))
                # Output recurrent weights
                My = By + h * np.eye(self.s_dim)
                y = self.run_neural_dynamics_nnwsubsparse(
                    x_current,
                    y,
                    nn_components,
                    sparse_components,
                    W,
                    My,
                    Be,
                    gamy,
                    game,
                    neural_lr_start,
                    neural_lr_stop,
                    neural_dynamic_iterations,
                    neural_dynamic_tol,
                )

                e = y - W @ x_current

                W = (1 - 1e-6) * W + muW * np.outer(e, x_current)

                z = By @ y
                By = (1 / lambday) * (By - gamy * np.outer(z, z))

                # ee = np.dot(Be,e)
                # Be = 1 / lambdae * (Be - game * np.outer(ee,ee))
                # # Record the seperated signal
                Y[:, idx[i_sample]] = y

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.By = By
                        self.Be = Be
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
        self.By = By
        self.Be = Be

    def fit_batch_nnwsubnnsparse(
        self,
        X,
        nnsparse_components,
        n_epochs=1,
        neural_dynamic_iterations=250,
        neural_lr_start=0.9,
        neural_lr_stop=1e-3,
        shuffle=False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):

        lambday, lambdae, muW, gamy, game, W, By, Be = (
            self.lambday,
            self.lambdae,
            self.muW,
            self.gamy,
            self.game,
            self.W,
            self.By,
            self.Be,
        )
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth

        h = 1 / gamy  # Hopefield parameter

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
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:, idx[i_sample]]
                y = np.zeros(self.s_dim)
                # y = np.random.uniform(0,1, size = (self.s_dim,))
                # Output recurrent weights
                My = By + h * np.eye(self.s_dim)
                y = self.run_neural_dynamics_nnwsubnnsparse(
                    x_current,
                    y,
                    nn_components,
                    nnsparse_components,
                    W,
                    My,
                    Be,
                    gamy,
                    game,
                    neural_lr_start,
                    neural_lr_stop,
                    neural_dynamic_iterations,
                    neural_dynamic_tol,
                )

                e = y - W @ x_current

                W = (1 - 1e-6) * W + muW * np.outer(e, x_current)

                z = By @ y
                By = (1 / lambday) * (By - gamy * np.outer(z, z))

                # ee = np.dot(Be,e)
                # Be = 1 / lambdae * (Be - game * np.outer(ee,ee))
                # Record the seperated signal
                Y[:, idx[i_sample]] = y

                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.By = By
                        self.Be = Be
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
        self.By = By
        self.Be = Be

class OnlineCorInfoMaxOlshaussen(OnlineCorInfoMax):

    @staticmethod
    @njit
    def run_neural_dynamics(
        x,
        y,
        W,
        By,
        Be,
        gamy,
        game,
        lr_start=0.9,
        lr_stop=1e-15,
        lr_rule="divide_by_loop_index",
        lr_decay_multiplier=0.01,
        stlambd_lr=1,
        neural_dynamic_iterations=100,
        neural_OUTPUT_COMP_TOL=1e-7,
        case = 'sparse',
    ):
        def SoftThresholding(X, thresh):
            X_absolute = np.abs(X)
            X_sign = np.sign(X)
            X_thresholded = (X_absolute > thresh) * (X_absolute - thresh) * X_sign
            return X_thresholded
        STLAMBD = 0
        dval = 0
        yke = np.dot(W, x)
        if case == 'sparse':
            for j in range(neural_dynamic_iterations):
                # mu_y = max(lr_start / (j + 1), lr_stop)
                if lr_rule == "constant":
                    mu_y = lr_start
                elif lr_rule == "divide_by_loop_index":
                    mu_y = max(lr_start / (j + 1), lr_stop)
                elif lr_rule == "divide_by_slow_loop_index":
                    mu_y = max(lr_start / (j * lr_decay_multiplier + 1), lr_stop)
                elif lr_rule == "divide_by_sqrt_loop_index":
                    MUV = max(lr_start / np.sqrt(j + 1), lr_stop)

                # STLAMBD = 0
                y_old = y.copy()
                e = yke - y
                grady = gamy * By @ y + game * (Be.T[0] * e)
                # y = y + mu_y * (grady)
                a = y + mu_y * (grady)
                # SOFT THRESHOLDING
                y = SoftThresholding(a, 0)
                temp = 1
                if np.linalg.norm(a,1) >= 1:
                    iter2 = 0
                    while ((np.abs(STLAMBD - temp) / np.abs(STLAMBD + 1e-10)) > 1e-5) & (iter2 < 10):
                        iter2 += 1
                        temp = STLAMBD

                        y = SoftThresholding(a, STLAMBD)
                        sstep = stlambd_lr / np.sqrt(iter2)
                        dval = np.linalg.norm(y,1) - 1
                        STLAMBD = STLAMBD + sstep * dval
                        if STLAMBD < 0:
                            STLAMBD = 0
                            y = a
                    y = np.clip(y, -1, 1)
                # dval = np.linalg.norm(y, 1) - 1
                # STLAMBD = max(STLAMBD + stlambd_lr * dval, 0)

                if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                    break
        elif case == "nnsparse":             
            for j in range(neural_dynamic_iterations):
                if lr_rule == "constant":
                    mu_y = lr_start
                elif lr_rule == "divide_by_loop_index":
                    mu_y = max(lr_start / (j + 1), lr_stop)
                elif lr_rule == "divide_by_slow_loop_index":
                    mu_y = max(lr_start / (j * lr_decay_multiplier + 1), lr_stop)

                y_old = y.copy()
                e = yke - y
                grady = gamy * By @ y + game * (Be.T[0] * e)
                a = y + mu_y * (grady)

                y = np.maximum(a - STLAMBD, 0)
                temp = 1
                if np.linalg.norm(a,1) >= 1:
                    iter2 = 0
                    while ((np.abs(STLAMBD - temp) / np.abs(STLAMBD + 1e-10)) > 1e-5) & (iter2 < 10):
                        iter2 += 1
                        temp = STLAMBD

                        y = np.maximum(a - STLAMBD, 0)
                        sstep = stlambd_lr / np.sqrt(iter2)
                        dval = np.sum(y) - 1
                        STLAMBD = STLAMBD + sstep * dval
                        if STLAMBD < 0:
                            STLAMBD = 0
                            y = a
                    y = np.clip(y, 0, 1)
                # dval = np.sum(y) - 1
                # STLAMBD = max(STLAMBD + stlambd_lr * dval, 0)

                if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                    break
        return y

    def ZeroOneNormalizeData(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def plot_receptive_fields(self, Wf):
        pl.clf()
        fig, ax = pl.subplots(12,12, figsize = (20,20))
        for l in range(144):
            rf = np.reshape(-Wf[l,:], (12,12))
            rf = self.ZeroOneNormalizeData(rf)
            ax[l//12, l%12].imshow(rf, cmap = 'gray')
            ax[l//12, l%12].axes.xaxis.set_visible(False)
            ax[l//12, l%12].axes.yaxis.set_visible(False)
        pl.subplots_adjust( right=0.97,\
                            left=0.03,\
                            bottom=0.03,\
                            top=0.97,\
                            wspace=0.1,\
                            hspace=0.1)
        clear_output(wait=True)
        display(pl.gcf())

    def fit_batch(
        self,
        X,
        n_epochs=1,
        case = "nnsparse", # options: 'sparse' or 'nnsparse'
        neural_dynamic_iterations=250,
        neural_lr_start=0.1,
        neural_lr_stop=1e-3,
        stlambd_lr=1,
        synaptic_lr_rule="constant",
        synaptic_lr_decay_divider=5000,
        neural_loop_lr_rule="divide_by_loop_index",
        neural_lr_decay_multiplier=0.005,
        use_error_corr_structured_connectivity=False,
        shuffle=True,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):
        lambday, lambdae, muW, gamy, game, W, By, Be = (
            self.lambday,
            self.lambdae,
            self.muW,
            self.gamy,
            self.game,
            self.W,
            self.By,
            self.Be,
        )
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        Be = np.diag(Be)[:,np.newaxis]
        assert X.shape[0] == self.x_dim, "You must input the transpose"

        samples = X.shape[1]
        Y = np.zeros((self.s_dim, samples))
        # Y = np.random.randn(self.s_dim, samples)

        if shuffle:
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:, idx[i_sample]]
                y = np.zeros(self.s_dim)

                y = self.run_neural_dynamics(
                    x_current,
                    y,
                    W,
                    By,
                    Be,
                    gamy,
                    game,
                    lr_start=neural_lr_start,
                    lr_stop=neural_lr_stop,
                    lr_rule=neural_loop_lr_rule,
                    lr_decay_multiplier=neural_lr_decay_multiplier,
                    stlambd_lr=stlambd_lr,
                    neural_dynamic_iterations=neural_dynamic_iterations,
                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                    case = case,
                )

                e = y - W @ x_current

                if synaptic_lr_rule == "constant":
                    muW_ = muW
                elif synaptic_lr_rule == "divide_by_log_index":
                    muW_ = np.max(
                        [
                            muW
                            / (1 + np.log(2 + (i_sample // synaptic_lr_decay_divider))),
                            1e-3,
                        ]
                    )
                elif synaptic_lr_rule == "divide_by_index":
                    muW_ = np.max([muW / (i_sample // synaptic_lr_decay_divider), 1e-3])

                W = W + muW_ * np.outer(e, x_current)

                z = By @ y
                By = (1 / lambday) * (By - gamy * np.outer(z, z))

                if use_error_corr_structured_connectivity:
                    ee = np.dot(Be, e)
                    Be = 1 / lambdae * (Be - game * np.diag(ee * ee))

                Y[:, idx[i_sample]] = y

                if plot_in_jupyter:
                    if ((i_sample % debug_iteration_point) == 0) | (
                        i_sample == samples - 1
                    ):
                        self.W = W
                        self.By = By
                        self.Be = np.diag(Be[:,0])
                        try:
                            Wf = self.compute_overall_mapping(return_mapping=True)
                            self.plot_receptive_fields(Wf)
                        except Exception as e:
                            print(str(e))

class OnlineCorInfoMaxCanonical(OnlineCorInfoMax):
    @staticmethod
    @njit
    def run_neural_dynamics(
        x,
        y,
        Apoly,
        bpoly,
        W,
        By,
        Be,
        gamy,
        game,
        lr_start=0.9,
        lr_stop=1e-15,
        lagrangian_lambd_lr = 0.01,
        lr_rule="divide_by_loop_index",
        lr_decay_multiplier=0.01,
        neural_dynamic_iterations=100,
        neural_OUTPUT_COMP_TOL=1e-7,        
    ):
        yke = np.dot(W, x)
        lambda_Lagrangian = np.zeros(Apoly.shape[0])
        for j in range(neural_dynamic_iterations):
            # mu_y = max(lr_start / (j + 1), lr_stop)
            if lr_rule == "constant":
                mu_y = lr_start
            elif lr_rule == "divide_by_loop_index":
                mu_y = max(lr_start / (j + 1), lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                mu_y = max(lr_start / (j * lr_decay_multiplier + 1), lr_stop)
            y_old = y.copy()
            e = yke - y
            grady = gamy * By @ y + game * Be @ e - Apoly.T @ lambda_Lagrangian
            # y = np.clip(y + mu_y * (grady), -1, 1)
            y = y + mu_y * (grady)
            lambda_Lagrangian = np.maximum(lambda_Lagrangian + lagrangian_lambd_lr * (Apoly @ y - bpoly), 0)
            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    def fit_batch(
        self,
        X,
        Apoly,
        bpoly,
        n_epochs=1,
        neural_dynamic_iterations=250,
        neural_lr_start=0.9,
        neural_lr_stop=1e-15,
        lagrangian_lambd_lr = 0.01,
        synaptic_lr_rule="constant",
        synaptic_lr_decay_divider=5000,
        neural_loop_lr_rule="divide_by_loop_index",
        neural_lr_decay_multiplier=0.005,
        use_error_corr_structured_connectivity=False,
        shuffle=False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):

        lambday, lambdae, muW, gamy, game, W, By, Be = (
            self.lambday,
            self.lambdae,
            self.muW,
            self.gamy,
            self.game,
            self.W,
            self.By,
            self.Be,
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

        Y = np.random.randn(self.s_dim, samples)

        if shuffle:
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:, idx[i_sample]]
                y = np.zeros(self.s_dim)

                # Output recurrent weights
                y = self.run_neural_dynamics(
                    x_current,
                    y,
                    Apoly,
                    bpoly,
                    W,
                    By,
                    Be,
                    gamy,
                    game,
                    lr_start=neural_lr_start,
                    lr_stop=neural_lr_stop,
                    lagrangian_lambd_lr=lagrangian_lambd_lr,
                    lr_rule=neural_loop_lr_rule,
                    lr_decay_multiplier=neural_lr_decay_multiplier,
                    neural_dynamic_iterations=neural_dynamic_iterations,
                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                )

                e = y - W @ x_current

                if synaptic_lr_rule == "constant":
                    muW_ = muW
                elif synaptic_lr_rule == "divide_by_log_index":
                    muW_ = np.max(
                        [
                            muW
                            / (1 + np.log(2 + (i_sample // synaptic_lr_decay_divider))),
                            1e-3,
                        ]
                    )
                elif synaptic_lr_rule == "divide_by_index":
                    muW_ = np.max([muW / (i_sample // synaptic_lr_decay_divider), 1e-3])

                W = W + muW_ * np.outer(e, x_current)

                z = By @ y
                By = (1 / lambday) * (By - gamy * np.outer(z, z))

                if use_error_corr_structured_connectivity:
                    ee = np.dot(Be, e)
                    Be = 1 / lambdae * (Be - game * np.diag(ee * ee))

                # Record the seperated signal
                Y[:, idx[i_sample]] = y

                if debugging:
                    if ((i_sample % debug_iteration_point) == 0) | (
                        i_sample == samples - 1
                    ):  # & (i_sample >= debug_iteration_point):
                        self.W = W
                        self.By = By
                        self.Be = Be
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
        self.W = W
        self.By = By
        self.Be = Be

class CorInfoMaxVideoSeparation(OnlineCorInfoMax):
    
    def seperate_videos(self, Wf_list, X, imsize = [360, 640], n_pixel_per_frame=5, 
                        n_iter = 1000000, neural_dynamic_iterations = 250, 
                        neural_lr_start = 0.9, neural_lr_stop = 1e-3, 
                        neural_loop_lr_rule="divide_by_loop_index",
                        neural_lr_decay_multiplier=0.005,
                        use_error_corr_structured_connectivity=False,
                        shuffle=False,
                        debug_iteration_point=1000,
                        plot_in_jupyter=False,):
        
        lambday, lambdae, muW, gamy, game, W, By, Be = (
            self.lambday,
            self.lambdae,
            self.muW,
            self.gamy,
            self.game,
            self.W,
            self.By,
            self.Be,
        )        
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
        number_of_mixtures = X.shape[1]
        number_of_frames = X.shape[0]

        
        samples = X.shape[2]
                
        if debugging:
            SIRlist = []
            A = self.A
            
        sample_counter = 0
        for i_sample in tqdm(range(n_iter)):
            x_current = X[np.mod(sample_counter // n_pixel_per_frame, 10),:, np.random.randint(samples)]
            y = np.zeros(self.s_dim)

            y = self.run_neural_dynamics_nnantisparse(
                                                    x_current,
                                                    y,
                                                    W,
                                                    By,
                                                    Be,
                                                    gamy,
                                                    game,
                                                    lr_start=neural_lr_start,
                                                    lr_stop=neural_lr_stop,
                                                    lr_rule=neural_loop_lr_rule,
                                                    lr_decay_multiplier=neural_lr_decay_multiplier,
                                                    neural_dynamic_iterations=neural_dynamic_iterations,
                                                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                                                )

            e = y - W @ x_current

            W = W + muW * np.outer(e, x_current)

            z = By @ y
            By = (1/lambday) * (By - gamy * np.outer(z, z))
            
            if debugging:
                if (i_sample % debug_iteration_point) == 0:
                    self.W = W
                    self.By = By
                    self.Be = Be
                    Wf = self.compute_overall_mapping(return_mapping = True)
                    Wf_list.append(Wf)
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
            sample_counter += 1
        return X

######## BELOW CLASSES ARE FOR EXPERIMENTAL PURPOSES.DO NOT TAKE THEM SERIOUS FOR NOW.
######## THESE ARE STILL UNDER DEVELOPMENT.
class OnlineCorInfoMaxICA(OnlineCorInfoMax):
    ## !!!!!!!!!!!!!!!!!! JUST TO TRY SOMETHING. NOT WORKING PROPERLY I GUESS #######
    ###############################################################
    ############### NEURAL DYNAMICS ALGORITHMS ####################
    ###############################################################
    @staticmethod
    @njit
    def run_neural_dynamics_ica(
        x,
        y,
        W,
        By,
        Be,
        gamy,
        game,
        lr_start=0.9,
        lr_stop=1e-15,
        lr_rule="divide_by_loop_index",
        lr_decay_multiplier=0.01,
        neural_dynamic_iterations=100,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):
        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            # mu_y = max(lr_start / (j + 1), lr_stop)
            if lr_rule == "constant":
                mu_y = lr_start
            elif lr_rule == "divide_by_loop_index":
                mu_y = max(lr_start / (j + 1), lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                mu_y = max(lr_start / (j * lr_decay_multiplier + 1), lr_stop)
            y_old = y.copy()
            e = yke - y
            grady = gamy * By @ y + game * Be @ e
            y = y + mu_y * (grady)
            y = np.tanh(y)
        # for j in range(neural_dynamic_iterations):
        #     y_old = y.copy()
        #     Wf = -np.linalg.solve(gamy * By - game * Be, game * Be @ W)
        #     y = Wf @ x
            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y
                      
    def fit_batch_ica(
        self,
        X,
        n_epochs=1,
        neural_dynamic_iterations=250,
        neural_lr_start=0.9,
        neural_lr_stop=1e-15,
        synaptic_lr_rule="constant",
        synaptic_lr_decay_divider=5000,
        neural_loop_lr_rule="divide_by_loop_index",
        neural_lr_decay_multiplier=0.005,
        use_error_corr_structured_connectivity=False,
        shuffle=False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):

        lambday, lambdae, muW, gamy, game, W, By, Be = (
            self.lambday,
            self.lambdae,
            self.muW,
            self.gamy,
            self.game,
            self.W,
            self.By,
            self.Be,
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

        Y = np.random.randn(self.s_dim, samples)

        if shuffle:
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:, idx[i_sample]]
                y = np.zeros(self.s_dim)

                # Output recurrent weights
                y = self.run_neural_dynamics_ica(
                    x_current,
                    y,
                    W,
                    By,
                    Be,
                    gamy,
                    game,
                    lr_start=neural_lr_start,
                    lr_stop=neural_lr_stop,
                    lr_rule=neural_loop_lr_rule,
                    lr_decay_multiplier=neural_lr_decay_multiplier,
                    neural_dynamic_iterations=neural_dynamic_iterations,
                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                )

                e = y - W @ x_current

                if synaptic_lr_rule == "constant":
                    muW_ = muW
                elif synaptic_lr_rule == "divide_by_log_index":
                    muW_ = np.max(
                        [
                            muW
                            / (1 + np.log(2 + (i_sample // synaptic_lr_decay_divider))),
                            1e-3,
                        ]
                    )
                elif synaptic_lr_rule == "divide_by_index":
                    muW_ = np.max([muW / (i_sample // synaptic_lr_decay_divider), 1e-3])

                W = W + muW_ * np.outer(e, x_current)

                z = By @ y
                By = (1 / lambday) * (By - gamy * np.outer(z, z))

                if use_error_corr_structured_connectivity:
                    ee = np.dot(Be, e)
                    Be = 1 / lambdae * (Be - game * np.diag(ee * ee))

                # Record the seperated signal
                Y[:, idx[i_sample]] = y

                if debugging:
                    if ((i_sample % debug_iteration_point) == 0) | (
                        i_sample == samples - 1
                    ):  # & (i_sample >= debug_iteration_point):
                        self.W = W
                        self.By = By
                        self.Be = Be
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
        self.W = W
        self.By = By
        self.Be = Be

class OnlineCorInfoMaxCanonicalv2(OnlineCorInfoMax):

    def run_neural_dynamics(
        self,
        x,
        y,
        Apoly,
        bpoly,
        W,
        By,
        Be,
        gamy,
        game,
        lr_start=0.9,
        lr_stop=1e-15,
        lr_rule="divide_by_loop_index",
        lr_decay_multiplier=0.01,
        neural_dynamic_iterations=100,
        neural_OUTPUT_COMP_TOL=1e-7,        
    ):
        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            # mu_y = max(lr_start / (j + 1), lr_stop)
            if lr_rule == "constant":
                mu_y = lr_start
            elif lr_rule == "divide_by_loop_index":
                mu_y = max(lr_start / (j + 1), lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                mu_y = max(lr_start / (j * lr_decay_multiplier + 1), lr_stop)
            y_old = y.copy()
            e = yke - y
            grady = gamy * By @ y + game * Be @ e
            y = y + mu_y * (grady)
            y = ProjectOntoPolytopeCanonical(y, Apoly, bpoly, np.zeros(y.shape[0]), print_message = False)
            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    def fit_batch(
        self,
        X,
        Apoly,
        bpoly,
        n_epochs=1,
        neural_dynamic_iterations=250,
        neural_lr_start=0.9,
        neural_lr_stop=1e-15,
        synaptic_lr_rule="constant",
        synaptic_lr_decay_divider=5000,
        neural_loop_lr_rule="divide_by_loop_index",
        neural_lr_decay_multiplier=0.005,
        use_error_corr_structured_connectivity=False,
        shuffle=False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):

        lambday, lambdae, muW, gamy, game, W, By, Be = (
            self.lambday,
            self.lambdae,
            self.muW,
            self.gamy,
            self.game,
            self.W,
            self.By,
            self.Be,
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

        Y = np.random.randn(self.s_dim, samples)

        if shuffle:
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:, idx[i_sample]]
                y = np.zeros(self.s_dim)

                # Output recurrent weights
                y = self.run_neural_dynamics(
                    x_current,
                    y,
                    Apoly,
                    bpoly,
                    W,
                    By,
                    Be,
                    gamy,
                    game,
                    lr_start=neural_lr_start,
                    lr_stop=neural_lr_stop,
                    lr_rule=neural_loop_lr_rule,
                    lr_decay_multiplier=neural_lr_decay_multiplier,
                    neural_dynamic_iterations=neural_dynamic_iterations,
                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                )

                e = y - W @ x_current

                if synaptic_lr_rule == "constant":
                    muW_ = muW
                elif synaptic_lr_rule == "divide_by_log_index":
                    muW_ = np.max(
                        [
                            muW
                            / (1 + np.log(2 + (i_sample // synaptic_lr_decay_divider))),
                            1e-3,
                        ]
                    )
                elif synaptic_lr_rule == "divide_by_index":
                    muW_ = np.max([muW / (i_sample // synaptic_lr_decay_divider), 1e-3])

                W = W + muW_ * np.outer(e, x_current)

                z = By @ y
                By = (1 / lambday) * (By - gamy * np.outer(z, z))

                if use_error_corr_structured_connectivity:
                    ee = np.dot(Be, e)
                    Be = 1 / lambdae * (Be - game * np.diag(ee * ee))

                # Record the seperated signal
                Y[:, idx[i_sample]] = y

                if debugging:
                    if ((i_sample % debug_iteration_point) == 0) | (
                        i_sample == samples - 1
                    ):  # & (i_sample >= debug_iteration_point):
                        self.W = W
                        self.By = By
                        self.Be = Be
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
        self.W = W
        self.By = By
        self.Be = Be
