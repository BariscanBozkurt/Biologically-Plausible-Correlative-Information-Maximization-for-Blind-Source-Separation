import argparse
import os
import pickle

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pylab as pl
from IPython import display
from mne.preprocessing import ICA
from scipy.signal import lfilter
from scipy.stats import chi2, invgamma, t

parser = argparse.ArgumentParser(description="Compare LD vs ICA")
parser.add_argument("--r", default=5, type=int, help="Number of sources")
parser.add_argument("--M", default=8, type=int, help="Number of mixtures")
parser.add_argument("--N", default=30000, type=int, help="Number of samples")
parser.add_argument(
    "--NumberofIterations",
    default=10000,
    type=int,
    help="Number of LDInfomax Iterations",
)
parser.add_argument("--NumAverages", default=100, type=int, help="Number of Averages")
parser.add_argument("--SNR", default=40, type=float, help="SNR")
parser.add_argument("--epsv", default=1e-5, type=float, help="epsilon value")
parser.add_argument(
    "--outname", default="SIR40Av100script.pickle", type=str, help="Output Name"
)


args = parser.parse_args()
# Number of sources
r = args.r
print("Number of sources:", r)
print(r)
# Number of mixtures
M = args.M
print("Number of mixtures:", M)
# Number of samples
N = args.N
print("Number of samples:", N)
# Number of Iterations
NumberofIterations = args.NumberofIterations
print("Number of iterations:", NumberofIterations)
# Number of Averages
NumAverages = args.NumAverages
print("Number of Averages:", NumAverages)
# SNR level
SNR = args.SNR  # dB
print("SNR:", SNR)
# Output filename
outname = args.outname
print("Output filename:", outname)
# Epsilon
epsvmax = args.epsv
print("Epsilon value:", epsvmax)


NoiseAmp = (10 ** (-SNR / 20)) * np.sqrt(r)


# Define number of sampling points
n_samples = N
# Degrees of freedom
df = 4

# Correlation values
rholist = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
SIRLDInfoMax = np.zeros((len(rholist), NumAverages))
SINRLDInfoMax = np.zeros((len(rholist), NumAverages))
MSELDInfoMax = np.zeros((len(rholist), NumAverages))
SigPowList = np.zeros((len(rholist), NumAverages))
SIRLDInfoMaxW = np.zeros((len(rholist), NumAverages))
SIRICAInfoMax = np.zeros((len(rholist), NumAverages))
SINRICAInfoMax = np.zeros((len(rholist), NumAverages))
MSEICAInfoMax = np.zeros((len(rholist), NumAverages))
# Nonnegative Clipping function
def nonnegativeclipping(inp):
    out = (inp > 0) * ((inp) < 1.0) * inp + (inp >= 1.0) * 1.0 + (inp <= 0) * 0.0
    return out


# Calculate SIR Function
def CalculateSIR(H, pH):
    G = pH @ H
    Gmax = np.diag(np.max(abs(G), axis=1))
    P = 1.0 * ((np.linalg.inv(Gmax) @ np.abs(G)) > 0.99)
    T = G @ P.T
    rankP = np.linalg.matrix_rank(P)
    SIRV = np.linalg.norm(np.diag(T)) ** 2 / (
        np.linalg.norm(T, "fro") ** 2 - np.linalg.norm(np.diag(T)) ** 2
    )
    return SIRV, rankP


def CalculateSINR(Out, S):
    r = S.shape[0]
    G = np.dot(
        Out - np.reshape(np.mean(Out, 1), (r, 1)),
        np.linalg.pinv(S - np.reshape(np.mean(S, 1), (r, 1))),
    )
    indmax = np.argmax(np.abs(G), 1)
    GG = np.zeros((r, r))
    for kk in range(r):
        GG[kk, indmax[kk]] = np.dot(
            Out[kk, :] - np.mean(Out[kk, :]),
            S[indmax[kk], :].T - np.mean(S[indmax[kk], :]),
        ) / np.dot(
            S[indmax[kk], :] - np.mean(S[indmax[kk], :]),
            S[indmax[kk], :].T - np.mean(S[indmax[kk], :]),
        )  #
    ZZ = GG @ (S - np.reshape(np.mean(S, 1), (r, 1))) + np.reshape(
        np.mean(Out, 1), (r, 1)
    )
    E = Out - ZZ
    MSE = np.linalg.norm(E, "fro") ** 2
    SigPow = np.linalg.norm(ZZ, "fro") ** 2
    SINR = SigPow / MSE
    return SINR, SigPow, MSE, G


########################################################################################
########################################################################################
###                                                                                  ###
###                        SIMULATION                                                ###
###                                                                                  ###
########################################################################################
########################################################################################
pl.figure(figsize=(15, 6), dpi=80)

for iter1 in range(NumAverages):
    iter0 = -1
    for rho in rholist:
        epsv = epsvmax * (1.0 / (1.0 - rho))
        iter0 = iter0 + 1
        calib_correl_matrix = np.eye(r) * (1 - rho) + rho * np.ones((r, r))
        mu = np.zeros(len(calib_correl_matrix))

        #######################################################
        #              GENERATE SOURCES                       #
        #######################################################
        s = chi2.rvs(df, size=n_samples)[:, np.newaxis]
        Z = np.random.multivariate_normal(mu, calib_correl_matrix, n_samples)
        X = np.sqrt(df / s) * Z  # chi-square method
        # Copula-t distriution with df degrees of freedom
        U = t.cdf(X, df).T
        # Generate correlated unity variance sources
        S = U * np.sqrt(12)

        #######################################################
        #              GENERATE MIXINGS                       #
        #######################################################
        # Generate Mxr random mixing from i.i.d N(0,1)
        H = np.random.randn(M, r)
        # Mixtures
        X = np.dot(H, S)
        X = X + NoiseAmp * np.random.randn(X.shape[0], X.shape[1])

        #######################################################
        #             LD InfoMax Algorithm                    #
        #######################################################
        # ALGORTITHM STATE INITIALIZATION
        W = np.eye(r, M) * 2
        W = np.random.randn(r, M)
        delWo = np.zeros((r, M))
        Wu = np.zeros((r, M))
        # SIR list
        SIRlist = np.zeros(NumberofIterations)
        SIRflist = np.zeros(NumberofIterations)
        erlist = np.zeros(N)
        erflist = np.zeros(N)
        Belist = np.zeros(N)
        SIRflisto = 0
        erflisto = 0
        Z = (np.random.rand(r, N)) / 2.0
        RcX = np.dot(X, X.T) / N
        muX = np.mean(X, axis=1).reshape(M, 1)
        RX = RcX - np.dot(muX, muX.T)
        RXinv = np.linalg.pinv(RX + epsv * np.eye(M))
        muv = 1e2
        DISPLAY_PERIOD = 200
        DelZo = np.zeros((r, N))

        # MAIN ALGORITHM LOOP
        for k in range(NumberofIterations):
            if k > 3e3:
                epsv = epsvmax
            RcZ = np.dot(Z, Z.T) / N
            muZ = np.mean(Z, axis=1).reshape(r, 1)
            RZ = RcZ - np.dot(muZ, muZ.T) + epsv * np.eye(r)
            Zbar = Z - muZ
            E = Zbar - np.dot(W, (X - muX))
            RcE = np.dot(E, E.T) / N
            muE = np.mean(E, axis=1).reshape(r, 1)
            RE = RcE - np.dot(muE, muE.T) + epsv * np.eye(r)
            DelZ = (
                np.dot(np.linalg.pinv(RZ), Z - muZ) / N
                - np.dot(np.linalg.pinv(RE), E - muE) / N
            )
            Upd = DelZ - DelZo
            Zu = Z + muv * Upd / np.sqrt(k + 1)
            Z = nonnegativeclipping(Zu)
            if np.mod(k, 1) == 0:
                RcZX = np.dot(Z, X.T) / N
                muZ = np.mean(Z, axis=1).reshape(r, 1)
                RZX = RcZX - np.dot(muZ, muX.T)
                W = np.dot(RZX, RXinv)
            # SIRv,rankP=CalculateSIR(H,W)
            # SIRlist[k]=SIRv
            lambdasirf = 1 - 1 / 10000
        # SIRflist[k]=lambdasirf*SIRflisto+(1-lambdasirf)*10*np.log10(SIRv)
        # if (np.abs(10*np.log10(SIRv)-SIRflisto)>SIRflisto):
        #    SIRflist[k]=10*np.log10(SIRv)
        #   SIRflisto=SIRflist[k];
        # END OF MAIN ALGORITHM LOOP
        Gld = np.dot(
            Z - np.reshape(np.mean(Z, 1), (r, 1)),
            np.linalg.pinv(S - np.reshape(np.mean(S, 1), (r, 1))),
        )
        SIRv, rankP = CalculateSIR(Gld, np.eye(r))
        SIRLDInfoMax[iter0, iter1] = SIRv
        SIRv, rankP = CalculateSIR(H, W)
        SIRLDInfoMaxW[iter0, iter1] = SIRv
        SINRv, SigPow, MSEv, G = CalculateSINR(Z, S)
        SINRLDInfoMax[iter0, iter1] = SINRv
        MSELDInfoMax[iter0, iter1] = MSEv
        SigPowList[iter0, iter1] = SigPow
        #######################################################
        #            ICA InfoMax Algorithm                    #
        #######################################################
        mneinfo = mne.create_info(M, 2000, ch_types=["eeg"] * M)
        ica = mne.preprocessing.ICA(
            n_components=r,
            method="infomax",
            fit_params={"extended": True, "n_subgauss": r, "max_iter": 10000},
            random_state=1,
            verbose=True,
        )
        mneobj = mne.io.RawArray(X, mneinfo)
        ica.fit(mneobj)
        Se = ica.get_sources(mneobj)
        o = Se.get_data()
        SS = S - np.reshape(np.mean(S, 1), (r, 1))
        G = o @ np.linalg.pinv(SS)
        SIRv, rankP = CalculateSIR(G, np.eye(r))
        SIRICAInfoMax[iter0, iter1] = SIRv
        SINRv, SigPow, MSEv, G = CalculateSINR(o, S)
        SINRICAInfoMax[iter0, iter1] = SINRv
        MSEICAInfoMax[iter0, iter1] = MSEv
    SIRS = {
        "SIRLDInfoMax": SIRLDInfoMax,
        "SINRLDInfoMax": SINRLDInfoMax,
        "MSELDInfoMax": MSELDInfoMax,
        "SIRICAInfoMax": SIRICAInfoMax,
        "SINRICAInfoMax": SINRICAInfoMax,
        "MSEICAInfoMax": MSEICAInfoMax,
        "SIRLDInfoMaxW": SIRLDInfoMaxW,
        "SigPowList": SigPowList,
        "M": M,
        "r": r,
        "N": N,
        "NumAverages": NumAverages,
        "NumberofIterations": NumberofIterations,
        "rholist": rholist,
        "iter1": iter1,
    }
    with open(outname, "wb") as handle:
        pickle.dump(SIRS, handle)
