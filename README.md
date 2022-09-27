# Correlative Information Maximization Based Biologically Plausible Neural Networks for Correlated Source Separation

This Github repository includes the implementation of the Correlative Information Maximization Based Biologically Plausible Neural Networks, which is submitted to ICLR 2023. All the codes are written in Python 3.

## CorInfoMax Neural Networks

CorInfoMax NN for Canonical Polytope Representation  |  CorInfoMax NN for "Feature-Based" Polytope Representation
:-------------------------:|:-------------------------:
![Sample Network Figures](./Figures/CorInfoMaxHRep3.png)   |  ![Sample Network Figures](./Figures/CorInfoMaxGen.png)

## Simulations In the Paper

All the simulation codes for the paper manuscript are included inside the folder "Simulations". The subfolders are named accordingly, e.g., "SparseNoisy" folder contains the experiments for the sparse source separation simulations in the paper. The jupyter notebooks inside the folder "Simulations/AnalyzeSimulationResultsFinal" illustrates the plots and tables presented in the paper. For example, the notebook "PlotSimulationResults_GeneralPoly.ipynb" includes the plots for the Appendix D.2.4. To replicate the figures in this specific notebook, you need to follow the below steps,

 * Run both python scripts in the folder "Simulations/GeneralPolytope" with the following commands:

    ``` python CorInfoMax_GeneralPolytope5dimV2.py```

    ``` python CorInfoMax_GeneralPolytope5dimV3.py```

 * When you run these python simulations, the following two pickle files will be created which contains the SINR results of each algorithm. 
    * "Simulations/Results/simulation_results_general_polytope_5dimV2.pkl"
    * "Simulations/Results/simulation_results_general_polytope_5dimV3.pkl"

 * The jupyter notebook "Simulations/AnalyzeSimulationResultsFinal/PlotSimulationResults_GeneralPoly.ipynb" reads the above pickle files and visualize the results. Moreover, the performances of the baseline algorithms are also reported.

To replicate each simulation in the paper, you can adapt the above procedure for the other sections (for antisparse, sparse, and simplex BSS simulations). The experiment procedure for video separation is included in "Simulations/VideoSeparation" as a separate readme file. The sparse dictionary learning experiment is located in "Simulations/SparseDictionaryLearning", and the notebook inside this folder produces the sparse receptive fields from prewhitened Olshaussen's image pathces (you need "imagepatcheselfwhitened.mat" file whereas it is not included here due to its file size).

# Python Version and Dependencies

* Python Version: Python 3.8.12

* pip version : 21.2.4

* Required Python Packages: Specified in requirements.txt file.

* Platform Info : "OS: Linux (x86_64-pc-linux-gnu) CPU: Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz"