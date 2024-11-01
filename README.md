# Expansionary-Fiscal-Consolidation

Replication code for "Expansionary Fiscal Consolidation Under Sovereign Risk" by Carlos Esquivel and Agustín Sámano

October, 2024:
https://cesquivelal.github.io/ES_ExFisCon.pdf

# Data

The folder Data contains the panel data used in Section 4 in the file fiscal_rules_macro_variables_2000_2019_readytouse.dta and STATA code for all regression exercises in the file Empirical_Analysis_Draft_October.do.

# Quantitative solution of model

The code is written in the Julia language, version 1.7.2 and uses the following packages:
      Distributed, Parameters, Interpolations, Optim, SharedArrays, DelimitedFiles,
      Distributions, FastGaussQuadrature, LinearAlgebra, Random, Statistics,
      SparseArrays, QuadGK, Sobol, Roots, Plots

The file Primitives.jl defines all objects and functions that are used to solve and simulate the model.

The file mainSolve.jl in the folder DutchDisease uses the file Primitives.jl to solve for the planner's and decentralized equilibria in the benchmark model, in the case with low elasticity of substitution and in the case for the benchmark with no commodities. These solutions are stored in separate .csv files. The file Setup.csv defines parameters used in each solution.

The file mainSolve.jl in the folder SolveWithSubsidyRule solves for the decentralized equilibrium with the fitted simple subsidy rules using the file Primitives.jl. It also uses the benchmark solution Decentralized.csv and stores the result in the file Decentralized_rules.csv.

The file ResultsForPaper.jl uses the solutions stored in the folders DutchDisease and SolveWithSubsidyRule to produce all the results reported in Section 3 of the paper.
