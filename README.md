# Expansionary-Fiscal-Rules

Replication code for "Expansionary Fiscal Rules Under Sovereign Risk" by Carlos Esquivel and Agustín Sámano

August, 2025:
https://cesquivelal.github.io/ES_ExFisCon.pdf

# Data

The folder Data contains the panel data used in Section 4 in the file fiscal_rules_macro_variables_2000_2019_readytouse.dta and STATA code for all regression exercises in the file Empirical_Analysis_Draft_October.do.

# Quantitative solution of model

The code is written in the Julia language, version 1.7.2 and uses the following packages:
      Distributed, Parameters, Interpolations, Optim, SharedArrays, DelimitedFiles,
      Distributions, FastGaussQuadrature, LinearAlgebra, Random, Statistics,
      SparseArrays, QuadGK, Sobol, Roots, Plots

The file Primitives.jl defines all objects and functions that are used to solve and simulate the model.

The file mainAll.jl in the folder Model uses the file Primitives.jl to solve for the benchmark model with no fiscal rule for the model with different debt limits, and for the model with the optimal debt limit and different deficit limits. All solutions are stored in separate .csv files. The file Setup.csv defines the main parameters.

The file ResultsForPaper.jl uses the solutions to produce all the model results reported in Section 3 of the paper.
