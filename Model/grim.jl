
using Distributed

include("Primitives.jl")

XX=readdlm("Setup.csv",',')
VEC_PAR=XX[2:end,2]*1.0
par, GRIDS=Setup_From_Vector(VEC_PAR)

###############################################################################
#Solve benchmark decentralized
###############################################################################
parFR=Pars(par,WithFR=true,FR=0.4421052631578947,χ=0.015789473684210527)

RuleAfterDefault=false
NAME_GRIM="SOL_GRIM_Strict.csv"
SolveAndSaveModel_GRIM(RuleAfterDefault,NAME_GRIM,GRIDS,par,parFR)
