
using Distributed
addprocs(62)

@everywhere include("Primitives.jl")

###############################################################################
#Try different debt limits, benchmark
###############################################################################
@everywhere SETUP_FILE="Setup.csv"
DoPairs=true; IsFirst=true
TestAll_DebtLimits_And_FiscalRules(DoPairs,IsFirst,2,SETUP_FILE)
IsFirst=false
TestAll_DebtLimits_And_FiscalRules(DoPairs,IsFirst,3,SETUP_FILE)
TestAll_DebtLimits_And_FiscalRules(DoPairs,IsFirst,4,SETUP_FILE)
TestAll_DebtLimits_And_FiscalRules(DoPairs,IsFirst,5,SETUP_FILE)
TestAll_DebtLimits_And_FiscalRules(DoPairs,IsFirst,6,SETUP_FILE)
