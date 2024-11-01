
using Distributed
addprocs(39)

@everywhere include("Primitives.jl")

@everywhere XX=readdlm("Setup.csv",',')
@everywhere VEC_PAR=XX[2:end,2]*1.0
@everywhere par, GRIDS=Setup_From_Vector(VEC_PAR)

###############################################################################
#Solve benchmark decentralized
###############################################################################
@everywhere NAME_DEC="SOL_DEC.csv"
Decentralized=true
SolveAndSaveModel(Decentralized,NAME_DEC,GRIDS,par)

###############################################################################
#Try different debt limits
###############################################################################
@everywhere N_FR=39
@everywhere fr_low=0.35
@everywhere fr_high=0.60
@everywhere grFR=collect(range(fr_low,stop=fr_high,length=N_FR))

@everywhere MOD_DEC=UnpackModel_File(NAME_DEC," ")
FR=TestManyFiscalRules_Dec(grFR,MOD_DEC)

@everywhere N_χ=39
@everywhere χ_low=0.00
@everywhere χ_high=0.20
@everywhere grχ=collect(range(χ_low,stop=χ_high,length=N_χ))
TestManyTransitionSmoothers_Dec(FR,grχ,MOD_DEC)
