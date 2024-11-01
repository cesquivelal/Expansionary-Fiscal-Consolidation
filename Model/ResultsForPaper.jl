
using Distributed, BenchmarkTools
using Plots; pyplot(fontfamily="serif",linewidth=2.0,grid=false,legend=true,
                    background_color_legend=nothing,foreground_color_legend=nothing,
                    legendfontsize=20,guidefontsize=20,titlefontsize=20,tickfontsize=20,
                    markersize=9,size=(650,500))

################################################################
######## Load all functions and files with solutions ###########
################################################################
#Load functions
include("Primitives.jl")
include("AnalysisFunctions.jl")

#Load benchmark model
FOLDER="AllExercises"
NAME_DEC="SOL_DEC.csv"
MOD=UnpackModel_File(NAME_DEC,FOLDER)

#Load benchmark model with debt limit and consolidation plan
FOLDER_FILE_WG="$FOLDER\\WelfareGains_DEC.csv"
FOLDER_FILE_WG_TR="$FOLDER\\WelfareGainsTransition_DEC.csv"

ARR_SOLS_DEC=readdlm("$FOLDER\\ALL_SOLS_DEC_FR_0.csv",',')
ARR_SOLS_DEC_TR=readdlm("$FOLDER\\ALL_SOLS_DEC_FR_CHI.csv",',')

XX=readdlm(FOLDER_FILE_WG,',')
GAINS=1.0*XX[2:end,2]
wg, i_fr=findmax(GAINS[:,1])
MOD_FR=UnpackModel_Vector(ARR_SOLS_DEC[:,i_fr])

XX=readdlm(FOLDER_FILE_WG_TR,',')
GAINS=1.0*XX[2:end,2]
wg, i_χ=findmax(GAINS[:,1])
MOD_FR_TR=UnpackModel_Vector(ARR_SOLS_DEC_TR[:,i_χ])

#Load models with no commitment, lax and strict
FOLDER_GRIM="GrimStrategy"
NAME_GRIM="SOL_GRIM_Lax.csv"
MOD_GRIM_Lax=UnpackModel_File(NAME_GRIM,FOLDER_GRIM)

NAME_GRIM="SOL_GRIM_Strict.csv"
MOD_GRIM_Strict=UnpackModel_File(NAME_GRIM,FOLDER_GRIM)

################################################################
################# Abstract: long-run changes ###################
################################################################
ErgodicStart=true; by0=0.46
NoDefaults=false; ForSlides=false
N=20000; Tbefore=4; Tafter=100
TS_Av, TS_FR_Av=Average_Consolidation_Paths(NoDefaults,ErgodicStart,by0,N,Tbefore,Tafter,MOD,MOD_FR)
100*((minimum(TS_FR_Av.Cons)/TS_FR_Av.Cons[1])-1)
100*((minimum(TS_FR_Av.Inv)/TS_FR_Av.Inv[1])-1)
100*((TS_FR_Av.GDP[end]/TS_FR_Av.GDP[1])-1)

################################################################
################# Table 2: Targeted moments ####################
################################################################
Tmom=1000; NSamplesMoments=100
MOM=AverageMomentsManySamples(Tmom,NSamplesMoments,MOD)
MOM.σ_inv/MOM.σ_GDP
MOM.MeanSpreads
MOM.Debt_GDP

################################################################
################ Table 3: Non-targeted moments #################
################################################################
MOM.DefaultPr
MOM.StdSpreads
MOM.σ_con/MOM.σ_GDP
MOM.σ_GDP
MOM.σ_TB_y
MOM.Corr_Spreads_GDP
MOM.Corr_TB_GDP

################################################################
################ Figure 1: Optimal debt limit ##################
################################################################
DeficitLimit=false
plt=Plot_WelfareGains(DeficitLimit,FOLDER_FILE_WG)
savefig(plt,"Graphs\\WG_DebtLimit.pdf")

################################################################
####### Figure 2: Expansionary fiscal consolidation ############
################################################################
ErgodicStart=true; by0=0.46
NoDefaults=false; ForSlides=false
N=20000; Tbefore=4; Tafter=20
pltEFC=Plot_OneConsolidationPath(ForSlides,NoDefaults,ErgodicStart,by0,N,Tbefore,Tafter,MOD,MOD_FR)
if ForSlides
    NAME="Graphs\\EFC_slides.pdf"
else
    NAME="Graphs\\EFC.pdf"
end
savefig(pltEFC,NAME)

################################################################
########## Figure 3: Spreads and the price of debt #############
################################################################
plt=PlotPriceAndSpreads(N,Tbefore,Tafter,MOD,MOD_FR)
NAME="Graphs\\PriceSchedule.pdf"
savefig(plt,NAME)

################################################################
############# Figure 4: Optimal consolidation plan #############
################################################################
DeficitLimit=true
plt=Plot_WelfareGains(DeficitLimit,FOLDER_FILE_WG_TR)
savefig(plt,"Graphs\\WG_DeficitLimitTransition.pdf")

################################################################
#### Figure 4: Expansionary fiscal consolidation, long-run #####
################################################################
Tafter=80
pltEFC=Plot_OneConsolidationPath(ForSlides,NoDefaults,ErgodicStart,by0,N,Tbefore,Tafter,MOD,MOD_FR)
if ForSlides
    NAME="Graphs\\EFC_LR_slides.pdf"
else
    NAME="Graphs\\EFC_LR.pdf"
end
savefig(pltEFC,NAME)

################################################################
############## Table 4: Moments different cases ################
################################################################
MOM_FR_TR=AverageMomentsManySamples(Tmom,NSamplesMoments,MOD_FR_TR)
MOM_GRIM_Strict=AverageMomentsManySamples(Tmom,NSamplesMoments,MOD_GRIM_Strict)

MOM.DefaultPr
MOM_FR_TR.DefaultPr
MOM_GRIM_Strict.DefaultPr

MOM.MeanSpreads
MOM_FR_TR.MeanSpreads
MOM_GRIM_Strict.MeanSpreads

MOM.StdSpreads
MOM_FR_TR.StdSpreads
MOM_GRIM_Strict.StdSpreads

MOM.Debt_GDP
MOM_FR_TR.Debt_GDP
MOM_GRIM_Strict.Debt_GDP

T=20000
TS_BEN=Simulate_Paths_Ergodic(T,MOD)
TS_FR_TR=Simulate_Paths_Ergodic(T,MOD_FR_TR)
TS_GRIM_STR=Simulate_Paths_Ergodic(T,MOD_GRIM_Strict)

100*(mean(TS_FR_TR.Cons ./ TS_BEN.Cons)-1)

μc_ben=mean(TS_BEN.Cons)
μc_fr_tr=mean(TS_FR_TR.Cons)
μc_str=mean(TS_GRIM_STR.Cons)

μc_fr_tr/μc_ben
μc_str/μc_ben

MOM.σ_con/MOM.σ_GDP
MOM_FR_TR.σ_con/MOM_FR_TR.σ_GDP
MOM_GRIM_Strict.σ_con/MOM_GRIM_Strict.σ_GDP

ZeroDebt=false; Nwg=10000
wg_ben=AverageWelfareGains(ZeroDebt,Nwg,MOD,MOD_FR_TR)
wg_str=AverageWelfareGains(ZeroDebt,Nwg,MOD,MOD_GRIM_Strict)

################################################################
#### Figure 6: Expansionary fiscal consolidation, with plan ####
################################################################
Tafter=20; WithCommitment=true
pltEFC=Plot_TwoConsolidationPaths(WithCommitment,ForSlides,ErgodicStart,by0,N,Tbefore,Tafter,MOD,MOD_FR,MOD_FR_TR)
if ForSlides
    NAME="Graphs\\EFC_OptTran_slides.pdf"
else
    NAME="Graphs\\EFC_OptTran.pdf"
end
savefig(pltEFC,NAME)



################################################################
### Figure 7: Expansionary fiscal consolidation, strict grim ###
################################################################
WithCommitment=false
pltEFC=Plot_TwoConsolidationPaths(WithCommitment,ForSlides,ErgodicStart,by0,N,Tbefore,Tafter,MOD,MOD_FR_TR,MOD_GRIM_Strict)
if ForSlides
    NAME="Graphs\\EFC_NoCommitment_slides.pdf"
else
    NAME="Graphs\\EFC_NoCommitment.pdf"
end
savefig(pltEFC,NAME)
