
using Distributed

include("Primitives.jl")
include("AnalysisFunctions.jl")

FOLDER="AllExercises"
FOLDER_GRAPHS="Graphs"
######################################################
######### Load benchmark model #######################
######################################################
case_coulumn=2
MOD, MOD_DL, MOD_DefL, MOD_FR=Unpack_All_Models_Case(case_coulumn,FOLDER)

######################################################
######### Table 2 - Targeted moments #################
######################################################
Tmom=1000; NSamplesMoments=10000
MOM=AverageMomentsManySamples(Tmom,NSamplesMoments,MOD)
MOM.σ_inv/MOM.σ_GDP
MOM.MeanSpreads
MOM.Debt_GDP

######################################################
######### Table 3 - Untargeted moments ###############
######################################################
MOM.DefaultPr
MOM.StdSpreads
MOM.σ_con/MOM.σ_GDP
MOM.σ_GDP
MOM.σ_TB_y
MOM.Corr_Spreads_GDP
MOM.Corr_TB_GDP

######################################################
######### Figure 1 and 3 - Welfare gains #############
######################################################
case_coulumn=2
fig_1, fig_3, fig_1App=All_WelfareGains_Plots(case_coulumn,FOLDER)
fig_1
savefig(fig_1,"$FOLDER_GRAPHS\\Figure_1.pdf")
fig_3
savefig(fig_3,"$FOLDER_GRAPHS\\Figure_3.pdf")
fig_1App
savefig(fig_1App,"$FOLDER_GRAPHS\\Figure_1App.pdf")

wg_dl, dl_star, wg_dr, dl_dr_star, χ_dr_star, wg_bbr, bbr_star=Optimal_Fiscal_Rules(case_coulumn,FOLDER)
wg_dl
wg_dr
wg_bbr

dl_star

dl_dr_star
χ_dr_star

bbr_star

######################################################
###### Figure 2 - Transition to fiscal rule ##########
######################################################
case_coulumn=2
N=20000
Tafter=40
fig_2, fig_2App=Plot_Consolidations_Case_Bav_And_B0(N,Tafter,case_coulumn,FOLDER)
fig_2
savefig(fig_2,"$FOLDER_GRAPHS\\Figure_2.pdf")
fig_2App
savefig(fig_2App,"$FOLDER_GRAPHS\\Figure_2App.pdf")

Tafter=160
fig_2_lr, fig_2App_lr=Plot_Consolidations_Case_Bav_And_B0(N,Tafter,case_coulumn,FOLDER)
fig_2_lr
savefig(fig_2_lr,"$FOLDER_GRAPHS\\Figure_2_lr.pdf")
fig_2App_lr
savefig(fig_2App_lr,"$FOLDER_GRAPHS\\Figure_2App_lr.pdf")

######################################################
###### Figure 4 - Spreads and price of debt ##########
######################################################
fig_4, fig_4App=PlotPriceAndSpreads_Case(case_coulumn,FOLDER)
fig_4
savefig(fig_4,"$FOLDER_GRAPHS\\Figure_4.pdf")
fig_4App
savefig(fig_4App,"$FOLDER_GRAPHS\\Figure_4App.pdf")

######################################################
######### Table 4 - Compare optimal rules ############
######################################################
#Benchmark case is case_column=2, foreign investors is 4, patient hh is 5
case_coulumn=5
MOD, MOD_DL, MOD_DefL, MOD_FR=Unpack_All_Models_Case(case_coulumn,FOLDER)

wg_dl, dl_star, wg_dr, dl_dr_star, χ_dr_star, wg_bbr, bbr_star=Optimal_Fiscal_Rules(case_coulumn,FOLDER)
dl_star
dl_dr_star
χ_dr_star
bbr_star

wg_dl
wg_dr
wg_bbr

MOM=AverageMomentsManySamples(Tmom,NSamplesMoments,MOD)
MOM_DL=AverageMomentsManySamples(Tmom,NSamplesMoments,MOD_DL)
MOM_FR=AverageMomentsManySamples(Tmom,NSamplesMoments,MOD_FR)
MOM_BBR=AverageMomentsManySamples(Tmom,NSamplesMoments,MOD_DefL)

MOM.DefaultPr
MOM_DL.DefaultPr
MOM_FR.DefaultPr
MOM_BBR.DefaultPr

MOM.MeanSpreads
MOM_DL.MeanSpreads
MOM_FR.MeanSpreads
MOM_BBR.MeanSpreads

MOM.StdSpreads
MOM_DL.StdSpreads
MOM_FR.StdSpreads
MOM_BBR.StdSpreads

MOM.Debt_GDP
MOM_DL.Debt_GDP
MOM_FR.Debt_GDP
MOM_BBR.Debt_GDP

MOM.σ_con/MOM.σ_GDP
MOM_DL.σ_con/MOM_DL.σ_GDP
MOM_FR.σ_con/MOM_FR.σ_GDP
MOM_BBR.σ_con/MOM_FR.σ_GDP

T=1000
TS_no_rule=Simulate_Paths_Ergodic(T,MOD)
TS_DL=Simulate_Paths_Ergodic(T,MOD_DL)
TS_FR=Simulate_Paths_Ergodic(T,MOD_FR)
TS_BBR=Simulate_Paths_Ergodic(T,MOD_DefL)
mean(TS_DL.Cons ./ TS_no_rule.Cons)
mean(TS_FR.Cons ./ TS_no_rule.Cons)
mean(TS_BBR.Cons ./ TS_no_rule.Cons)

#############################################################
### Figure 5 - Transition to fiscal rule, foreign capital ###
#############################################################
case_coulumn=4
Tafter=40
fig_5, fig_5App=Plot_Consolidations_Case_Bav_And_B0(N,Tafter,case_coulumn,FOLDER)
fig_5
savefig(fig_5,"$FOLDER_GRAPHS\\Figure_5.pdf")
fig_5App
savefig(fig_5App,"$FOLDER_GRAPHS\\Figure_5App.pdf")

Tafter=160
fig_5_lr, fig_5App_lr=Plot_Consolidations_Case_Bav_And_B0(N,Tafter,case_coulumn,FOLDER)
fig_5_lr
savefig(fig_5_lr,"$FOLDER_GRAPHS\\Figure_5_lr.pdf")
fig_5App_lr
savefig(fig_5App_lr,"$FOLDER_GRAPHS\\Figure_5App_lr.pdf")

case_coulumn=5
Tafter=40
fig_5hh, fig_5hhApp=Plot_Consolidations_Case_Bav_And_B0(N,Tafter,case_coulumn,FOLDER)
fig_5hh
savefig(fig_5hh,"$FOLDER_GRAPHS\\Figure_5hh.pdf")
fig_5hhApp
savefig(fig_5hhApp,"$FOLDER_GRAPHS\\Figure_5hhApp.pdf")

Tafter=160
fig_5hh_lr, fig_5hhApp_lr=Plot_Consolidations_Case_Bav_And_B0(N,Tafter,case_coulumn,FOLDER)
fig_5hh_lr
savefig(fig_5hh_lr,"$FOLDER_GRAPHS\\Figure_5hh_lr.pdf")
fig_5hhApp_lr
savefig(fig_5hhApp_lr,"$FOLDER_GRAPHS\\Figure_5hhApp_lr.pdf")

######################################################
############### Welfare decomposition ################
######################################################
MOD_Benchmark, MOD_Planner, MOD_Covenants, MOD_Efficient=Models_Welfare_Decomposition(FOLDER)

N=10000
#Average welfare gains of first-best
av_wg_fb=AverageWelfareGains(N,MOD_Benchmark,MOD_Efficient)

#Welfare gains of first-best from average state
#for welfare decomposition
wg_Ben_Eff=WelfareGains_AvState(N,MOD_Benchmark,MOD_Benchmark,MOD_Efficient)

#First decomposition
#benchmark -> covenants -> centralized investment
wg_Ben_Cov=WelfareGains_AvState(N,MOD_Benchmark,MOD_Benchmark,MOD_Covenants)
wg_Cov_Eff=WelfareGains_AvState(N,MOD_Benchmark,MOD_Covenants,MOD_Efficient)

#Second decomposition
#benchmark -> centralized investment -> covenants
wg_Ben_Pla=WelfareGains_AvState(N,MOD_Benchmark,MOD_Benchmark,MOD_Planner)
wg_Pla_Eff=WelfareGains_AvState(N,MOD_Benchmark,MOD_Planner,MOD_Efficient)

#Average welfare gains from 0 debt
#Footnote 15
case_coulumn=2
MOD, MOD_DL, MOD_DefL, MOD_FR=Unpack_All_Models_Case(case_coulumn,FOLDER)
wg=AverageWelfareGains_0(N,MOD_Benchmark,MOD_DefL)
wg=AverageWelfareGains_0(N,MOD_Benchmark,MOD_Efficient)
