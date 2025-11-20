
using Plots; pythonplot(fontfamily="serif",linewidth=2.0,grid=false,legend=true,
                        background_color_legend=nothing,foreground_color_legend=nothing,
                        legendfontsize=20,guidefontsize=20,titlefontsize=20,tickfontsize=20,
                        markersize=9,size=(650,500))


###############################################################################
### Welfare gains analysis
###############################################################################
function Unpack_wg_DL(IsDebtLimit::Bool,case_coulumn::Int64,FOLDER::String)
    if IsDebtLimit
        FILE="WelfareGains_DL.csv"
    else
        FILE="WelfareGains_DefL.csv"
    end

    if FOLDER==" "
        VEC_0=readdlm(FILE,',')
    else
        VEC_0=readdlm("$FOLDER\\$FILE",',')
    end
    #case_column corresponds to Setup file
    VEC=1.0*VEC_0[2:end,case_coulumn-1]
    N_dl=convert(Int64,VEC[1])
    grDL=VEC[2:2+N_dl-1]
    wg_vec=VEC[2+N_dl:2+N_dl+N_dl-1]
    def_pr_vec=VEC[2+N_dl+N_dl:end]

    return grDL, wg_vec, def_pr_vec
end

function Unpack_wg_pair(NAME_CASE::String,FOLDER::String)
    NAME="WelfareGains_Pairs_$NAME_CASE.csv"

    if FOLDER==" "
        VEC=readdlm(NAME,',')
    else
        VEC=readdlm("$FOLDER\\$NAME",',')
    end
    N_χ=convert(Int64,VEC[1])
    N_dl=convert(Int64,VEC[2])
    grχ=VEC[3:3+N_χ-1]
    grDL=VEC[3+N_χ:3+N_χ+N_dl-1]
    wg_vec=VEC[3+N_χ+N_dl:end]
    MAT_WG=reshape(wg_vec,(N_dl,N_χ))

    return grχ, grDL, MAT_WG
end

function Plot_WelfareGains(IsDebtLimit::Bool,case_coulumn::Int64,FOLDER::String)
    #case_column corresponds to Setup file
    grDL, wg_vec, def_pr_vec=Unpack_wg_DL(IsDebtLimit,case_coulumn,FOLDER)

    zz=zeros(Float64,length(wg_vec))
    COLORS=[:green :black]
    LINESTYLES=[:solid :solid]

    size_width=650
    size_height=500
    SIZE=(size_width,size_height)

    #First plot wg alone
    TITLE="welfare gains"
    YLABEL="c.e. units, percentage points"
    if IsDebtLimit
        XLABEL="limit to debt-to-GDP ratio"
    else
        XLABEL="limit to primary deficit (relative to GDP)"
    end
    plt_wg=plot(grDL,[wg_vec zz],
                ylabel=YLABEL,
                xlabel=XLABEL,
                legend=false,
                linecolor=COLORS,
                title=TITLE,
                SIZE=(size_width,size_height),
                linestyle=LINESTYLES)

    #Plot default probability vs wg
    TITLE="default probability"
    YLABEL="percentage"
    plt_dp=plot(grDL,def_pr_vec,
                ylabel=YLABEL,
                xlabel=XLABEL,
                legend=false,
                linecolor=COLORS,
                title=TITLE,
                SIZE=(size_width,size_height),
                linestyle=LINESTYLES)

    l1 = @layout([a b])
    plt=plot(plt_wg,plt_dp,
              layout=l1,size=(size_width*2,size_height*1))
    return plt
end

function Plot_WelfareGains_Pairs(NAME_CASE::String,FOLDER::String)
    grχ, grDL, MAT_WG=Unpack_wg_pair(NAME_CASE,FOLDER)

    #Meshgrids
    mesh_χ = repeat(grχ', length(grDL), 1)*100
    mesh_DL = repeat(grDL, 1, length(grχ))*100

    #Labels
    XLABEL="deficit limit"
    YLABEL="debt limit"
    ZLABEL="welfare gains, c.e. units"

    plt=surface(mesh_χ,mesh_DL,MAT_WG,
                colorbar=false)
        annotate!(-0.0,45.0,-1.4,text("$XLABEL",:black,17))
        annotate!(8.0,45.0,-0.65,text("$YLABEL",:black,17))
        annotate!(4.0,60.0,0.55,text("$ZLABEL",:black,17))

    return plt
end

function All_WelfareGains_Plots(case_coulumn::Int64,FOLDER::String)
    #case_column corresponds to Setup file
    SETUP_FILE="$FOLDER\\Setup.csv"
    XX=readdlm(SETUP_FILE,',')
    NAME_CASE=convert(String,XX[1,case_coulumn])

    IsDebtLimit=true
    plt_dl=Plot_WelfareGains(IsDebtLimit,case_coulumn,FOLDER)
    IsDebtLimit=false
    plt_defl=Plot_WelfareGains(IsDebtLimit,case_coulumn,FOLDER)
    plt_fr=Plot_WelfareGains_Pairs(NAME_CASE,FOLDER)

    return plt_dl, plt_fr, plt_defl
end

function Optimal_Fiscal_Rules(case_coulumn::Int64,FOLDER::String)
    grDL, wg_vec, def_pr_vec=Unpack_wg_DL(true,case_coulumn,FOLDER)
    grDefL, wg_DefL_vec, def_pr_DefL_vec=Unpack_wg_DL(false,case_coulumn,FOLDER)

    #case_column corresponds to Setup file
    SETUP_FILE="$FOLDER\\Setup.csv"
    XX=readdlm(SETUP_FILE,',')
    NAME_CASE=convert(String,XX[1,case_coulumn])
    grχ, grDL, MAT_WG=Unpack_wg_pair(NAME_CASE,FOLDER)

    #Debt limit
    wg_dl, i_dl=findmax(wg_vec)
    dl_star=grDL[i_dl]

    #Fiscal rule with consolidation plan
    wg_dr, I_dr=findmax(MAT_WG)
    (i_dl_dr,i_χ)=Tuple(I_dr)
    dl_dr_star=grDL[i_dl_dr]
    χ_dr_star=grχ[i_χ]

    #Budget balanced rule
    wg_bbr, i_bbr=findmax(wg_DefL_vec)
    bbr_star=grDefL[i_bbr]

    return wg_dl, dl_star, wg_dr, dl_dr_star, χ_dr_star, wg_bbr, bbr_star
end

###############################################################################
### Average paths after introducing fiscal rules
###############################################################################
function Allocate_TSi_to_TS!(TS::Paths,t0::Int64,t1::Int64,TSi::Paths,ti0::Int64,ti1::Int64)
    #Last element of TS0 is equal to first element of TS1
    #Paths of shocks
    TS.z[t0:t1] .= TSi.z[ti0:ti1]
    TS.zP[t0:t1] .= TSi.zP[ti0:ti1]
    TS.Readmission[t0:t1] .= TSi.Readmission[ti0:ti1]

    #Paths of chosen states
    TS.Def[t0:t1] .= TSi.Def[ti0:ti1]
    TS.K[t0:t1] .= TSi.K[ti0:ti1]
    TS.B[t0:t1] .= TSi.B[ti0:ti1]

    #Path of relevant variables
    TS.Spreads[t0:t1] .= TSi.Spreads[ti0:ti1]
    TS.GDP[t0:t1] .= TSi.GDP[ti0:ti1]
    TS.Cons[t0:t1] .= TSi.Cons[ti0:ti1]
    TS.Inv[t0:t1] .= TSi.Inv[ti0:ti1]
    TS.AdjCost[t0:t1] .= TSi.AdjCost[ti0:ti1]
    TS.TB[t0:t1] .= TSi.TB[ti0:ti1]
    TS.CA[t0:t1] .= TSi.CA[ti0:ti1]

    return nothing
end

function Concatenate_TSpaths!(TS_FR::Paths,TS_before::Paths,TS_after::Paths)
    #The last element in TS_before is the same as in TS_after
    l0=length(TS_before.z)
    l1=length(TS_after.z)
    T=length(TS_FR.z)

    Allocate_TSi_to_TS!(TS_FR,1,l0-1,TS_before,1,l0-1)
    Allocate_TSi_to_TS!(TS_FR,l0,T,TS_after,1,l1)

    return nothing
end

function FiscalConsolidation_OneTS!(i_starting::Int64,TS_starting::Paths,
                                    TS_FR::Paths,TS_before::Paths,TS_after::Paths,
                                    MOD::Model,MOD_FR::Model)

    #Get initial long-run starting point
    Def0=TS_starting.Def[i_starting]; z0=TS_starting.z[i_starting]
    K0=TS_starting.K[i_starting]; B0=TS_starting.B[i_starting]

    #Use same shocks for both
    Fill_Path_Simulation!(TS_before,Def0,z0,K0,B0,MOD)
    Fill_Path_Simulation!(TS_after,TS_before.Def[end],TS_before.z[end],TS_before.K[end],TS_before.B[end],MOD_FR)
    Concatenate_TSpaths!(TS_FR,TS_before,TS_after)

    return nothing
end

function Add_to_AvPaths!(PATHS_AV::Paths,PATH_i::Paths)
    N=1
    #Paths of shocks
    PATHS_AV.z .= PATHS_AV.z .+ (PATH_i.z ./ N)
    PATHS_AV.zP .= PATHS_AV.zP .+ (PATH_i.zP ./ N)
    PATHS_AV.Readmission .= PATHS_AV.Readmission .+ (PATH_i.Readmission ./ N)

    #Paths of chosen states
    PATHS_AV.Def .= PATHS_AV.Def .+ (PATH_i.Def ./ N)
    PATHS_AV.K .= PATHS_AV.K .+ (PATH_i.K ./ N)
    PATHS_AV.B .= PATHS_AV.B .+ (PATH_i.B ./ N)

    #Path of relevant variables
        #use path without default for spreads
        if sum(PATH_i.Def)==0.0
            PATHS_AV.Spreads .= PATHS_AV.Spreads .+ (PATH_i.Spreads ./ N)
        end
    PATHS_AV.GDP .= PATHS_AV.GDP .+ (PATH_i.GDP ./ N)
    PATHS_AV.Cons .= PATHS_AV.Cons .+ (PATH_i.Cons ./ N)
    PATHS_AV.Inv .= PATHS_AV.Inv .+ (PATH_i.Inv ./ N)
    PATHS_AV.AdjCost .= PATHS_AV.AdjCost .+ (PATH_i.AdjCost ./ N)
    PATHS_AV.TB .= PATHS_AV.TB .+ (PATH_i.TB ./ N)
    PATHS_AV.CA .= PATHS_AV.CA .+ (PATH_i.CA ./ N)

    return nothing
end

function Divide_SumPaths_By_Nsample!(PATHS_AV::Paths,N::Int64,N_noDef::Int64)
    #Paths of shocks
    PATHS_AV.z .= PATHS_AV.z ./ N
    PATHS_AV.zP .= PATHS_AV.zP ./ N
    PATHS_AV.Readmission .= PATHS_AV.Readmission ./ N

    #Paths of chosen states
    PATHS_AV.Def .= PATHS_AV.Def ./ N
    PATHS_AV.K .= PATHS_AV.K ./ N
    PATHS_AV.B .= PATHS_AV.B ./ N

    #Path of relevant variables
        #use path without default for spreads
    PATHS_AV.Spreads .= PATHS_AV.Spreads ./ N_noDef
    PATHS_AV.GDP .= PATHS_AV.GDP ./ N
    PATHS_AV.Cons .= PATHS_AV.Cons ./ N
    PATHS_AV.Inv .= PATHS_AV.Inv ./ N
    PATHS_AV.AdjCost .= PATHS_AV.AdjCost ./ N
    PATHS_AV.TB .= PATHS_AV.TB ./ N
    PATHS_AV.CA .= PATHS_AV.CA ./ N

    return nothing
end

function Average_Consolidation_Paths(B0_zero::Bool,N::Int64,Tbefore::Int64,Tafter::Int64,MOD::Model,MOD_FR::Model)
    #Allocate TS for averages
    T=Tbefore+Tafter+1

    TS_FR_av=InitiateEmptyPaths(T)
    TS_FR_i=InitiateEmptyPaths(T)
    TS_before=InitiateEmptyPaths(Tbefore+1)
    TS_after=InitiateEmptyPaths(Tafter+1)

    #Get sample of initial long-run starting points
    Random.seed!(1234)
    drp=500000#MOD.par.drp
    TS_starting=InitiateEmptyPaths(drp+N)
    Def00=0.0; z00=1.0; K00=0.5*(MOD.par.klow+MOD.par.khigh); B00=0.0
    Fill_Path_Simulation!(TS_starting,Def00,z00,K00,B00,MOD)

    N_noDef=N
    for i in 1:N
        println(i)
        i_starting=drp+i
        if B0_zero
            TS_starting.B[i_starting]=0.0
        end
        FiscalConsolidation_OneTS!(i_starting,TS_starting,TS_FR_i,TS_before,TS_after,MOD,MOD_FR)
        Add_to_AvPaths!(TS_FR_av,TS_FR_i)
        if sum(TS_FR_i.Def)>0.0
            N_noDef=N_noDef-1
        end
    end
    Divide_SumPaths_By_Nsample!(TS_FR_av,N,max(1,N_noDef))
    return TS_FR_av
end

function Unpack_All_Models_Case(case_coulumn::Int64,FOLDER::String)
    SETUP_FILE="$FOLDER\\Setup.csv"
    XX=readdlm(SETUP_FILE,',')
    NAME_CASE=convert(String,XX[1,case_coulumn])

    MAT=readdlm("$FOLDER\\Models $NAME_CASE.csv",',')
    MOD0=UnpackModel_Vector(MAT[:,1])
    MOD_DL=UnpackModel_Vector(MAT[:,2])
    MOD_DefL=UnpackModel_Vector(MAT[:,3])
    (nn, Nmod)=size(MAT)
    if Nmod>3
        MOD_FRpair=UnpackModel_Vector(MAT[:,4])
    else
        MOD_FRpair=deepcopy(MOD0)
    end

    return MOD0, MOD_DL, MOD_DefL, MOD_FRpair
end

function Plot_OneConsolidationPath(ForSlides::Bool,Tbefore::Int64,Tafter::Int64,TS_FR::Paths,parFR::Pars)
    xx=collect(range(-Tbefore,stop=Tafter,length=Tbefore+Tafter+1)) ./ 4
    t0=1#Tbefore

    #Common parameters for plots
    size_width=650
    size_height=500
    SIZE=(size_width,size_height)
    XLABEL="years after implementation"
    T=length(TS_FR.z)
    FRrnd=round(parFR.FR,digits=2)
    χrnd=round(parFR.χ,digits=3)

    #(1) Productivity shocks
    TITLE="effective productivity"
    YLABEL="z, zD"
    COLORS=[:blue :red]
    LINESTYLES=[:solid :dash]
    LABEL=["realized" "effective"]
    yy=TS_FR.z
    yyP=TS_FR.zP
    plt_z=plot(xx,[yy yyP],title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,label=LABEL,
               size=SIZE,ylabel=YLABEL,legend=:topright,ylims=[0.98,1.01])

    #(2) Spreads
    TITLE="spreads"
    YLABEL="percentage points"
    yy=TS_FR.Spreads
    plt_r=plot(xx,yy,title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false)

    #(3) Capital
    TITLE="capital"
    YLABEL="k"
    yy=TS_FR.K
    plt_k=plot(xx,yy,title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false)

    #(4) Debt
    TITLE="debt"
    YLABEL="percentage of GDP"
    yy=100*TS_FR.B ./ (4 .* TS_FR.GDP)
    fr=parFR.FR
    DL=100*fr*ones(length(yy))
    COLORS=[:blue :black]
    LINESTYLES=[:solid :solid]
    LABEL=["b/(4*y)" "limit"]
    plt_b=plot(xx,[yy DL],title=TITLE,xlabel=XLABEL,label=LABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=:topright)

    #(5) Default state
    TITLE="default state"
    YLABEL="share in default"
    yy=TS_FR.Def
    plt_d=plot(xx,yy,title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false)

    #(6) GDP
    TITLE="GDP"
    YLABEL="percentage change"
    yy=100*((TS_FR.GDP ./ TS_FR.GDP[t0]) .- 1)
    plt_y=plot(xx,yy,title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false)

    #(7) Consumption
    TITLE="consumption"
    YLABEL="percentage change"
    YLIMSci=[-8.5,2.5]
    yy=100*((TS_FR.Cons ./ TS_FR.Cons[t0]) .- 1)
    plt_c=plot(xx,yy,title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false,ylims=YLIMSci)

    #(8) Investment
    TITLE="investment"
    YLABEL="percentage change"
    yy=100*((TS_FR.Inv ./ TS_FR.Inv[t0]) .- 1)
    plt_i=plot(xx,yy,title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false,ylims=YLIMSci)

    #(9) Trade balance
    TITLE="trade balance"
    YLABEL="percentage of (annual) GDP"
    yy=100*TS_FR.TB ./ (4*TS_FR.GDP)
    plt_tb=plot(xx,yy,title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false)

    #(10) Current account
    TITLE="current account"
    YLABEL="percentage of GDP"
    yy=100*TS_FR.CA ./ TS_FR.GDP
    plt_ca=plot(xx,yy,title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false)

    #(11) B/K
    TITLE="B/K ratio"
    YLABEL="B/K"
    yy=TS_FR.B ./ TS_FR.K
    plt_bk=plot(xx,yy,title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false)

    #Make the plot arrays
    if ForSlides
        l1 = @layout([a b c d; e f g h])
        plt1=plot(plt_b,plt_k,plt_d,plt_tb,
                  plt_r,plt_y,plt_c,plt_i,
                  layout=l1,size=(size_width*4,size_height*2))
        # savefig(plt,"Graphs\\GrimGainsDuration.pdf")
        return plt1
    else
        l1 = @layout([a b c; d e f; g h i])
        plt1=plot(plt_r,plt_d,plt_z,
                  plt_b,plt_c,plt_i,
                  plt_k,plt_y,plt_tb,layout=l1,size=(size_width*3,size_height*3))
        # savefig(plt,"Graphs\\GrimGainsDuration.pdf")
        return plt1
    end
end

function Plot_TwoConsolidationPaths(WithCommitment::Bool,ForSlides::Bool,Tbefore::Int64,Tafter::Int64,TS_FR_1::Paths,TS_FR_2::Paths,parFR::Pars,parFR_pair::Pars)
    xx=collect(range(-Tbefore,stop=Tafter,length=Tbefore+Tafter+1)) ./ 4
    t0=1#Tbefore

    #Common parameters for plots
    size_width=650
    size_height=500
    SIZE=(size_width,size_height)
    XLABEL="years after implementation"
    T=length(TS_FR_1.z)
    FRrnd=round(parFR.FR,digits=2)
    FRrnd_pair=round(parFR_pair.FR,digits=2)
    χrnd=round(parFR_pair.χ,digits=3)
    COLORS=[:blue :red :black :black]
    LINESTYLES=[:solid :dash :solid :dashdot]

    #(1) Productivity shocks
    TITLE="effective productivity"
    COLORz=[:black :blue :red]
    LINESTYLEz=[:solid :solid :dash]
    YLABEL="z, zD"
    LABELz=["realized" "" ""]
    yy=TS_FR_1.z
    yyP1=TS_FR_1.zP
    yyP2=TS_FR_2.zP
    plt_z=plot(xx,[yy yyP1 yyP2],title=TITLE,xlabel=XLABEL,
               linecolor=COLORz,linestyle=LINESTYLEz,label=LABELz,
               size=SIZE,ylabel=YLABEL,legend=:topright,ylims=[0.98,1.01])

    #(2) Spreads
    TITLE="spreads"
    YLABEL="percentage points"
    if WithCommitment
        LABEL=["debt limit" "debt limit with deficit limit"]
    else
        LABEL=["with commitment" "without commitment"]
    end
    yy1=TS_FR_1.Spreads
    yy2=TS_FR_2.Spreads
    plt_r=plot(xx,[yy1 yy2],title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,label=LABEL,
               size=SIZE,ylabel=YLABEL,legend=:topright)

    #(3) Capital
    TITLE="capital"
    YLABEL="k"
    yy1=TS_FR_1.K
    yy2=TS_FR_2.K
    plt_k=plot(xx,[yy1 yy2],title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false)

    #(4) Debt
    TITLE="debt"
    YLABEL="percentage of GDP"
    yy1=100*TS_FR_1.B ./ (4 .* TS_FR_1.GDP)
    yy2=100*TS_FR_2.B ./ (4 .* TS_FR_2.GDP)
    fr=parFR.FR
    fr_pair=parFR_pair.FR
    DL=100*fr*ones(length(yy))
    DL_pair=100*fr_pair*ones(length(yy))
    LABELb=["" "" "limit" "limit with deficit limit"]
    plt_b=plot(xx,[yy1 yy2 DL DL_pair],title=TITLE,xlabel=XLABEL,label=LABELb,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=:bottomright)

    #(5) Default state
    TITLE="default state"
    YLABEL="share in default"
    yy1=TS_FR_1.Def
    yy2=TS_FR_2.Def
    plt_d=plot(xx,[yy1 yy2],title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false)

    #(6) GDP
    TITLE="GDP"
    YLABEL="percentage change"
    yy1=100*((TS_FR_1.GDP ./ TS_FR_1.GDP[t0]) .- 1)
    yy2=100*((TS_FR_2.GDP ./ TS_FR_2.GDP[t0]) .- 1)
    plt_y=plot(xx,[yy1 yy2],title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false)

    #(7) Consumption
    TITLE="consumption"
    YLABEL="percentage change"
    YLIMSci=[-8.5,2.5]
    yy1=100*((TS_FR_1.Cons ./ TS_FR_1.Cons[t0]) .- 1)
    yy2=100*((TS_FR_2.Cons ./ TS_FR_2.Cons[t0]) .- 1)
    plt_c=plot(xx,[yy1 yy2],title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false,ylims=YLIMSci)

    #(8) Investment
    TITLE="investment"
    YLABEL="percentage change"
    yy1=100*((TS_FR_1.Inv ./ TS_FR_1.Inv[t0]) .- 1)
    yy2=100*((TS_FR_2.Inv ./ TS_FR_2.Inv[t0]) .- 1)
    plt_i=plot(xx,[yy1 yy2],title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false,ylims=YLIMSci)

    #(9) Trade balance
    TITLE="trade balance"
    YLABEL="percentage of (annual) GDP"
    yy1=100*TS_FR_1.TB ./ (4*TS_FR_1.GDP)
    yy2=100*TS_FR_2.TB ./ (4*TS_FR_2.GDP)
    plt_tb=plot(xx,[yy1 yy2],title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false)

    #(10) Current account
    TITLE="current account"
    YLABEL="percentage of GDP"
    yy1=100*TS_FR_1.CA ./ TS_FR_1.GDP
    yy2=100*TS_FR_2.CA ./ TS_FR_2.GDP
    plt_ca=plot(xx,[yy1 yy2],title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false)

    #(11) B/K
    TITLE="B/K ratio"
    YLABEL="B/K"
    yy1=TS_FR_1.B ./ TS_FR_1.K
    yy2=TS_FR_2.B ./ TS_FR_2.K
    plt_bk=plot(xx,[yy1 yy2],title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false)

    #Make the plot arrays
    if ForSlides
        l1 = @layout([a b c d; e f g h])
        plt1=plot(plt_b,plt_k,plt_d,plt_tb,
                  plt_r,plt_y,plt_c,plt_i,
                  layout=l1,size=(size_width*4,size_height*2))
        # savefig(plt,"Graphs\\GrimGainsDuration.pdf")
        return plt1
    else
        l1 = @layout([a b c; d e f; g h i])
        plt1=plot(plt_r,plt_d,plt_z,
                  plt_b,plt_c,plt_i,
                  plt_k,plt_y,plt_tb,layout=l1,size=(size_width*3,size_height*3))
        # savefig(plt,"Graphs\\GrimGainsDuration.pdf")
        return plt1
    end
end

function Plot_ThreeConsolidationPaths(B0_zero::Bool,ForSlides::Bool,Tbefore::Int64,Tafter::Int64,TS_FR_1::Paths,TS_FR_2::Paths,TS_FR_3::Paths,parFR::Pars,parFR_pair::Pars,parDefL::Pars)
    xx=collect(range(-Tbefore,stop=Tafter,length=Tbefore+Tafter+1)) ./ 4
    t0=1#Tbefore

    #Common parameters for plots
    size_width=650
    size_height=500
    SIZE=(size_width,size_height)
    XLABEL="years after implementation"
    T=length(TS_FR_1.z)
    FRrnd=round(parFR.FR,digits=2)
    FRrnd_pair=round(parFR_pair.FR,digits=2)
    χrnd_pair=round(parFR_pair.χ,digits=3)
    χrnd_DefL=round(parDefL.χ,digits=3)
    COLORS=[:blue :red :green :black :black]
    LINESTYLES=[:solid :dash :dashdot :solid :dashdot]

    #(1) Productivity shocks
    TITLE="effective productivity"
    COLORz=[:black :blue :red :green]
    LINESTYLEz=[:solid :solid :dash :dashdot]
    YLABEL="z, zD"
    LABELz=["realized" "" "" ""]
    yy=TS_FR_1.z
    yyP1=TS_FR_1.zP
    yyP2=TS_FR_2.zP
    yyP3=TS_FR_3.zP
    plt_z=plot(xx,[yy yyP1 yyP2 yyP3],title=TITLE,xlabel=XLABEL,
               linecolor=COLORz,linestyle=LINESTYLEz,label=LABELz,
               size=SIZE,ylabel=YLABEL,legend=:topright,ylims=[0.98,1.01])

    #(2) Spreads
    TITLE="spreads"
    YLABEL="percentage points"
    LABEL=["debt limit" "dual rule" "primary deficit limit"]
    yy1=TS_FR_1.Spreads
    yy2=TS_FR_2.Spreads
    yy3=TS_FR_3.Spreads
    plt_r=plot(xx,[yy1 yy2 yy3],title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,label=LABEL,
               size=SIZE,ylabel=YLABEL,legend=:topright)

    #(3) Capital
    TITLE="capital"
    YLABEL="k"
    yy1=TS_FR_1.K
    yy2=TS_FR_2.K
    yy3=TS_FR_3.K
    plt_k=plot(xx,[yy1 yy2 yy3],title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false)

    #(4) Debt
    TITLE="debt"
    YLABEL="percentage of GDP"
    yy1=100*TS_FR_1.B ./ (4 .* TS_FR_1.GDP)
    yy2=100*TS_FR_2.B ./ (4 .* TS_FR_2.GDP)
    yy3=100*TS_FR_3.B ./ (4 .* TS_FR_3.GDP)
    fr=parFR.FR
    fr_pair=parFR_pair.FR
    DL=100*fr*ones(length(yy))
    DL_pair=100*fr_pair*ones(length(yy))
    LABELb=["" "" "" "only debt limit" "limit with dual rule"]
    plt_b=plot(xx,[yy1 yy2 yy3 DL DL_pair],title=TITLE,xlabel=XLABEL,label=LABELb,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=:bottomright)

    #(5) Default state
    TITLE="default state"
    YLABEL="share in default"
    yy1=TS_FR_1.Def
    yy2=TS_FR_2.Def
    yy3=TS_FR_3.Def
    plt_d=plot(xx,[yy1 yy2 yy3],title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false)

    #(6) GDP
    TITLE="GDP"
    YLABEL="percentage change"
    yy1=100*((TS_FR_1.GDP ./ TS_FR_1.GDP[t0]) .- 1)
    yy2=100*((TS_FR_2.GDP ./ TS_FR_2.GDP[t0]) .- 1)
    yy3=100*((TS_FR_3.GDP ./ TS_FR_3.GDP[t0]) .- 1)
    plt_y=plot(xx,[yy1 yy2 yy3],title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false)

    #(7) Consumption and (8) Investment
    TITLE="consumption"
    YLABEL="percentage change"
    yy1c=100*((TS_FR_1.Cons ./ TS_FR_1.Cons[t0]) .- 1)
    yy2c=100*((TS_FR_2.Cons ./ TS_FR_2.Cons[t0]) .- 1)
    yy3c=100*((TS_FR_3.Cons ./ TS_FR_3.Cons[t0]) .- 1)
    yylowc=minimum([minimum(yy1c),minimum(yy2c),minimum(yy3c)])
    yyhighc=maximum([maximum(yy1c),maximum(yy2c),maximum(yy3c)])

    yy1i=100*((TS_FR_1.Inv ./ TS_FR_1.Inv[t0]) .- 1)
    yy2i=100*((TS_FR_2.Inv ./ TS_FR_2.Inv[t0]) .- 1)
    yy3i=100*((TS_FR_3.Inv ./ TS_FR_3.Inv[t0]) .- 1)
    yylowi=minimum([minimum(yy1i),minimum(yy2i),minimum(yy3i)])
    yyhighi=maximum([maximum(yy1i),maximum(yy2i),maximum(yy3i)])

    yylow=min(yylowc,yylowi)
    yyhigh=max(yyhighc,yyhighi)

    YLIMSci=[yylow,yyhigh]
    plt_c=plot(xx,[yy1c yy2c yy3c],title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false,ylims=YLIMSci)

    TITLE="investment"
    YLABEL="percentage change"
    plt_i=plot(xx,[yy1i yy2i yy3i],title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false,ylims=YLIMSci)

    #(9) Trade balance
    TITLE="trade balance"
    YLABEL="percentage of (annual) GDP"
    yy1=100*TS_FR_1.TB ./ (4*TS_FR_1.GDP)
    yy2=100*TS_FR_2.TB ./ (4*TS_FR_2.GDP)
    yy3=100*TS_FR_3.TB ./ (4*TS_FR_3.GDP)
    plt_tb=plot(xx,[yy1 yy2 yy3],title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false)

    #(10) Current account
    TITLE="current account"
    YLABEL="percentage of GDP"
    yy1=100*TS_FR_1.CA ./ TS_FR_1.GDP
    yy2=100*TS_FR_2.CA ./ TS_FR_2.GDP
    yy3=100*TS_FR_3.CA ./ TS_FR_3.GDP
    plt_ca=plot(xx,[yy1 yy2 yy3],title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false)

    #(11) B/K
    TITLE="B/K ratio"
    YLABEL="B/K"
    yy1=TS_FR_1.B ./ TS_FR_1.K
    yy2=TS_FR_2.B ./ TS_FR_2.K
    yy3=TS_FR_3.B ./ TS_FR_3.K
    plt_bk=plot(xx,[yy1 yy2 yy3],title=TITLE,xlabel=XLABEL,
               linecolor=COLORS,linestyle=LINESTYLES,
               size=SIZE,ylabel=YLABEL,legend=false)

    #Make the plot arrays
    if B0_zero
        l1 = @layout([a b; c d])
        plt1=plot(plt_r,plt_b,
                  plt_k,plt_y,
                  layout=l1,size=(size_width*2,size_height*2))
        # savefig(plt,"Graphs\\GrimGainsDuration.pdf")
        return plt1
    else
        if ForSlides
            l1 = @layout([a b c d; e f g h])
            plt1=plot(plt_b,plt_k,plt_d,plt_tb,
                      plt_r,plt_y,plt_c,plt_i,
                      layout=l1,size=(size_width*4,size_height*2))
            # savefig(plt,"Graphs\\GrimGainsDuration.pdf")
            return plt1
        else
            l1 = @layout([a b c; d e f; g h i])
            plt1=plot(plt_r,plt_d,plt_z,
                      plt_b,plt_c,plt_i,
                      plt_k,plt_y,plt_tb,layout=l1,size=(size_width*3,size_height*3))
            # savefig(plt,"Graphs\\GrimGainsDuration.pdf")
            return plt1
        end
    end
end

function Plot_Consolidations_Case_Bav_And_B0(N::Int64,Tafter::Int64,case_coulumn::Int64,FOLDER::String)
    MOD0, MOD_DL, MOD_DefL, MOD_FRpair=Unpack_All_Models_Case(case_coulumn,FOLDER)

    Tbefore=4#; Tafter=20; N=20000
    B0_zero=false
    TS_DL=Average_Consolidation_Paths(B0_zero,N,Tbefore,Tafter,MOD0,MOD_DL)
    TS_FRpair=Average_Consolidation_Paths(B0_zero,N,Tbefore,Tafter,MOD0,MOD_FRpair)
    TS_DefL=Average_Consolidation_Paths(B0_zero,N,Tbefore,Tafter,MOD0,MOD_DefL)
    ForSlides=false
    plt_consolidation=Plot_ThreeConsolidationPaths(B0_zero,ForSlides,Tbefore,Tafter,TS_DL,TS_FRpair,TS_DefL,MOD_DL.par,MOD_FRpair.par,MOD_DefL.par)

    Tbefore=1; Tafter_0=Tafter+3#; N=20000
    B0_zero=true
    TS_DL0=Average_Consolidation_Paths(B0_zero,N,Tbefore,Tafter_0,MOD0,MOD_DL)
    TS_FRpair0=Average_Consolidation_Paths(B0_zero,N,Tbefore,Tafter_0,MOD0,MOD_FRpair)
    TS_DefL0=Average_Consolidation_Paths(B0_zero,N,Tbefore,Tafter_0,MOD0,MOD_DefL)
    plt_B0=Plot_ThreeConsolidationPaths(B0_zero,ForSlides,Tbefore,Tafter_0,TS_DL0,TS_FRpair0,TS_DefL0,MOD_DL.par,MOD_FRpair.par,MOD_DefL.par)


    return plt_consolidation, plt_B0
end

###############################################################################
### Default regions and sensitivity of price schedule
###############################################################################

function Plot_Default_Region(N::Int64,MOD::Model,MOD_FR::Model)
    @unpack GRIDS = MOD
    @unpack GR_z, GR_b = GRIDS

    #Fine grids for matrix of default regions
    T=1000
    TS=Simulate_Paths_Ergodic(T,MOD)
    k_at=mean(TS.K)
    av_gdp=mean(TS.GDP)
    z_grid=collect(range(GR_z[1],stop=GR_z[end],length=N))
    b_grid=collect(range(GR_b[1],stop=GR_b[end],length=N))
    b_y=100*b_grid ./ (4*av_gdp)
    R=Array{Float64,2}(undef,N,N) #Repay, for colors
    R_FR=Array{Float64,2}(undef,N,N) #Repay, for colors
    for z_ind in 1:N
        z=z_grid[z_ind]
        vd=MOD.SOLUTION.itp_VD(k_at,z)
        vd_fr=MOD_FR.SOLUTION.itp_VD(k_at,z)
        for b_ind in 1:N
            b=b_grid[b_ind]
            vp=MOD.SOLUTION.itp_VP(b,k_at,z)
            vp_fr=MOD_FR.SOLUTION.itp_VP(b,k_at,z)
            if vd>vp
                R[z_ind,b_ind]=0.0
            else
                R[z_ind,b_ind]=1.0
            end
            if vd_fr>vp_fr
                R_FR[z_ind,b_ind]=0.0
            else
                R_FR[z_ind,b_ind]=1.0
            end
        end
    end
    Rdif=R .- R_FR

    size_width=650
    size_height=500
    SIZE=(size_width,size_height)
    TITLE="benchmark"
    plt=heatmap(b_grid,z_grid,R,
                color = :hot,
                colorbar = false,
                xlabel = "B",
                ylabel = "z",
                legend = false,
                title = TITLE,
                size = SIZE)

    TITLE="fiscal rule"
    plt_fr=heatmap(b_grid,z_grid,R_FR,
                color = :hot,
                colorbar = false,
                xlabel = "B",
                ylabel = "z",
                legend = false,
                title = TITLE,
                size = SIZE)

    plt_dif=heatmap(b_grid,z_grid,Rdif,
                color = :hot,
                xlabel = "B",
                ylabel = "z",
                size = SIZE)

    l = @layout([a b])
    plt=plot(plt,plt_fr,
              layout=l,size=(size_width*2,size_height*1))
    return plt, plt_dif
end

function Plot_PriceSchedule(CASE::String,Bdl::Float64,X0::Array{Float64,1},MOD::Model,MOD_FR::Model)
    @unpack GRIDS, par = MOD

    zbar=X0[1]
    Kbar=X0[2]
    Bbar=X0[3]
    zbar_rnd=round(zbar,digits=1)

    y=FinalOutput(zbar,Kbar,par)
    gdp=4*y
    BLim=MOD_FR.par.FR*gdp

    xx=GRIDS.GR_b
    foo_ben(bprime::Float64)=MOD.SOLUTION.itp_q1(bprime,Kbar,zbar)
    foo_fr(bprime::Float64)=MOD_FR.SOLUTION.itp_q1(bprime,Kbar,zbar)
    yy_ben=foo_ben.(xx)
    yy_fr=foo_fr.(xx)

    COLORS=[:blue :green]
    LINESTYLES=[:solid :dash]
    # TITLE="price schedule, K'=long-run average"
    TITLE="price schedule"
    YLABEL="q(x',z)"; XLABEL="B'"
    YLIMS=[0.0,1.3]
    XLIMS=[0.0,6.0]
    LABEL=["without rule" "with $CASE"]
    plt=plot(xx,[yy_ben yy_fr],title=TITLE,ylabel=YLABEL,
             xlabel=XLABEL,label=LABEL,legend=:bottomleft,
             linecolor=COLORS,linestyle=LINESTYLES,
             ylims=YLIMS,xlims=XLIMS)
        vline!([Bbar],linecolor=:black,label="Av(B), without rule")
        vline!([Bdl],linecolor=:black,linestyle=:dash,label="Av(B), $CASE")

    return plt
end

function Plot_PriceSchedule_z(X0::Array{Float64,1},MOD::Model,MOD_FR::Model)
    @unpack GRIDS, par = MOD

    zbar=X0[1]
    Kbar=X0[2]
    Bbar=X0[3]

    xx=GRIDS.GR_z
    foo_l0(z::Float64)=MOD.SOLUTION.itp_q1(Bbar,Kbar,z)
    foo_l1(z::Float64)=MOD_FR.SOLUTION.itp_q1(Bbar,Kbar,z)
    l0=foo_l0.(xx)
    l1=foo_l1.(xx)

    COLORS=[:blue :green :orange :red]
    LINESTYLES=[:solid :dash :dashdot :dot]
    TITLE="price schedule"
    YLABEL="q(x',z)"; XLABEL="z"
    YLIMS=[0.0,1.3]
    LABEL=["benchmark" "fiscal rule"]
    plt=plot(xx,[l0 l1],title=TITLE,ylabel=YLABEL,
             xlabel=XLABEL,label=LABEL,legend=:bottomright,
             linecolor=COLORS,linestyle=LINESTYLES,
             ylims=YLIMS)

    return plt
end

function Plot_PathOfSpreads(CASE::String,N::Int64,Tbefore::Int64,Tafter::Int64,MOD::Model,MOD_FR::Model)
    NoDefaults=false; ErgodicStart=true; by0=0.44; B0_zero=false
    TS_FR=Average_Consolidation_Paths(B0_zero,N,Tbefore,Tafter,MOD,MOD_FR)

    TSaltSpr=Array{Float64,1}(undef,length(TS_FR.Spreads))
    TSaltSpr .= TS_FR.Spreads
    for t in Tbefore+1:length(TSaltSpr)
        z=TS_FR.z[t]; kk=TS_FR.K[t]; bb=TS_FR.B[t]
        qq=max(MOD.SOLUTION.itp_q1(bb,kk,z),1e-3)
        TSaltSpr[t]=ComputeSpreads(z,kk,bb,MOD.SOLUTION,MOD.par)
    end

    TITLE="spreads"
    LABELS=["without rule" "with $CASE"]
    YLABEL="percentage points"
    XLABEL="years after implementation"
    YLIMS=[0.0,7.0]
    COLORS=[:blue :green]
    LINESTYLES=[:solid :dash]
    xx=collect(range(-Tbefore,stop=Tafter,length=Tbefore+Tafter+1))
    yy1=TSaltSpr
    yy2=TS_FR.Spreads
    plt=plot(xx./4,[yy1 yy2],title=TITLE,ylabel=YLABEL,
             xlabel=XLABEL,label=LABELS,legend=:bottomright,
             ylims=YLIMS,linecolor=COLORS,linestyle=LINESTYLES)

    tat=1
    X0=[TS_FR.z[tat]; TS_FR.K[tat]; TS_FR.B[tat]]

    PPP=InitiateEmptyPaths(10000)
    Simulate_z_shocks!(1.0,PPP,MOD.par)
    zstd=std(PPP.z)
    zL=TS_FR.z[tat]-zstd
    X0L=[zL; TS_FR.K[tat]; TS_FR.B[tat]]
    return plt, X0, X0L, TS_FR.B[end]
end

function PlotPriceAndSpreads(CASE::String,N::Int64,Tbefore::Int64,Tafter::Int64,MOD::Model,MOD_FR::Model)
    #Then plot path of spreads
    plt_s, X0, X0L, Bdl=Plot_PathOfSpreads(CASE,N,Tbefore,Tafter,MOD,MOD_FR)

    #First plot price schedule
    plt_q=Plot_PriceSchedule(CASE,Bdl,X0,MOD,MOD_FR)

    #Make the plot arrays
    # size_width=650
    # size_height=500
    # l1 = @layout([a b; c d])
    # plt=plot(plt_q,plt_s,
    #          plt_qL,plt_z,layout=l1,size=(size_width*2,size_height*2))
    # return plt

    #Make the plot arrays
    size_width=650
    size_height=500
    l1 = @layout([a b])
    plt=plot(plt_q,plt_s,layout=l1,size=(size_width*2,size_height*1))
    return plt
end

function Two_PlotsPriceAndSpreads(N::Int64,Tbefore::Int64,Tafter::Int64,MOD::Model,MOD_DL::Model,MOD_BBR::Model)
    #Then plot path of spreads
    plt_s_dl, X0_dl, X0L_dl, Bdl_dl=Plot_PathOfSpreads("debt limit",N,Tbefore,Tafter,MOD,MOD_DL)
    plt_s_bbr, X0_bbr, X0L_bbr, Bdl_bbr=Plot_PathOfSpreads("primary-deficit limit",N,Tbefore,Tafter,MOD,MOD_BBR)

    #First plot price schedule
    plt_q_dl=Plot_PriceSchedule("debt limit",Bdl_dl,X0_dl,MOD,MOD_DL)
    plt_q_bbr=Plot_PriceSchedule("primary-deficit limit",Bdl_bbr,X0_bbr,MOD,MOD_BBR)

    #Make the plot arrays
    size_width=650
    size_height=500
    l1 = @layout([a b; c d])
    plt=plot(plt_q_dl,plt_s_dl,
             plt_q_bbr,plt_s_bbr,
             layout=l1,size=(size_width*2,size_height*2))
    return plt
end

function PlotPriceAndSpreads_Case(case_coulumn::Int64,FOLDER::String)
    MOD0, MOD_DL, MOD_BBR, MOD_FRpair=Unpack_All_Models_Case(case_coulumn,FOLDER)
    Tbefore=4; Tafter=20; N=20000
    plt_dl_bbr=Two_PlotsPriceAndSpreads(N,Tbefore,Tafter,MOD0,MOD_DL,MOD_BBR)
    plt_dual_rule=PlotPriceAndSpreads("dual rule",N,Tbefore,Tafter,MOD0,MOD_FRpair)
    return plt_dl_bbr, plt_dual_rule
end
