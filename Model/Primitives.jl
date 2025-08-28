
using Parameters, Interpolations, Optim, SharedArrays, DelimitedFiles,
      Distributions, FastGaussQuadrature, LinearAlgebra, Random, Statistics,
      SparseArrays, Sobol, Roots

################################################################
#### Defining parameters and other structures for the model ####
################################################################
#Define parameter and grid structure, use quarterly calibration
@with_kw struct Pars
    ################################################################
    ######## Preferences and technology ############################
    ################################################################
    #Preferences
    σ::Float64 = 2.0          #CRRA parameter
    β::Float64 = 0.98         #Discount factor consistent with annualized r=8.4%
    r_star::Float64 = 0.01    #Risk-fr  ee interest rate
    βhh::Float64 = 0.98       #Household discount factor
    PatientHH::Bool = false   #Is there relative myopia?
    #Debt parameters
    γ::Float64 = 0.05         #Reciprocal of average maturity
    κ::Float64 = 0.03         #Coupon payments
    qmax::Float64 = (γ+κ*(1-γ))/(γ+r_star)
    Covenants::Bool = false   #Include debt covenants to avoid dilution
    #Default cost and debt parameters
    θ::Float64 = 1/16        #Probability of re-admission (median 4.0 years Gelos et al)
    d0::Float64 = -0.18819#-0.2914       #income default cost
    d1::Float64 = 0.24558#0.4162        #income default cost
    #Production and capital accumulation
    Decentralized::Bool = false  #Who makes the capita accumulation decision
    Strict::Bool = false
    ForeignInvestors::Bool = false
    α::Float64 = 0.33            #Capital share
    δ::Float64 = 0.05            #Capital depreciation rate
    φ::Float64 = 21.0             #Capital adjustment cost parameter
    #Fiscal rules and government expenditure
    WithFR::Bool = false         #Define whether to apply fiscal rule
    FR::Float64 = 0.30           #Maximum b'/GDP
    χ::Float64 = 0.03               #Fiscal consolidation parameter
    OnlyBudgetBalance::Bool = false
    #Stochastic process
    #parameters for productivity shock
    σ_ϵz::Float64 = 0.017
    μ_ϵz::Float64 = 0.0-0.5*(σ_ϵz^2)
    dist_ϵz::UnivariateDistribution = Normal(μ_ϵz,σ_ϵz)
    ρ_z::Float64 = 0.95
    μ_z::Float64 = exp(μ_ϵz+0.5*(σ_ϵz^2))
    zlow::Float64 = exp(log(μ_z)-3.0*sqrt((σ_ϵz^2.0)/(1.0-(ρ_z^2.0))))
    zhigh::Float64 = exp(log(μ_z)+3.0*sqrt((σ_ϵz^2.0)/(1.0-(ρ_z^2.0))))
    #Quadrature parameters
    N_GL::Int64 = 21
    #Grids
    Nz::Int64 = 11
    Nk::Int64 = 11
    Nb::Int64 = 21
    klow::Float64 = 0.15
    khigh::Float64 = 5.0
    blow::Float64 = 0.0
    bhigh::Float64 = 4.0#15.0
    #Parameters for solution algorithm
    cmin::Float64 = 1e-3
    Tol_V::Float64 = 1e-6       #Tolerance for absolute distance for value functions
    Tol_q::Float64 = 1e-6       #Tolerance for absolute distance for q
    cnt_max::Int64 = 200           #Maximum number of iterations on VFI
    klowOpt::Float64 = 0.9*klow             #Minimum level of capital for optimization
    khighOpt::Float64 = 1.1*khigh           #Maximum level of capital for optimization
    blowOpt::Float64 = blow-0.01             #Minimum level of debt for optimization
    bhighOpt::Float64 = bhigh+0.01            #Maximum level of debt for optimization
    #Simulation parameters
    drp::Int64 = 1000
    Tmom::Int64 = 1000
    NSamplesMoments::Int64 = 10000
    HPFilter_Par::Float64 = 1600.0
end

@with_kw struct Grids
    #Grids of states
    GR_z::Array{Float64,1}
    GR_k::Array{Float64,1}
    GR_b::Array{Float64,1}
    #Quadrature vectors for integrals
    ϵz_weights::Vector{Float64}
    ZPRIME::Array{Float64,2}
    PDFz::Array{Float64,1}
    FacQz::Float64
end

function Create_GL_objects(GR_z::Array{Float64,1},N_GL::Int64,σϵ::Float64,ρ::Float64,μ_z::Float64,dist_ϵ::UnivariateDistribution)
    #Gauss-Legendre vectors for y'
    GL_nodes, GL_weights = gausslegendre(N_GL)
    ϵlow=-3.0*σϵ
    ϵhigh=3.0*σϵ
    ϵ_nodes=0.5*(ϵhigh-ϵlow).*GL_nodes .+ 0.5*(ϵhigh+ϵlow)
    ϵ_weights=GL_weights .* 0.5*(ϵhigh-ϵlow)
    #Matrices for integration over shock y
    N=length(GR_z)
    ZPRIME=Array{Float64,2}(undef,N,N_GL)
    PDFz=pdf.(dist_ϵ,ϵ_nodes)
    for z_ind in 1:N
        z=GR_z[z_ind]
        ZPRIME[z_ind,:]=exp.((1.0-ρ)*log(μ_z) + ρ*log(z) .+ ϵ_nodes)
    end
    FacQz=dot(ϵ_weights,PDFz)
    return ϵ_weights, ZPRIME, PDFz, FacQz
end

function CreateGrids(par::Pars)
    #Grid for z
    @unpack Nz, zlow, zhigh = par
    GR_z=collect(range(zlow,stop=zhigh,length=Nz))

    #Gauss-Legendre objects
    @unpack N_GL, σ_ϵz, ρ_z, μ_z, dist_ϵz = par
    ϵz_weights, ZPRIME, PDFz, FacQz=Create_GL_objects(GR_z,N_GL,σ_ϵz,ρ_z,μ_z,dist_ϵz)

    #Grid of capital
    @unpack Nk, klow, khigh = par
    GR_k=collect(range(klow,stop=khigh,length=Nk))

    #Grid of debt
    @unpack Nb, blow, bhigh = par
    GR_b=collect(range(blow,stop=bhigh,length=Nb))

    return Grids(GR_z,GR_k,GR_b,ϵz_weights,ZPRIME,PDFz,FacQz)
end

function Setup()
    par=Pars()
    GRIDS=CreateGrids(par)
    return par, GRIDS
end

@with_kw mutable struct Solution{T1,T2,T3,T4,T5,T6,T7}
    ### Arrays
    #Value Functions
    VD::T1
    VP::T2
    V::T2
    #Expectations and price
    EVD::T1
    EV::T2
    q1::T2
    #Policy function
    kprime_D::T1
    kprime::T2
    bprime::T2
    Tr::T2
    DEV_RULE::T2
    ### Interpolation objects
    #Value Functions
    itp_VD::T3
    itp_VP::T4
    itp_V::T4
    #Expectations and price
    itp_EVD::T3
    itp_EV::T4
    itp_q1::T5
    #Policy functions
    itp_kprime_D::T6
    itp_kprime::T7
    itp_bprime::T7

    #Household value
    VDhh::T1
    VPhh::T2
    EVDhh::T1
    EVhh::T2
    itp_VDhh::T3
    itp_VPhh::T4
    itp_EVDhh::T3
    itp_EVhh::T4
end

@with_kw struct Model
    SOLUTION::Solution
    GRIDS::Grids
    par::Pars
end

@with_kw mutable struct HH_itpObjects{T1,T2,T3,T4}
    #Auxiliary arrays for expectations
    dR_Def::T1
    dR_Rep::T2
    #Arrays of expectations
    ER_Def::T1
    ER_Rep::T2
    #Interpolation objects
    itp_dR_Def::T3
    itp_dR_Rep::T4
    itp_ER_Def::T3
    itp_ER_Rep::T4
end

@with_kw mutable struct State
    Default::Bool
    z::Float64
    k::Float64
    b::Float64
end

################################################################
#################### Auxiliary functions #######################
################################################################
function MyBisection(foo,a::Float64,b::Float64;xatol::Float64=1e-8)
    s=sign(foo(a))
    x=(a+b)/2.0
    d=(b-a)/2.0
    while d>xatol
        d=d/2.0
        if s==sign(foo(x))
            x=x+d
        else
            x=x-d
        end
    end
    return x
end

function TransformIntoBounds(x::Float64,min::Float64,max::Float64)
    (max - min) * (1.0/(1.0 + exp(-x))) + min
end

function TransformIntoReals(x::Float64,min::Float64,max::Float64)
    log((x - min)/(max - x))
end

################################################################
############# Functions to interpolate matrices ################
################################################################
function CreateInterpolation_ValueFunctions(MAT::Array{Float64},IsDefault::Bool,GRIDS::Grids)
    @unpack GR_z, GR_k = GRIDS
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    Ks=range(GR_k[1],stop=GR_k[end],length=length(GR_k))
    ORDER_SHOCKS=Linear()
    ORDER_K_STATES=Linear()#Cubic(Line(OnGrid()))
    ORDER_B_STATES=Linear()
    if IsDefault==true
        INT_DIMENSIONS=(BSpline(ORDER_K_STATES),BSpline(ORDER_SHOCKS))
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Ks,Zs),Interpolations.Line())
    else
        @unpack GR_b = GRIDS
        Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
        INT_DIMENSIONS=(BSpline(ORDER_B_STATES),BSpline(ORDER_K_STATES),BSpline(ORDER_SHOCKS))
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,Ks,Zs),Interpolations.Line())
    end
end

function CreateInterpolation_Price(MAT::Array{Float64},GRIDS::Grids)
    @unpack GR_z, GR_k, GR_b = GRIDS
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    Ks=range(GR_k[1],stop=GR_k[end],length=length(GR_k))
    Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
    ORDER_SHOCKS=Linear()
    # ORDER_STATES=Cubic(Line(OnGrid()))
    ORDER_STATES=Linear()
    INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS))
    return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,Ks,Zs),Interpolations.Flat())
end

function CreateInterpolation_Policies(MAT::Array{Float64},IsDefault::Bool,GRIDS::Grids)
    @unpack GR_z, GR_k = GRIDS
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    Ks=range(GR_k[1],stop=GR_k[end],length=length(GR_k))
    ORDER_SHOCKS=Linear()
    # ORDER_STATES=Cubic(Line(OnGrid()))
    ORDER_STATES=Linear()
    if IsDefault==true
        INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS))
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Ks,Zs),Interpolations.Flat())
    else
        @unpack GR_b = GRIDS
        Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
        INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS))
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,Ks,Zs),Interpolations.Flat())
    end
end

function CreateInterpolation_ForExpectations(MAT::Array{Float64,1},GRIDS::Grids)
    @unpack GR_z = GRIDS
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    ORDER_SHOCKS=Linear()
    INT_DIMENSIONS=(BSpline(ORDER_SHOCKS))
    return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Zs),Interpolations.Line())
end

function CreateInterpolation_HouseholdObjects(MAT::Array{Float64},IsDefault::Bool,GRIDS::Grids)
    @unpack GR_z, GR_k = GRIDS
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    Ks=range(GR_k[1],stop=GR_k[end],length=length(GR_k))
    ORDER_SHOCKS=Linear()
    ORDER_K_STATES=Linear()#Cubic(Line(OnGrid()))
    ORDER_B_STATES=Linear()
    if IsDefault==true
        INT_DIMENSIONS=(BSpline(ORDER_K_STATES),BSpline(ORDER_SHOCKS))
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Ks,Zs),Interpolations.Line())
    else
        @unpack GR_b = GRIDS
        Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
        INT_DIMENSIONS=(BSpline(ORDER_B_STATES),BSpline(ORDER_K_STATES),BSpline(ORDER_SHOCKS))
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,Ks,Zs),Interpolations.Line())
    end
end

################################################################################
### Functions to pack models in vectors and save to CSV
################################################################################
function InitiateEmptySolution(GRIDS::Grids,par::Pars)
    @unpack Nz, Nk, Nb = par
    ### Allocate all values to object
    VD=zeros(Float64,Nk,Nz)
    VP=zeros(Float64,Nb,Nk,Nz)
    V=zeros(Float64,Nb,Nk,Nz)
    itp_VD=CreateInterpolation_ValueFunctions(VD,true,GRIDS)
    itp_VP=CreateInterpolation_ValueFunctions(VP,false,GRIDS)
    itp_V=CreateInterpolation_ValueFunctions(V,false,GRIDS)
    #Expectations and price
    EVD=zeros(Float64,Nk,Nz)
    EV=zeros(Float64,Nb,Nk,Nz)
    q1=zeros(Float64,Nb,Nk,Nz)

    itp_EVD=CreateInterpolation_ValueFunctions(EVD,true,GRIDS)
    itp_EV=CreateInterpolation_ValueFunctions(EV,false,GRIDS)
    itp_q1=CreateInterpolation_Price(q1,GRIDS)

    #Policy function
    kprime_D=zeros(Float64,Nk,Nz)
    kprime=zeros(Float64,Nb,Nk,Nz)
    bprime=zeros(Float64,Nb,Nk,Nz)
    Tr=zeros(Float64,Nb,Nk,Nz)
    DEV_RULE=zeros(Float64,Nb,Nk,Nz)

    #Policy function
    itp_kprime_D=CreateInterpolation_Policies(kprime_D,true,GRIDS)
    itp_kprime=CreateInterpolation_Policies(kprime,false,GRIDS)
    itp_bprime=CreateInterpolation_Policies(bprime,false,GRIDS)
    #Household value
    VDhh=zeros(Float64,Nk,Nz)
    VPhh=zeros(Float64,Nb,Nk,Nz)
    EVDhh=zeros(Float64,Nk,Nz)
    EVhh=zeros(Float64,Nb,Nk,Nz)
    itp_VDhh=CreateInterpolation_ValueFunctions(VDhh,true,GRIDS)
    itp_VPhh=CreateInterpolation_ValueFunctions(VPhh,false,GRIDS)
    itp_EVDhh=CreateInterpolation_ValueFunctions(VDhh,true,GRIDS)
    itp_EVhh=CreateInterpolation_ValueFunctions(VPhh,false,GRIDS)
    return Solution(VD,VP,V,EVD,EV,q1,kprime_D,kprime,bprime,Tr,DEV_RULE,itp_VD,itp_VP,itp_V,itp_EVD,itp_EV,itp_q1,itp_kprime_D,itp_kprime,itp_bprime,VDhh,VPhh,EVDhh,EVhh,itp_VDhh,itp_VPhh,itp_EVDhh,itp_EVhh)
end

function StackSolution_Vector(SOLUTION::Solution)
    #Stack vectors of repayment first
    @unpack VP, V, EV, q1, VPhh, EVhh = SOLUTION
    @unpack kprime, bprime, Tr, DEV_RULE = SOLUTION
    VEC=reshape(VP,(:))
    VEC=vcat(VEC,reshape(V,(:)))
    VEC=vcat(VEC,reshape(EV,(:)))
    VEC=vcat(VEC,reshape(q1,(:)))
    VEC=vcat(VEC,reshape(kprime,(:)))
    VEC=vcat(VEC,reshape(bprime,(:)))
    VEC=vcat(VEC,reshape(Tr,(:)))
    VEC=vcat(VEC,reshape(DEV_RULE,(:)))
    VEC=vcat(VEC,reshape(VPhh,(:)))
    VEC=vcat(VEC,reshape(EVhh,(:)))

    #Then stack vectors of default
    @unpack VD, EVD, kprime_D, VDhh, EVDhh = SOLUTION
    VEC=vcat(VEC,reshape(VD,(:)))
    VEC=vcat(VEC,reshape(EVD,(:)))
    VEC=vcat(VEC,reshape(kprime_D,(:)))
    VEC=vcat(VEC,reshape(VDhh,(:)))
    VEC=vcat(VEC,reshape(EVDhh,(:)))

    return VEC
end

function VectorOfRelevantParameters(par::Pars)
    #Stack important values from parameters
    #Parameters for Grids go first
    #Grids sizes
    VEC=par.N_GL                #1
    VEC=vcat(VEC,par.Nz)        #2
    VEC=vcat(VEC,par.Nk)        #3
    VEC=vcat(VEC,par.Nb)        #4

    #Grids bounds
    VEC=vcat(VEC,par.klow)      #5
    VEC=vcat(VEC,par.khigh)     #6
    VEC=vcat(VEC,par.blow)      #7
    VEC=vcat(VEC,par.bhigh)     #8

    #Parameter values
    VEC=vcat(VEC,par.β)         #9
    VEC=vcat(VEC,par.d0)        #10
    VEC=vcat(VEC,par.d1)        #11
    VEC=vcat(VEC,par.φ)         #12

    #Extra parameters
    VEC=vcat(VEC,par.cnt_max)   #13

    if par.WithFR               #14
        VEC=vcat(VEC,1.0)
    else
        VEC=vcat(VEC,0.0)
    end

    VEC=vcat(VEC,par.FR)        #15
    VEC=vcat(VEC,par.χ)         #16
    VEC=vcat(VEC,par.θ)         #17
    if par.Covenants
        VEC=vcat(VEC,1.0)         #18
    else
        VEC=vcat(VEC,0.0)         #18
    end

    if par.Decentralized
        VEC=vcat(VEC,1.0)         #19
    else
        VEC=vcat(VEC,0.0)         #19
    end

    if par.ForeignInvestors
        VEC=vcat(VEC,1.0)         #20
    else
        VEC=vcat(VEC,0.0)         #20
    end

    if par.Strict
        VEC=vcat(VEC,1.0)         #21
    else
        VEC=vcat(VEC,0.0)         #21
    end

    if par.PatientHH
        VEC=vcat(VEC,1.0)         #22
    else
        VEC=vcat(VEC,0.0)         #22
    end

    VEC=vcat(VEC,par.γ)         #23

    if par.OnlyBudgetBalance
        VEC=vcat(VEC,1.0)         #24
    else
        VEC=vcat(VEC,0.0)         #24
    end


    return VEC
end

function Create_Model_Vector(MODEL::Model)
    @unpack SOLUTION, par = MODEL
    VEC_PAR=VectorOfRelevantParameters(par)
    N_parameters=length(VEC_PAR)
    VEC=vcat(N_parameters,VEC_PAR)

    #Stack SOLUTION in one vector
    VEC_SOL=StackSolution_Vector(SOLUTION)

    return vcat(VEC,VEC_SOL)
end

function SaveModel_Vector(NAME::String,MODEL::Model)
    VEC=Create_Model_Vector(MODEL)
    writedlm(NAME,VEC,',')
    return nothing
end

function ExtractMatrixFromSolutionVector(start::Int64,size::Int64,IsDefault::Bool,VEC::Vector{Float64},par::Pars)
    @unpack Nz, Nk, Nb = par
    if IsDefault
        I=(Nk,Nz)
    else
        I=(Nb,Nk,Nz)
    end
    finish=start+size-1
    vec=VEC[start:finish]
    return reshape(vec,I)
end

function TransformVectorToSolution(VEC::Array{Float64},GRIDS::Grids,par::Pars)
    #The file SolutionVector.csv must be in FOLDER
    #for this function to work
    @unpack Nz, Nk, Nb = par
    size_repayment=Nz*Nk*Nb
    size_default=Nz*Nk

    #Allocate vectors into matrices
    #Repayment
    start=1
    VP=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment
    V=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment
    EV=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment
    q1=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment
    kprime=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment
    bprime=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment
    Tr=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment
    DEV_RULE=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment
    VPhh=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment
    EVhh=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)

    #Default
    start=start+size_repayment
    VD=ExtractMatrixFromSolutionVector(start,size_default,true,VEC,par)
    start=start+size_default
    EVD=ExtractMatrixFromSolutionVector(start,size_default,true,VEC,par)
    start=start+size_default
    kprime_D=ExtractMatrixFromSolutionVector(start,size_default,true,VEC,par)
    start=start+size_default
    VDhh=ExtractMatrixFromSolutionVector(start,size_default,true,VEC,par)
    start=start+size_default
    EVDhh=ExtractMatrixFromSolutionVector(start,size_default,true,VEC,par)
    #Create interpolation objects
    itp_VD=CreateInterpolation_ValueFunctions(VD,true,GRIDS)
    itp_VP=CreateInterpolation_ValueFunctions(VP,false,GRIDS)
    itp_V=CreateInterpolation_ValueFunctions(V,false,GRIDS)

    itp_EVD=CreateInterpolation_ValueFunctions(EVD,true,GRIDS)
    itp_EV=CreateInterpolation_ValueFunctions(EV,false,GRIDS)
    itp_q1=CreateInterpolation_Price(q1,GRIDS)

    itp_kprime_D=CreateInterpolation_Policies(kprime_D,true,GRIDS)

    itp_kprime=CreateInterpolation_Policies(kprime,false,GRIDS)
    itp_bprime=CreateInterpolation_Policies(bprime,false,GRIDS)

    itp_VDhh=CreateInterpolation_ValueFunctions(VDhh,true,GRIDS)
    itp_VPhh=CreateInterpolation_ValueFunctions(VPhh,false,GRIDS)
    itp_EVDhh=CreateInterpolation_ValueFunctions(VDhh,true,GRIDS)
    itp_EVhh=CreateInterpolation_ValueFunctions(VPhh,false,GRIDS)

    ### Interpolation objects
    #Input values for end of time
    return Solution(VD,VP,V,EVD,EV,q1,kprime_D,kprime,bprime,Tr,DEV_RULE,itp_VD,itp_VP,itp_V,itp_EVD,itp_EV,itp_q1,itp_kprime_D,itp_kprime,itp_bprime,VDhh,VPhh,EVDhh,EVhh,itp_VDhh,itp_VPhh,itp_EVDhh,itp_EVhh)
end

function UnpackParameters_Vector(VEC::Array{Float64})
    par=Pars()

    #Parameters for Grids go first
    #Grids sizes
    par=Pars(par,N_GL=convert(Int64,VEC[1]))
    par=Pars(par,Nz=convert(Int64,VEC[2]))
    par=Pars(par,Nk=convert(Int64,VEC[3]))
    par=Pars(par,Nb=convert(Int64,VEC[4]))

    #Grids bounds
    par=Pars(par,klow=VEC[5])
    par=Pars(par,khigh=VEC[6])
    par=Pars(par,klowOpt=0.999*par.klow,khighOpt=1.001*par.khigh)
    par=Pars(par,blow=VEC[7])
    par=Pars(par,bhigh=VEC[8])
    par=Pars(par,blowOpt=par.blow-0.001,bhighOpt=par.bhigh+0.001)

    #Parameter values
    par=Pars(par,β=VEC[9])
    par=Pars(par,d0=VEC[10])
    par=Pars(par,d1=VEC[11])
    par=Pars(par,φ=VEC[12])

    #Extra parameters
    par=Pars(par,cnt_max=VEC[13])

    if VEC[14]==1.0
        par=Pars(par,WithFR=true)
    else
        par=Pars(par,WithFR=false)
    end

    par=Pars(par,FR=VEC[15])
    par=Pars(par,χ=VEC[16])
    par=Pars(par,θ=VEC[17])

    if VEC[18]==1.0
        par=Pars(par,Covenants=true)
    else
        par=Pars(par,Covenants=false)
    end

    if VEC[19]==1.0
        par=Pars(par,Decentralized=true)
    else
        par=Pars(par,Decentralized=false)
    end

    if VEC[20]==1.0
        par=Pars(par,ForeignInvestors=true)
    else
        par=Pars(par,ForeignInvestors=false)
    end

    if VEC[21]==1.0
        par=Pars(par,Strict=true)
    else
        par=Pars(par,Strict=false)
    end

    if VEC[22]==1.0
        par=Pars(par,PatientHH=true)
        βstar=1/(1+par.r_star)
        βhh=0.5*(par.β+βstar)
        par=Pars(par,βhh=βhh)
    else
        par=Pars(par,PatientHH=false)
        par=Pars(par,βhh=par.β)
    end

    par=Pars(par,γ=VEC[23])

    if VEC[24]==1.0
        par=Pars(par,OnlyBudgetBalance=true)
    else
        par=Pars(par,OnlyBudgetBalance=false)
    end

    return par
end

function Setup_From_Vector(VEC_PAR::Array{Float64})
    #Vector of parameters has to have the correct structure
    par=UnpackParameters_Vector(VEC_PAR)
    GRIDS=CreateGrids(par)
    return par, GRIDS
end

function UnpackModel_Vector(VEC)
    #Extract parameters and create grids
    N_parameters=convert(Int64,VEC[1])
    VEC_PAR=1.0*VEC[2:N_parameters+1]
    par, GRIDS=Setup_From_Vector(VEC_PAR)

    #Extract solution object
    VEC_SOL=VEC[N_parameters+2:end]
    SOL=TransformVectorToSolution(VEC_SOL,GRIDS,par)

    return Model(SOL,GRIDS,par)
end

function UnpackModel_File(NAME::String,FOLDER::String)
    #Unpack Vector with data
    if FOLDER==" "
        VEC=readdlm(NAME,',')
    else
        VEC=readdlm("$FOLDER\\$NAME",',')
    end

    return UnpackModel_Vector(VEC)
end

################################################################
########## Functions to compute expectations ###################
################################################################
function Expectation_over_zprime(foo,z_ind::Int64,GRIDS::Grids)
    #foo is a function of floats for z'
    #kN', kT', and b' are given
    @unpack ϵz_weights, ZPRIME, PDFz, FacQz = GRIDS
    int=0.0
    for j in 1:length(ϵz_weights)
        int=int+ϵz_weights[j]*PDFz[j]*foo(ZPRIME[z_ind,j])
    end
    return int/FacQz
end

function Expectation_over_zprime_2(foo,z_ind::Int64,GRIDS::Grids)
    #foo is a function of floats for z' that returns a
    #tuple (v,q)
    @unpack ϵz_weights, ZPRIME, PDFz, FacQz = GRIDS
    int_v=0.0
    int_q=0.0
    int_vhh=0.0
    for j in 1:length(ϵz_weights)
        v, q, vhh=foo(ZPRIME[z_ind,j])
        int_v=int_v+ϵz_weights[j]*PDFz[j]*v
        int_q=int_q+ϵz_weights[j]*PDFz[j]*q
        int_vhh=int_vhh+ϵz_weights[j]*PDFz[j]*vhh
    end
    return int_v/FacQz, int_q/FacQz, int_vhh/FacQz
end

function SDF_Lenders(par::Pars)
    @unpack r_star = par
    return 1/(1+r_star)
end

function Calculate_Covenant_b(x::State,kprime::Float64,bprime::Float64,SOLUTION::Solution,par::Pars)
    @unpack Covenants, Decentralized = par
    if Covenants
        @unpack γ = par
        @unpack itp_q1 = SOLUTION
        @unpack z, k, b = x
        #Compute compensation per remaining bond (1-γ)b
        if Decentralized
            #Compensate for dilution due to change in debt
            qq0=itp_q1((1-γ)*b,kprime,z)
        else
            #Compensate for dilution due to change in debt and capital
            qq0=itp_q1((1-γ)*b,k,z)
        end
        qq1=itp_q1(bprime,kprime,z)
        return max(0.0,qq0-qq1)
    else
        return 0.0
    end
end

function ValueAndBondsPayoff(zprime::Float64,kprime::Float64,bprime::Float64,SOLUTION::Solution,par::Pars)
    @unpack itp_VP, itp_VD = SOLUTION
    vd=min(0.0,itp_VD(kprime,zprime))
    vp=min(0.0,itp_VP(bprime,kprime,zprime))
    if vd>vp
        @unpack PatientHH = par
        if PatientHH
            @unpack itp_VDhh = SOLUTION
            return vd, 0.0, itp_VDhh(kprime,zprime)
        else
            return vd, 0.0, vd
        end
    else
        @unpack γ, κ, qmax = par
        SDF=SDF_Lenders(par)
        if γ==1.0
            @unpack PatientHH = par
            if PatientHH
                return vp, SDF, vp
            else
                return vp, SDF, vp
            end
        else
            @unpack itp_q1, itp_kprime, itp_bprime = SOLUTION
            kk=itp_kprime(bprime,kprime,zprime)
            bb=itp_bprime(bprime,kprime,zprime)
            qq=min(qmax,max(0.0,itp_q1(bb,kk,zprime)))
            if par.Covenants
                xprime=State(false,zprime,kprime,bprime)
                CC=Calculate_Covenant_b(xprime,kk,bb,SOLUTION,par)
            else
                CC=0.0
            end
            @unpack PatientHH = par
            if PatientHH
                @unpack itp_VPhh = SOLUTION
                return vp, SDF*(γ+(1-γ)*(κ+qq+CC)), itp_VPhh(bprime,kprime,zprime)
            else
                return vp, SDF*(γ+(1-γ)*(κ+qq+CC)), vp
            end
        end
    end
end

function ValueAndBondsPayoff_Grim(zprime::Float64,kprime::Float64,bprime::Float64,SOLUTION::Solution,SOLUTION_NO_RULE::Solution,par::Pars)
    @unpack itp_VP, itp_VD = SOLUTION
    vd=min(0.0,itp_VD(kprime,zprime))
    vp=min(0.0,itp_VP(bprime,kprime,zprime))
    vp_no_rule=min(0.0,SOLUTION_NO_RULE.itp_VP(bprime,kprime,zprime))
    if vd>max(vp,vp_no_rule)
        #Default, keep rule for later
        @unpack PatientHH = par
        if PatientHH
            @unpack itp_VDhh = SOLUTION
            return vd, 0.0, itp_VDhh(kprime,zprime)
        else
            return vd, 0.0, vd
        end
    else
        @unpack γ, κ, qmax = par
        SDF=SDF_Lenders(par)
        if vp>=vp_no_rule
            #Do not deviate
            if γ==1.0
                @unpack PatientHH = par
                if PatientHH
                    @unpack itp_VPhh = SOLUTION
                    return vp, SDF, itp_VPhh(bprime,kprime,zprime)
                else
                    return vp, SDF, vp
                end
            else
                @unpack itp_q1, itp_kprime, itp_bprime = SOLUTION
                kk=itp_kprime(bprime,kprime,zprime)
                bb=itp_bprime(bprime,kprime,zprime)
                qq=min(qmax,max(0.0,itp_q1(bb,kk,zprime)))
                if par.Covenants
                    xprime=State(false,zprime,kprime,bprime)
                    CC=Calculate_Covenant_b(xprime,kk,bb,SOLUTION,par)
                else
                    CC=0.0
                end
                @unpack PatientHH = par
                if PatientHH
                    @unpack itp_VPhh = SOLUTION
                    return vp, SDF*(γ+(1-γ)*(κ+qq+CC)), itp_VPhh(bprime,kprime,zprime)
                else
                    return vp, SDF*(γ+(1-γ)*(κ+qq+CC)), vp
                end
            end
        else
            #Deviate
            if γ==1.0
                @unpack PatientHH = par
                if PatientHH
                    @unpack itp_VPhh = SOLUTION_NO_RULE
                    return vp_no_rule, SDF, itp_VPhh(bprime,kprime,zprime)
                else
                    return vp_no_rule, SDF, vp_no_rule
                end
            else
                @unpack itp_q1, itp_kprime, itp_bprime = SOLUTION_NO_RULE
                kk=itp_kprime(bprime,kprime,zprime)
                bb=itp_bprime(bprime,kprime,zprime)
                qq=min(qmax,max(0.0,itp_q1(bb,kk,zprime)))
                if par.Covenants
                    xprime=State(false,zprime,kprime,bprime)
                    CC=Calculate_Covenant_b(xprime,kk,bb,SOLUTION_NO_RULE,par)
                else
                    CC=0.0
                end
                @unpack PatientHH = par
                if PatientHH
                    @unpack itp_VPhh = SOLUTION_NO_RULE
                    return vp_no_rule, SDF*(γ+(1-γ)*(κ+qq+CC)), itp_VPhh(bprime,kprime,zprime)
                else
                    return vp_no_rule, SDF*(γ+(1-γ)*(κ+qq+CC)), vp_no_rule
                end
            end
        end
    end
end

function UpdateExpectations!(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Loop over all states to compute expectations over p and z

    #Default
    for I in CartesianIndices(SOLUTION.EVD)
        (k_ind,z_ind)=Tuple(I)
        foo_mat_D=CreateInterpolation_ForExpectations(SOLUTION.VD[k_ind,:],GRIDS)
        SOLUTION.EVD[I]=Expectation_over_zprime(foo_mat_D,z_ind,GRIDS)

        foo_mat_Dhh=CreateInterpolation_ForExpectations(SOLUTION.VDhh[k_ind,:],GRIDS)
        SOLUTION.EVDhh[I]=Expectation_over_zprime(foo_mat_Dhh,z_ind,GRIDS)
    end

    #Repayment and bond price
    for I in CartesianIndices(SOLUTION.q1)
        (b_ind,k_ind,z_ind)=Tuple(I)
        kprime=GRIDS.GR_k[k_ind]
        bprime=GRIDS.GR_b[b_ind]
        foo_mat_vq(zprime::Float64)=ValueAndBondsPayoff(zprime,kprime,bprime,SOLUTION,par)
        SOLUTION.EV[I], SOLUTION.q1[I], SOLUTION.EVhh[I]=Expectation_over_zprime_2(foo_mat_vq,z_ind,GRIDS)
    end

    return nothing
end

function UpdateExpectations_Grim!(SOLUTION::Solution,SOLUTION_NO_RULE::Solution,GRIDS::Grids,par::Pars)
    #Loop over all states to compute expectations over p and z

    #Default
    for I in CartesianIndices(SOLUTION.EVD)
        (k_ind,z_ind)=Tuple(I)
        foo_mat_D=CreateInterpolation_ForExpectations(SOLUTION.VD[k_ind,:],GRIDS)
        SOLUTION.EVD[I]=Expectation_over_zprime(foo_mat_D,z_ind,GRIDS)

        foo_mat_Dhh=CreateInterpolation_ForExpectations(SOLUTION.VDhh[k_ind,:],GRIDS)
        SOLUTION.EVDhh[I]=Expectation_over_zprime(foo_mat_Dhh,z_ind,GRIDS)
    end

    #Repayment and bond price
    for I in CartesianIndices(SOLUTION.q1)
        (b_ind,k_ind,z_ind)=Tuple(I)
        kprime=GRIDS.GR_k[k_ind]
        bprime=GRIDS.GR_b[b_ind]
        foo_mat_vq(zprime::Float64)=ValueAndBondsPayoff_Grim(zprime,kprime,bprime,SOLUTION,SOLUTION_NO_RULE,par)
        SOLUTION.EV[I], SOLUTION.q1[I], SOLUTION.EVhh[I]=Expectation_over_zprime_2(foo_mat_vq,z_ind,GRIDS)
    end

    return nothing
end

################################################################
################# Preferences and technology ###################
################################################################
function FinalOutput(z::Float64,k::Float64,par::Pars)
    @unpack α = par
    return z*(k^α)
end

function Utility(c::Float64,par::Pars)
    @unpack σ = par
    return (c^(1.0-σ))/(1.0-σ)
end

function U_c(c::Float64,par::Pars)
    @unpack σ = par
    return c^(-σ)
end

function zDefault(z::Float64,par::Pars)
    @unpack d0, d1 = par
    return z-max(0.0,d0*z+d1*z*z)
end

function CapitalAdjustment(kprime::Float64,k::Float64,par::Pars)
    @unpack φ = par
    return 0.5*φ*((kprime-k)^2.0)/k
end

function dΨ_d1(kprime::Float64,k::Float64,par::Pars)
    @unpack φ = par
    return φ*(kprime-k)/k
end

function dΨ_d2(kprime::Float64,k::Float64,par::Pars)
    @unpack φ = par
    return -φ*0.5*((kprime^2)-(k^2))/(k^2)
end

function DebtLimit(z::Float64,k::Float64,b::Float64,par::Pars)
    @unpack FR, χ, γ, OnlyBudgetBalance = par
    #Maximum b/gdp
    y=FinalOutput(z,k,par)
    gdp=4*y    #annualized GDP
    if OnlyBudgetBalance
        DR=χ*gdp   #maximum primary deficit
        return (1-γ)*b+DR
    else
        DL=FR*gdp  #debt limit
        DR=χ*gdp   #maximum primary deficit if b>DL
        #the parameter χ controls the maximum forced debt reduction
        #if shock is so bad that b is way above DL then the
        #government is not forced to pay all the way down to DL,
        #just pay, at most, DR
        if b>=DL
            #Is binding today, allow adjustment
            return max(DL,(1-γ)*b+DR)
        else
            #Not binding today, cannot get to DL
            return DL
        end
    end
end

function ComputeSpreadWithQ(qq::Float64,par::Pars)
    @unpack r_star, γ, κ = par
    ib=(((γ+(1-γ)*(κ+qq))/qq)^4)-1
    rf=((1+r_star)^4)-1
    return 100*(ib-rf)
end

function Calculate_Tr(x::State,kprime::Float64,bprime::Float64,SOLUTION::Solution,par::Pars)
    @unpack γ, κ = par
    @unpack itp_q1 = SOLUTION
    @unpack z, b = x
    #Compute net borrowing from the rest of the world
    Covenant=Calculate_Covenant_b(x,kprime,bprime,SOLUTION,par)
    qq=itp_q1(bprime,kprime,z)
    return qq*(bprime-(1-γ)*b)-(γ+κ*(1-γ))*b-(1-γ)*b*Covenant
end

function r_dec(z::Float64,k::Float64,par::Pars)
    @unpack α = par
    return α*z*(k^(α-1.0))
end

function R_dec(kprime::Float64,z::Float64,k::Float64,par::Pars)
    @unpack δ = par
    #Return to capital in terms of the final consumption good
    r=r_dec(z,k,par)
    ψ2=dΨ_d2(kprime,k,par)
    return r+(1.0-δ)-ψ2
end

###############################################################################
#Function to compute consumption and value given the state and policies
###############################################################################
function Evaluate_cons_state(x::State,kprime::Float64,Tr::Float64,par::Pars)
    @unpack ForeignInvestors = par
    @unpack z, k = x
    if x.Default
        zD=zDefault(z,par)
        y=FinalOutput(zD,k,par)
    else
        y=FinalOutput(z,k,par)
    end

    if ForeignInvestors
        @unpack α = par
        W=(1-α)*y
        return W+Tr
    else
        @unpack δ = par
        inv=kprime-(1-δ)*k
        AdjCost=CapitalAdjustment(kprime,k,par)
        return y-inv-AdjCost+Tr
    end
end

function Evaluate_ValueFunction(IsHousehold::Bool,x::State,kprime::Float64,bprime::Float64,SOLUTION::Solution,par::Pars)
    @unpack β, cmin = par
    @unpack z = x
    if x.Default
        @unpack θ = par
        @unpack itp_EV, itp_EVD = SOLUTION
        Tr=0.0
        cons=Evaluate_cons_state(x,kprime,Tr,par)
        if cons>cmin
            if IsHousehold
                @unpack βhh = par
                @unpack itp_EVhh, itp_EVDhh = SOLUTION
                return Utility(cons,par)+βhh*θ*itp_EVhh(0.0,kprime,z)+βhh*(1.0-θ)*itp_EVDhh(kprime,z)
            else
                return Utility(cons,par)+β*θ*itp_EV(0.0,kprime,z)+β*(1.0-θ)*itp_EVD(kprime,z)
            end
        else
            return Utility(cmin,par)+cons
        end
    else
        @unpack itp_EV, itp_q1 = SOLUTION
        Tr=Calculate_Tr(x,kprime,bprime,SOLUTION,par)
        cons=Evaluate_cons_state(x,kprime,Tr,par)
        if cons>0.0
            if IsHousehold
                @unpack βhh = par
                @unpack itp_EVhh = SOLUTION
                return Utility(cons,par)+βhh*itp_EVhh(bprime,kprime,z)
            else
                qq=itp_q1(bprime,kprime,z)
                if bprime>0.0 && qq==0.0
                    #Small penalty for larger debt positions
                    #wrong side of laffer curve, it is decreasing
                    return Utility(cons,par)+β*itp_EV(bprime,kprime,z)-abs(bprime)*sqrt(eps(Float64))
                else
                    return Utility(cons,par)+β*itp_EV(bprime,kprime,z)
                end
            end
        else
            return Utility(cmin,par)+cons
        end
    end
end

###############################################################################
#Function to compute and update household objects
###############################################################################
function ComputeExpectations_HHDef!(itp_D,itp_R,E_MAT::Array{Float64,2},GRIDS::Grids,par::Pars)
    #It will use MAT_D and MAT_R to compute expectations
    #It will change the values in E_MAT
    #Loop over all states to compute expectations over p and z
    @unpack θ = par
    @unpack GR_k, GR_z = GRIDS
    for I in CartesianIndices(E_MAT)
        (k_ind,z_ind)=Tuple(I)
        kprime=GR_k[k_ind]
        function foo_mat(zprime::Float64)
            return θ*max(0.0,itp_R(0.0,kprime,zprime))+(1-θ)*max(0.0,itp_D(kprime,zprime))
        end
        E_MAT[I]=Expectation_over_zprime(foo_mat,z_ind,GRIDS)
    end
    return nothing
end

function ComputeExpectations_HHRep!(itp_D,itp_R,E_MAT::Array{Float64,3},SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #It will use MAT_D and MAT_R to compute expectations
    #It will change the values in E_MAT
    #Loop over all states to compute expectations over p and z
    @unpack itp_VD, itp_VP = SOLUTION
    @unpack GR_b, GR_k, GR_z = GRIDS
    for I in CartesianIndices(E_MAT)
        (b_ind,k_ind,z_ind)=Tuple(I)
        bprime=GR_b[b_ind]; kprime=GR_k[k_ind]
        function foo_mat(zprime::Float64)
            vd=itp_VD(kprime,zprime)
            vp=itp_VP(bprime,kprime,zprime)
            if vd>vp
                return max(0.0,itp_D(kprime,zprime))
            else
                return max(0.0,itp_R(bprime,kprime,zprime))
            end
        end
        E_MAT[I]=Expectation_over_zprime(foo_mat,z_ind,GRIDS)
    end
    return nothing
end

function ComputeExpectations_HHRep_Grim!(itp_D,itp_R,itp_R_no_rule,E_MAT::Array{Float64,3},SOLUTION::Solution,SOLUTION_NO_RULE::Solution,GRIDS::Grids,par::Pars)
    #It will use MAT_D and MAT_R to compute expectations
    #It will change the values in E_MAT
    #Loop over all states to compute expectations over p and z
    @unpack itp_VD, itp_VP = SOLUTION
    @unpack GR_b, GR_k, GR_z = GRIDS
    for I in CartesianIndices(E_MAT)
        (b_ind,k_ind,z_ind)=Tuple(I)
        bprime=GR_b[b_ind]; kprime=GR_k[k_ind]
        function foo_mat(zprime::Float64)
            vd=itp_VD(kprime,zprime)
            vp=itp_VP(bprime,kprime,zprime)
            vp_no_rule=SOLUTION_NO_RULE.itp_VP(bprime,kprime,zprime)
            if vd>max(vp,vp_no_rule)
                return max(0.0,itp_D(kprime,zprime))
            else
                if vp>=vp_no_rule
                    return max(0.0,itp_R(bprime,kprime,zprime))
                else
                    return max(0.0,itp_R_no_rule(bprime,kprime,zprime))
                end
            end
        end
        E_MAT[I]=Expectation_over_zprime(foo_mat,z_ind,GRIDS)
    end
    return nothing
end

function HH_k_Returns(Default::Bool,I::CartesianIndex,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack cmin, βhh, ForeignInvestors = par
    @unpack GR_z, GR_k = GRIDS

    if Default
        @unpack kprime_D = SOLUTION
        #Unpack state and compute output
        (k_ind,z_ind)=Tuple(I)
        z=GR_z[z_ind]; k=GR_k[k_ind]
        zD=zDefault(z,par)
        kprimef=kprime_D[I]; T=0.0

        if ForeignInvestors
            return R_dec(kprimef,zD,k,par)
        else
            x=State(Default,z,k,0.0)
            cons=Evaluate_cons_state(x,kprimef,T,par)
            if cons<=0.0
                cons=cmin
            end
            return βhh*U_c(cons,par)*R_dec(kprimef,zD,k,par)
        end
    else
        @unpack GR_b = GRIDS
        @unpack kprime, Tr, VP, VD = SOLUTION
        #Unpack state index
        (b_ind,k_ind,z_ind)=Tuple(I)
        #Unpack state and policies
        z=GR_z[z_ind]; k=GR_k[k_ind]; b=GR_b[b_ind]
        kprimef=kprime[I]; T=Tr[I]

        if ForeignInvestors
            return R_dec(kprimef,z,k,par)
        else
            x=State(Default,z,k,b)
            cons=Evaluate_cons_state(x,kprimef,T,par)

            if cons>0.0
                #Calculate return
                return βhh*U_c(cons,par)*R_dec(kprimef,z,k,par)
            else
                #Use return in default
                #this avoids overshooting of interpolation of RN and RT
                return HH_OBJ.dR_Def[k_ind,z_ind]
            end
        end
    end
end

function HH_k_Returns_Grim(Default::Bool,I::CartesianIndex,HH_OBJ::HH_itpObjects,SOLUTION::Solution,SOLUTION_NO_RULE::Solution,GRIDS::Grids,par::Pars)
    @unpack cmin, βhh, ForeignInvestors = par
    @unpack GR_z, GR_k = GRIDS

    if Default
        @unpack kprime_D = SOLUTION
        #Unpack state and compute output
        (k_ind,z_ind)=Tuple(I)
        z=GR_z[z_ind]; k=GR_k[k_ind]
        zD=zDefault(z,par)
        kprimef=kprime_D[I]; T=0.0

        if ForeignInvestors
            return R_dec(kprimef,zD,k,par)
        else
            x=State(Default,z,k,0.0)
            cons=Evaluate_cons_state(x,kprimef,T,par)
            if cons<=0.0
                cons=cmin
            end
            return βhh*U_c(cons,par)*R_dec(kprimef,zD,k,par)
        end
    else
        @unpack GR_b = GRIDS
        if SOLUTION_NO_RULE.VP[I]>SOLUTION.VP[I]
            @unpack kprime, Tr, VP, VD = SOLUTION_NO_RULE
        else
            @unpack kprime, Tr, VP, VD = SOLUTION
        end
        #Unpack state index
        (b_ind,k_ind,z_ind)=Tuple(I)
        #Unpack state and policies
        z=GR_z[z_ind]; k=GR_k[k_ind]; b=GR_b[b_ind]
        kprimef=kprime[I]; T=Tr[I]

        if ForeignInvestors
            return R_dec(kprimef,z,k,par)
        else
            x=State(Default,z,k,b)
            cons=Evaluate_cons_state(x,kprimef,T,par)

            if cons>0.0
                #Calculate return
                return βhh*U_c(cons,par)*R_dec(kprimef,z,k,par)
            else
                #Use return in default
                #this avoids overshooting of interpolation of RN and RT
                return HH_OBJ.dR_Def[k_ind,z_ind]
            end
        end
    end
end

function UpdateHH_Obj!(HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Loop over all states to fill matrices in default
    Default=true
    for I in CartesianIndices(HH_OBJ.dR_Def)
        HH_OBJ.dR_Def[I]=HH_k_Returns(Default,I,HH_OBJ,SOLUTION,GRIDS,par)
    end
    HH_OBJ.itp_dR_Def=CreateInterpolation_HouseholdObjects(HH_OBJ.dR_Def,true,GRIDS)

    #Loop over all states to fill matrices in repayment
    Default=false
    for I in CartesianIndices(HH_OBJ.dR_Rep)
        HH_OBJ.dR_Rep[I]=HH_k_Returns(Default,I,HH_OBJ,SOLUTION,GRIDS,par)
    end
    HH_OBJ.itp_dR_Rep=CreateInterpolation_HouseholdObjects(HH_OBJ.dR_Rep,false,GRIDS)

    #Compute expectations in default
    ComputeExpectations_HHDef!(HH_OBJ.itp_dR_Def,HH_OBJ.itp_dR_Rep,HH_OBJ.ER_Def,GRIDS,par)
    HH_OBJ.itp_ER_Def=CreateInterpolation_HouseholdObjects(HH_OBJ.ER_Def,true,GRIDS)

    #Compute expectations in repayment
    ComputeExpectations_HHRep!(HH_OBJ.itp_dR_Def,HH_OBJ.itp_dR_Rep,HH_OBJ.ER_Rep,SOLUTION,GRIDS,par)
    HH_OBJ.itp_ER_Rep=CreateInterpolation_HouseholdObjects(HH_OBJ.ER_Rep,false,GRIDS)
    return nothing
end

function UpdateHH_Obj_Grim!(HH_OBJ::HH_itpObjects,HH_OBJ_NO_RULE::HH_itpObjects,SOLUTION::Solution,SOLUTION_NO_RULE::Solution,GRIDS::Grids,par::Pars)
    #Loop over all states to fill matrices in default
    Default=true
    for I in CartesianIndices(HH_OBJ.dR_Def)
        HH_OBJ.dR_Def[I]=HH_k_Returns_Grim(Default,I,HH_OBJ,SOLUTION,SOLUTION_NO_RULE,GRIDS,par)
    end
    HH_OBJ.itp_dR_Def=CreateInterpolation_HouseholdObjects(HH_OBJ.dR_Def,true,GRIDS)

    #Loop over all states to fill matrices in repayment
    Default=false
    for I in CartesianIndices(HH_OBJ.dR_Rep)
        HH_OBJ.dR_Rep[I]=HH_k_Returns_Grim(Default,I,HH_OBJ,SOLUTION,SOLUTION_NO_RULE,GRIDS,par)
    end
    HH_OBJ.itp_dR_Rep=CreateInterpolation_HouseholdObjects(HH_OBJ.dR_Rep,false,GRIDS)

    #Compute expectations in default
    ComputeExpectations_HHDef!(HH_OBJ.itp_dR_Def,HH_OBJ.itp_dR_Rep,HH_OBJ.ER_Def,GRIDS,par)
    HH_OBJ.itp_ER_Def=CreateInterpolation_HouseholdObjects(HH_OBJ.ER_Def,true,GRIDS)

    #Compute expectations in repayment
    ComputeExpectations_HHRep_Grim!(HH_OBJ.itp_dR_Def,HH_OBJ.itp_dR_Rep,HH_OBJ_NO_RULE.itp_dR_Rep,HH_OBJ.ER_Rep,SOLUTION,SOLUTION_NO_RULE,GRIDS,par)
    HH_OBJ.itp_ER_Rep=CreateInterpolation_HouseholdObjects(HH_OBJ.ER_Rep,false,GRIDS)
    return nothing
end

function InitiateHH_Obj(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Nz, Nk, Nb = par
    #Allocate empty arrays
    #Auxiliary arrays for expectations
    dR_Def=zeros(Float64,Nk,Nz)
    dR_Rep=zeros(Float64,Nb,Nk,Nz)
    #Arrays of expectations
    ER_Def=zeros(Float64,Nk,Nz)
    ER_Rep=zeros(Float64,Nb,Nk,Nz)
    #Create interpolation objects
    itp_dR_Def=CreateInterpolation_ValueFunctions(ER_Def,true,GRIDS)
    itp_dR_Rep=CreateInterpolation_ValueFunctions(ER_Rep,false,GRIDS)

    itp_ER_Def=CreateInterpolation_ValueFunctions(ER_Def,true,GRIDS)
    itp_ER_Rep=CreateInterpolation_ValueFunctions(ER_Rep,false,GRIDS)
    #Arrange objects in structure
    HH_OBJ=HH_itpObjects(dR_Def,dR_Rep,ER_Def,ER_Rep,itp_dR_Def,itp_dR_Rep,itp_ER_Def,itp_ER_Rep)
    #Modify values using SOLUTION at end of time
    UpdateHH_Obj!(HH_OBJ,SOLUTION,GRIDS,par)
    return HH_OBJ
end

###############################################################################
#Functions to optimize given guesses and state
###############################################################################
#Functions to compute FOCs given (s,x) and a try of K'
function HH_FOC(x::State,kprime::Float64,bprime::Float64,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack cmin, ForeignInvestors = par
    @unpack Default, z, k = x
    #compute term related to adjustment cost
    ψ1=dΨ_d1(kprime,k,par)
    if Default
        @unpack itp_ER_Def = HH_OBJ
        # compute expectation over z
        Ev=itp_ER_Def(kprime,z)
        if ForeignInvestors
            @unpack r_star = par
            return Ev-(1+r_star)*(1.0+ψ1)
        else
            #Compute present consumption
            Tr=0.0
            cons=Evaluate_cons_state(x,kprime,Tr,par)
            c=max(cmin,cons)
            #Return FOC
            return Ev-U_c(c,par)*(1.0+ψ1)
        end
    else
        @unpack itp_ER_Rep = HH_OBJ
        #compute expectation over z
        Ev=itp_ER_Rep(bprime,kprime,z)
        if ForeignInvestors
            @unpack r_star = par
            return Ev-(1+r_star)*(1.0+ψ1)
        else
            #Compute present consumption
            Tr=Calculate_Tr(x,kprime,bprime,SOLUTION,par)
            cons=Evaluate_cons_state(x,kprime,Tr,par)
            c=max(cmin,cons)
            #Return FOC
            return Ev-U_c(c,par)*(1.0+ψ1)
        end
    end
end

#Functions to compute optimal capital policy from FOCs
function FindBracketing(foo,ν::Float64,x0::Float64,xmin::Float64,xmax::Float64)
    #foo is the FOC function of x
    #x0 is the first value to test, will be at the 45° line
    if foo(x0)>0.0
        #the lower value should make foo positive
        xL=x0
        xH=x0*(1+ν)
        while foo(xH)>0.0
            #the upper value should make foo negative
            xH=xH*(1+ν)
            if xH>xmax
                break
            end
        end
    else
        #the upper value should make foo negative
        xH=x0
        xL=x0*(1-ν)
        while foo(xL)<0.0
            #the lower value should make foo positive
            xL=xL*(1-ν)
            if xL<xmin
                break
            end
        end
    end
    return xL, xH
end

function HHOptim(x::State,bprime::Float64,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack k = x
    foo(kprime::Float64)=HH_FOC(x,kprime,bprime,HH_OBJ,SOLUTION,GRIDS,par)
    ν=0.05
    kkL, kkH=FindBracketing(foo,ν,k,k-ν,k+ν)
    kk=MyBisection(foo,kkL,kkH;xatol=1e-6)

    if x.Default
        Tr=0.0
        cons=Evaluate_cons_state(x,kk,Tr,par)
        while cons<0.0
            kk=0.5*(k+kk)
            cons=Evaluate_cons_state(x,kk,Tr,par)
        end
        return kk
    else
        return kk
    end
end

function GridSearchOverK_D(x::State,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    kpol=0; bprime=0.0 #This is only used for the planner in Default
    val=-Inf
    for ktry in 1:length(GRIDS.GR_k)
        kprime=GRIDS.GR_k[ktry]
        vv=Evaluate_ValueFunction(false,x,kprime,bprime,SOLUTION,par)
        if vv>val
            val=vv
            kpol=ktry
        else
            break
        end
    end

    if kpol<=1
        klow=par.klowOpt
        kpol_r=GRIDS.GR_k[1]
    else
        klow=GRIDS.GR_k[kpol-1]
        kpol_r=GRIDS.GR_k[kpol]
    end

    if kpol==par.Nk
        khigh=par.khighOpt
        kpol_r=GRIDS.GR_k[end]
    else
        khigh=GRIDS.GR_k[kpol+1]
        kpol_r=GRIDS.GR_k[kpol]
    end

    return klow, khigh
end

function Search_bprime_unconstrained(x::State,kprime::Float64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack GR_b = GRIDS
    val=-Inf
    bpol=0
    for btry in 1:par.Nb
        vv=Evaluate_ValueFunction(false,x,kprime,GR_b[btry],SOLUTION,par)
        if vv>val
            val=vv
            bpol=btry
        end
    end
    if bpol>1
        blowOpt=GR_b[bpol-1]
    else
        blowOpt=par.blowOpt
    end
    if bpol<par.Nb
        bhighOpt=GR_b[bpol+1]
    else
        bhighOpt=par.bhighOpt
    end
    return blowOpt, bhighOpt
end

function BoundsForBprime(I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack GR_z, GR_k, GR_b = GRIDS
    (b_ind,k_ind,z_ind)=Tuple(I)
    Default=false; z=GR_z[z_ind]; k=GR_k[k_ind]; b=GR_b[b_ind]
    x=State(Default,z,k,b)

    if SOLUTION.kprime[I]==0.0
        kprime=GR_k[k_ind]
    else
        kprime=SOLUTION.kprime[I]
    end

    blowOpt, bhighOpt=Search_bprime_unconstrained(x,kprime,SOLUTION,GRIDS,par)

    if par.WithFR
        bbar=DebtLimit(z,k,b,par)
        if bbar<blowOpt
            #blowOpt is lower bound of tight interval
            #that contains best b' in grid, this means
            #that the debt limit is binding
            return 0.9*bbar, bbar
        else
            return blowOpt, min(bbar,bhighOpt)
        end
    else
        return blowOpt, bhighOpt
    end
end

function InitialPolicyGuess_Planner(PreviousPolicy::Bool,I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    (b_ind,k_ind,z_ind)=Tuple(I)
    if PreviousPolicy
        #Choose policy from previous iteration as initial guess for capital
        if SOLUTION.kprime[I]==0.0
            #It is the first attempt, use current k
            k0=GRIDS.GR_k[k_ind]
        else
            if SOLUTION.kprime[I]<GRIDS.GR_k[end]
                k0=max(SOLUTION.kprime[I],GRIDS.GR_k[1])
            else
                k0=min(SOLUTION.kprime[I],GRIDS.GR_k[end])
            end
        end

        if SOLUTION.bprime[I]<GRIDS.GR_b[end]
            b0=max(SOLUTION.bprime[I],GRIDS.GR_b[1])
        else
            b0=min(SOLUTION.bprime[I],GRIDS.GR_b[end])
        end

        return [k0, b0]
    else
        #Choose present k as initial guess (stay the same)
        k0=GRIDS.GR_k[k_ind]
        b0=GRIDS.GR_b[b_ind]

        return [k0, b0]
    end
end

function ValueInRepayment_PLA(x::State,klowOpt::Float64,khighOpt::Float64,blowOpt::Float64,bhighOpt::Float64,X_REAL::Array{Float64,1},SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #transform policy tries into interval
    kprime=TransformIntoBounds(X_REAL[1],klowOpt,khighOpt)
    bprime=TransformIntoBounds(X_REAL[2],blowOpt,bhighOpt)
    vv=Evaluate_ValueFunction(false,x,kprime,bprime,SOLUTION,par)
    return vv
end

function ValueInRepayment_DEC(x::State,I::CartesianIndex,bprime::Float64,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Strict = par
    if Strict
        kprime=HHOptim(x,bprime,HH_OBJ,SOLUTION,GRIDS,par)
    else
        kprime=SOLUTION.kprime[I]
    end
    vv=Evaluate_ValueFunction(false,x,kprime,bprime,SOLUTION,par)
    return vv
end

function Optimizer_Planner(Default::Bool,I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack GR_z, GR_k = GRIDS
    if Default
        (k_ind,z_ind)=Tuple(I)
        z=GR_z[z_ind]; k=GR_k[k_ind]; b=0.0
        x=State(Default,z,k,b); bprime=0.0
        klow, khigh=GridSearchOverK_D(x,SOLUTION,GRIDS,par)
        foo(kprime::Float64)=-Evaluate_ValueFunction(false,x,kprime,bprime,SOLUTION,par)
        res=optimize(foo,klow,khigh,GoldenSection())
        return -Optim.minimum(res), Optim.minimizer(res)
    else
        @unpack klowOpt, khighOpt = par
        @unpack GR_b = GRIDS
        (b_ind,k_ind,z_ind)=Tuple(I)
        z=GR_z[z_ind]; k=GR_k[k_ind]; b=GR_b[b_ind]
        x=State(Default,z,k,b)

        blowOpt, bhighOpt=BoundsForBprime(I,SOLUTION,GRIDS,par)
        if blowOpt==bhighOpt
            blowOpt=par.blowOpt
        end

        PreviousPolicy=true
        X0_BOUNDS=InitialPolicyGuess_Planner(PreviousPolicy,I,SOLUTION,GRIDS,par)
        X0=Array{Float64,1}(undef,2)
        X0[1]=TransformIntoReals(X0_BOUNDS[1],klowOpt,khighOpt)
        if X0_BOUNDS[2]<bhighOpt && X0_BOUNDS[2]>blowOpt
            X0[2]=TransformIntoReals(X0_BOUNDS[2],blowOpt,bhighOpt)
        else
            X0[2]=TransformIntoReals(0.5*(blowOpt+bhighOpt),blowOpt,bhighOpt)
        end

        foo_P(X_REAL::Array{Float64,1})=-ValueInRepayment_PLA(x,klowOpt,khighOpt,blowOpt,bhighOpt,X_REAL,SOLUTION,GRIDS,par)
        res=optimize(foo_P,X0,NelderMead())
        vv=-Optim.minimum(res)
        kprime=TransformIntoBounds(Optim.minimizer(res)[1],klowOpt,khighOpt)
        bprime=TransformIntoBounds(Optim.minimizer(res)[2],blowOpt,bhighOpt)
        Tr=Calculate_Tr(x,kprime,bprime,SOLUTION,par)
        return vv, kprime, bprime, Tr
    end
end

function Optimizer_Decentralized(Default::Bool,I::CartesianIndex,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack GR_z, GR_k = GRIDS
    if Default
        (k_ind,z_ind)=Tuple(I)
        z=GR_z[z_ind]; k=GR_k[k_ind]; b=0.0
        x=State(Default,z,k,b); bprime=0.0

        kprime=HHOptim(x,bprime,HH_OBJ,SOLUTION,GRIDS,par)
        val=Evaluate_ValueFunction(false,x,kprime,bprime,SOLUTION,par)
        return val, kprime
    else
        @unpack GR_b = GRIDS
        (b_ind,k_ind,z_ind)=Tuple(I)
        z=GR_z[z_ind]; k=GR_k[k_ind]; b=GR_b[b_ind]
        x=State(Default,z,k,b)

        blowOpt, bhighOpt=BoundsForBprime(I,SOLUTION,GRIDS,par)
        foo(bprime::Float64)=-ValueInRepayment_DEC(x,I,bprime,HH_OBJ,SOLUTION,GRIDS,par)
        res=optimize(foo,blowOpt,bhighOpt,GoldenSection())

        bprime=Optim.minimizer(res)
        kprime=HHOptim(x,bprime,HH_OBJ,SOLUTION,GRIDS,par)
        Tr=Calculate_Tr(x,kprime,bprime,SOLUTION,par)
        return -Optim.minimum(res), kprime, bprime, Tr
    end
end

###############################################################################
#Update solution
###############################################################################
function DefaultUpdater!(I::CartesianIndex,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Decentralized, PatientHH = par
    Default=true
    if Decentralized
        SOLUTION.VD[I], SOLUTION.kprime_D[I]=Optimizer_Decentralized(Default,I,HH_OBJ,SOLUTION,GRIDS,par)
        if PatientHH
            @unpack GR_z, GR_k = GRIDS
            IsHousehold=true
            (k_ind,z_ind)=Tuple(I)
            z=GR_z[z_ind]; k=GR_k[k_ind]; b=0.0
            x=State(Default,z,k,b)
            kprime=SOLUTION.kprime_D[I]
            bprime=0.0
            SOLUTION.VDhh[I]=Evaluate_ValueFunction(IsHousehold,x,kprime,bprime,SOLUTION,par)
        else
            SOLUTION.VDhh[I]=SOLUTION.VD[I]
        end
        return nothing
    else
        SOLUTION.VD[I], SOLUTION.kprime_D[I]=Optimizer_Planner(Default,I,SOLUTION,GRIDS,par)
        SOLUTION.VDhh[I]=SOLUTION.VD[I]
        return nothing
    end
end

function RepaymentUpdater!(I::CartesianIndex,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Decentralized, PatientHH = par
    Default=false
    if Decentralized
        SOLUTION.VP[I], SOLUTION.kprime[I], SOLUTION.bprime[I], SOLUTION.Tr[I]=Optimizer_Decentralized(Default,I,HH_OBJ,SOLUTION,GRIDS,par)
    else
        SOLUTION.VP[I], SOLUTION.kprime[I], SOLUTION.bprime[I], SOLUTION.Tr[I]=Optimizer_Planner(Default,I,SOLUTION,GRIDS,par)
    end
    (b_ind,k_ind,z_ind)=Tuple(I)
    if SOLUTION.VP[I]<SOLUTION.VD[k_ind,z_ind]
        SOLUTION.V[I]=SOLUTION.VD[k_ind,z_ind]
    else
        SOLUTION.V[I]=SOLUTION.VP[I]
    end
    if PatientHH
        @unpack GR_z, GR_k, GR_b = GRIDS
        IsHousehold=true
        (b_ind,k_ind,z_ind)=Tuple(I)
        z=GR_z[z_ind]; k=GR_k[k_ind]; b=GR_b[b_ind]
        x=State(Default,z,k,b)
        kprime=SOLUTION.kprime[I]
        bprime=SOLUTION.bprime[I]
        SOLUTION.VPhh[I]=Evaluate_ValueFunction(IsHousehold,x,kprime,bprime,SOLUTION,par)
    else
        SOLUTION.VPhh[I]=SOLUTION.VP[I]
    end
    return nothing
end

function UpdateDefault!(HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Loop over all states to fill array of VD
    for I in CartesianIndices(SOLUTION.VD)
        DefaultUpdater!(I,HH_OBJ,SOLUTION,GRIDS,par)
    end

    IsDefault=true
    SOLUTION.itp_VD=CreateInterpolation_ValueFunctions(SOLUTION.VD,IsDefault,GRIDS)
    SOLUTION.itp_kprime_D=CreateInterpolation_Policies(SOLUTION.kprime_D,IsDefault,GRIDS)
    SOLUTION.itp_VDhh=CreateInterpolation_ValueFunctions(SOLUTION.VDhh,IsDefault,GRIDS)

    return nothing
end

function UpdateRepayment!(HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Loop over all states to fill array of VD
    for I in CartesianIndices(SOLUTION.VP)
        RepaymentUpdater!(I,HH_OBJ,SOLUTION,GRIDS,par)
    end

    IsDefault=false
    SOLUTION.itp_VP=CreateInterpolation_ValueFunctions(SOLUTION.VP,IsDefault,GRIDS)
    SOLUTION.itp_V=CreateInterpolation_ValueFunctions(SOLUTION.V,IsDefault,GRIDS)
    SOLUTION.itp_kprime=CreateInterpolation_Policies(SOLUTION.kprime,IsDefault,GRIDS)
    SOLUTION.itp_bprime=CreateInterpolation_Policies(SOLUTION.bprime,IsDefault,GRIDS)
    SOLUTION.itp_VPhh=CreateInterpolation_ValueFunctions(SOLUTION.VPhh,IsDefault,GRIDS)

    return nothing
end

function UpdateSolution!(HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    UpdateDefault!(HH_OBJ,SOLUTION,GRIDS,par)
    UpdateRepayment!(HH_OBJ,SOLUTION,GRIDS,par)
    UpdateExpectations!(SOLUTION,GRIDS,par)

    #Compute expectation interpolations
    IsDefault=true
    SOLUTION.itp_EVD=CreateInterpolation_ValueFunctions(SOLUTION.EVD,IsDefault,GRIDS)
    SOLUTION.itp_EVDhh=CreateInterpolation_ValueFunctions(SOLUTION.EVDhh,IsDefault,GRIDS)
    IsDefault=false
    SOLUTION.itp_EV=CreateInterpolation_ValueFunctions(SOLUTION.EV,IsDefault,GRIDS)
    SOLUTION.itp_q1=CreateInterpolation_Price(SOLUTION.q1,GRIDS)
    SOLUTION.itp_EVhh=CreateInterpolation_ValueFunctions(SOLUTION.EVhh,IsDefault,GRIDS)

    @unpack Decentralized = par
    if Decentralized
        UpdateHH_Obj!(HH_OBJ,SOLUTION,GRIDS,par)
    end
    return nothing
end

function UpdateSolution_Grim!(HH_OBJ::HH_itpObjects,HH_OBJ_NO_RULE::HH_itpObjects,SOLUTION::Solution,SOLUTION_NO_RULE::Solution,GRIDS::Grids,par::Pars)
    UpdateDefault!(HH_OBJ,SOLUTION,GRIDS,par)
    UpdateRepayment!(HH_OBJ,SOLUTION,GRIDS,par)
    UpdateExpectations_Grim!(SOLUTION,SOLUTION_NO_RULE,GRIDS,par)

    #Compute expectation interpolations
    IsDefault=true
    SOLUTION.itp_EVD=CreateInterpolation_ValueFunctions(SOLUTION.EVD,IsDefault,GRIDS)
    SOLUTION.itp_EVDhh=CreateInterpolation_ValueFunctions(SOLUTION.EVDhh,IsDefault,GRIDS)
    IsDefault=false
    SOLUTION.itp_EV=CreateInterpolation_ValueFunctions(SOLUTION.EV,IsDefault,GRIDS)
    SOLUTION.itp_q1=CreateInterpolation_Price(SOLUTION.q1,GRIDS)
    SOLUTION.itp_EVhh=CreateInterpolation_ValueFunctions(SOLUTION.EVhh,IsDefault,GRIDS)

    @unpack Decentralized = par
    if Decentralized
        UpdateHH_Obj_Grim!(HH_OBJ,HH_OBJ_NO_RULE,SOLUTION,SOLUTION_NO_RULE,GRIDS,par)
    end
    return nothing
end

function UpdateSolution_Closed!(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Closed policies and values
    UpdateDefault_PLA!(SOLUTION,GRIDS,par)

    #Closed expectations
    for I in CartesianIndices(SOLUTION.EVD)
        (k_ind,z_ind)=Tuple(I)
        foo_mat_D=CreateInterpolation_ForExpectations(SOLUTION.VD[k_ind,:],GRIDS)
        SOLUTION.EVD[I]=Expectation_over_zprime(foo_mat_D,z_ind,GRIDS)
    end

    #Compute expectation interpolations
    IsDefault=true
    SOLUTION.itp_EVD=CreateInterpolation_ValueFunctions(SOLUTION.EVD,IsDefault,GRIDS)
    return nothing
end

###############################################################################
#Value Function Iteration
###############################################################################
function ComputeDistance_q(SOLUTION_CURRENT::Solution,SOLUTION_NEXT::Solution,par::Pars)
    @unpack Tol_q = par
    dst_q, Ix=findmax(abs.(SOLUTION_CURRENT.q1 .- SOLUTION_NEXT.q1))
    NotConv=sum(abs.(SOLUTION_CURRENT.q1 .- SOLUTION_NEXT.q1) .> Tol_q)
    NotConvPct=100.0*NotConv/length(SOLUTION_CURRENT.q1)
    return round(dst_q,digits=7), round(NotConvPct,digits=2), Ix
end

function ComputeDistanceV(SOLUTION_CURRENT::Solution,SOLUTION_NEXT::Solution,par::Pars)
    dst_D=maximum(abs.(SOLUTION_CURRENT.VD .- SOLUTION_NEXT.VD))
    dst_V, Iv=findmax(abs.(SOLUTION_CURRENT.V .- SOLUTION_NEXT.V))

    NotConv=sum(abs.(SOLUTION_CURRENT.V .- SOLUTION_NEXT.V) .> par.Tol_V)
    NotConvPct=100.0*NotConv/length(SOLUTION_CURRENT.V)
    return round(abs(dst_D),digits=7), round(abs(dst_V),digits=7), Iv, round(NotConvPct,digits=2)
end

function SolveModel_VFI(PrintProg::Bool,GRIDS::Grids,par::Pars)
    @unpack Tol_q, Tol_V, cnt_max, Decentralized, ForeignInvestors = par
    if PrintProg
        println("Preparing solution guess")
    end
    SOLUTION_CURRENT=InitiateEmptySolution(GRIDS,par)
    SOLUTION_NEXT=deepcopy(SOLUTION_CURRENT)
    HH_OBJ=InitiateHH_Obj(SOLUTION_CURRENT,GRIDS,par)
    dst_V=1.0; dst_D=1.0; dst_P=1.0; NotConvPct_P=1.0; dst_q=1.0; NotConvPct=100.0
    cnt=0
    if PrintProg
        println("Starting VFI")
    end
    while cnt<cnt_max && (dst_V>Tol_V || dst_q>Tol_q)
        UpdateSolution!(HH_OBJ,SOLUTION_NEXT,GRIDS,par)
        dst_q, NotConvPct, Ix=ComputeDistance_q(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        dst_D, dst_P, Iv, NotConvPct_P=ComputeDistanceV(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        dst_V=max(dst_D,dst_P)
        cnt=cnt+1
        SOLUTION_CURRENT=deepcopy(SOLUTION_NEXT)
        if PrintProg
            println("cnt=$cnt, dst_D=$dst_D, dst_P=$dst_P at $Iv, $NotConvPct_P% of V not converged, dst_q=$dst_q")
        end
    end

    return Model(SOLUTION_NEXT,GRIDS,par)
end

function SolveModel_VFI_Grim(PrintProg::Bool,GRIDS::Grids,par::Pars,par_No_Rule::Pars)
    @unpack Tol_q, Tol_V, cnt_max, Decentralized, ForeignInvestors = par
    if PrintProg
        println("Preparing solution guess")
    end
    SOLUTION_NO_RULE=InitiateEmptySolution(GRIDS,par_No_Rule)
    SOLUTION_CURRENT=InitiateEmptySolution(GRIDS,par)
    SOLUTION_NEXT=deepcopy(SOLUTION_CURRENT)
    HH_OBJ_NO_RULE=InitiateHH_Obj(SOLUTION_NO_RULE,GRIDS,par_No_Rule)
    HH_OBJ=InitiateHH_Obj(SOLUTION_CURRENT,GRIDS,par)
    dst_V=1.0; dst_D=1.0; dst_P=1.0; NotConvPct_P=1.0; dst_q=1.0; NotConvPct=100.0
    cnt=0
    if PrintProg
        println("Starting VFI")
    end
    while cnt<cnt_max && (dst_V>Tol_V || dst_q>Tol_q)
        UpdateSolution!(HH_OBJ_NO_RULE,SOLUTION_NO_RULE,GRIDS,par_No_Rule)
        UpdateSolution_Grim!(HH_OBJ,HH_OBJ_NO_RULE,SOLUTION_NEXT,SOLUTION_NO_RULE,GRIDS,par)
        dst_q, NotConvPct, Ix=ComputeDistance_q(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        dst_D, dst_P, Iv, NotConvPct_P=ComputeDistanceV(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        dst_V=max(dst_D,dst_P)
        cnt=cnt+1
        SOLUTION_CURRENT=deepcopy(SOLUTION_NEXT)
        if PrintProg
            println("cnt=$cnt, dst_D=$dst_D, dst_P=$dst_P at $Iv, $NotConvPct_P% of V not converged, dst_q=$dst_q")
        end
    end

    return Model(SOLUTION_NEXT,GRIDS,par), Model(SOLUTION_NO_RULE,GRIDS,par_No_Rule)
end

function Model_FromSetup(setup_coulumn::Int64,SETUP_FILE::String)
    XX=readdlm(SETUP_FILE,',')
    NAME=convert(String,XX[1,setup_coulumn])
    VEC_PAR=XX[2:end,setup_coulumn]*1.0
    par, GRIDS=Setup_From_Vector(VEC_PAR)

    PrintProg=true; PrintAll=true
    MOD=SolveModel_VFI(PrintProg,GRIDS,par)

    return MOD, NAME
end

function Models_Welfare_Decomposition(FOLDER::String)
    benchmark_coulumn=2
    planner_column=3
    covenants_column=6

    MOD_Benchmark, xx, yy, zz=Unpack_All_Models_Case(benchmark_coulumn,FOLDER)
    MOD_Planner, xx, yy, zz=Unpack_All_Models_Case(planner_column,FOLDER)
    MOD_Covenants, xx, yy, zz=Unpack_All_Models_Case(covenants_column,FOLDER)

    par_Efficient=Pars(MOD_Benchmark.par,Covenants=true,Decentralized=false)

    PrintProg=true; PrintAll=true
    MOD_Efficient=SolveModel_VFI(PrintProg,MOD_Benchmark.GRIDS,par_Efficient)

    return MOD_Benchmark, MOD_Planner, MOD_Covenants, MOD_Efficient
end

function Save_Model_FromSetup(setup_coulumn::Int64,SETUP_FILE::String)
    MOD, NAME=Model_FromSetup(setup_coulumn,SETUP_FILE)
    SaveModel_Vector(NAME,MOD)
    return nothing
end

function Model_NoCommitment(Strict::Bool,Only_DL::Bool,setup_coulumn::Int64,SETUP_FILE::String)
    SETUP_FILE="$FOLDER\\Setup.csv"
    XX=readdlm(SETUP_FILE,',')
    NAME_CASE=convert(String,XX[1,case_coulumn])

    MAT=readdlm("$FOLDER\\Models $NAME_CASE.csv",',')
    MOD0=UnpackModel_Vector(MAT[:,1])
    par_No_Rule=Pars(MOD0.par,Strict=Strict)
    GRIDS=deepcopy(MOD0.GRIDS)
    if Only_DL
        MOD_DL=UnpackModel_Vector(MAT[:,2])
        par=Pars(MOD_DL.par,Strict=Strict)
    else
        MOD_FRpair=UnpackModel_Vector(MAT[:,3])
        par=Pars(MOD_FRpair.par,Strict=Strict)
    end

    PrintProg=true; PrintAll=true
    MOD_GRIM, MOD_NO_RULE=SolveModel_VFI_Grim(PrintProg,GRIDS,par,par_No_Rule)

    return MOD_GRIM, MOD_NO_RULE
end

################################################################################
### Functions for simulations
################################################################################
@with_kw mutable struct Paths
    #Paths of shocks
    z::Array{Float64,1}
    zP::Array{Float64,1}
    Readmission::Array{Float64,1}

    #Paths of chosen states
    Def::Array{Float64,1}
    K::Array{Float64,1}
    B::Array{Float64,1}

    #Path of relevant variables
    Spreads::Array{Float64,1}
    GDP::Array{Float64,1}
    Cons::Array{Float64,1}
    Inv::Array{Float64,1}
    AdjCost::Array{Float64,1}
    TB::Array{Float64,1}
    CA::Array{Float64,1}
end

function InitiateEmptyPaths(T::Int64)
    #Initiate with zeros to facilitate averages
    #Paths of shocks
    f1=zeros(Float64,T)
    f2=zeros(Float64,T)
    f3=zeros(Float64,T)
    f4=zeros(Float64,T)
    f5=zeros(Float64,T)
    f6=zeros(Float64,T)
    f7=zeros(Float64,T)
    f8=zeros(Float64,T)
    f9=zeros(Float64,T)
    f10=zeros(Float64,T)
    f11=zeros(Float64,T)
    f12=zeros(Float64,T)
    f13=zeros(Float64,T)
    return Paths(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13)
end

function Simulate_z_shocks!(z0::Float64,PATHS::Paths,par::Pars)
    @unpack μ_z, ρ_z, dist_ϵz = par
    T=length(PATHS.z)
    ϵz_TS=rand(dist_ϵz,T)

    PATHS.z[1]=z0
    for t in 2:T
        PATHS.z[t]=exp((1.0-ρ_z)*log(μ_z)+ρ_z*log(PATHS.z[t-1])+ϵz_TS[t])
    end
    return nothing
end

function Simulate_Readmission!(PATHS::Paths,par::Pars)
    T=length(PATHS.Readmission)
    PATHS.Readmission .= rand(T)
end

function ComputeSpreads(z::Float64,kprime::Float64,bprime::Float64,
                        SOLUTION::Solution,par::Pars)
    @unpack itp_q1 = SOLUTION
    qq=max(itp_q1(bprime,kprime,z),1e-3)
    return ComputeSpreadWithQ(qq,par)
end

function Simulate_EndogenousVariables!(Def0::Float64,K0::Float64,B0::Float64,PATHS::Paths,SOLUTION::Solution,par::Pars)
    @unpack δ = par
    @unpack itp_VP, itp_VD, itp_kprime_D, itp_kprime, itp_bprime = SOLUTION
    #Allocate floats once
    z=0.0; zD=0.0; k=0.0; b=0.0; kprime=0.0; bprime=0.0; y=0.0; Tr=0.0

    #Must have already simulated productivity and readmission shocks
    T=length(PATHS.B)
    PATHS.K[1]=K0
    PATHS.B[1]=B0
    Def_1=Def0 #Default state from previous period
    for t in 1:T
        z=PATHS.z[t]
        zD=zDefault(z,par)
        k=PATHS.K[t]
        b=PATHS.B[t]

        if Def_1==1.0
            #Coming from default, check if reentry
            # if PATHS.Readmission[t]<=par.θ
            if rand()<=par.θ
                #Reentry
                Default=false
                x=State(Default,z,k,0.0)
                PATHS.zP[t]=z
                PATHS.B[t]=0.0
                b=0.0
                kprime=itp_kprime(b,k,z)
                bprime=itp_bprime(b,k,z)
                y=FinalOutput(z,k,par)
                Tr=Calculate_Tr(x,kprime,bprime,SOLUTION,par)
                PATHS.Def[t]=0.0
                PATHS.Spreads[t]=ComputeSpreads(z,kprime,bprime,SOLUTION,par)
            else
                #Remain in default
                Default=true
                x=State(Default,z,k,0.0)
                PATHS.zP[t]=zD
                kprime=itp_kprime_D(k,z)
                bprime=b
                y=FinalOutput(zD,k,par)
                Tr=0.0
                PATHS.Def[t]=1.0
                PATHS.Spreads[t]=0.0
            end
        else
            #Coming from repayment, check if would default today
            if itp_VD(k,z)>itp_VP(b,k,z)
                #Default
                Default=true
                x=State(Default,z,k,0.0)
                PATHS.zP[t]=zD
                kprime=itp_kprime_D(k,z)
                bprime=b
                y=FinalOutput(zD,k,par)
                Tr=0.0
                PATHS.Def[t]=1.0
                PATHS.Spreads[t]=0.0
            else
                #Repayment
                Default=false
                x=State(Default,z,k,b)
                PATHS.zP[t]=z
                kprime=itp_kprime(b,k,z)
                bprime=itp_bprime(b,k,z)
                y=FinalOutput(z,k,par)
                Tr=Calculate_Tr(x,kprime,bprime,SOLUTION,par)
                PATHS.Def[t]=0.0
                PATHS.Spreads[t]=ComputeSpreads(z,kprime,bprime,SOLUTION,par)
            end
        end
        PATHS.GDP[t]=y
        PATHS.AdjCost[t]=CapitalAdjustment(kprime,k,par)
        PATHS.Inv[t]=kprime-(1-δ)*k+PATHS.AdjCost[t]
        PATHS.Cons[t]=Evaluate_cons_state(x,kprime,Tr,par)
        PATHS.TB[t]=PATHS.GDP[t]-PATHS.Cons[t]-PATHS.Inv[t]
        PATHS.CA[t]=-(bprime-b)

        Def_1=PATHS.Def[t]
        if t<T
            PATHS.K[t+1]=kprime
            PATHS.B[t+1]=bprime
        end
    end

    return nothing
end

function Simulate_Paths(T::Int64,Def0::Float64,z0::Float64,K0::Float64,B0::Float64,MODEL::Model)
    @unpack SOLUTION, par = MODEL
    PATHS=InitiateEmptyPaths(T)
    Simulate_z_shocks!(z0,PATHS,par)
    Simulate_Readmission!(PATHS,par)
    Simulate_EndogenousVariables!(Def0,K0,B0,PATHS,SOLUTION,par)

    return PATHS
end

function Fill_Path_Simulation!(PATHS::Paths,Def0::Float64,z0::Float64,K0::Float64,B0::Float64,MODEL::Model)
    @unpack SOLUTION, par = MODEL
    Simulate_z_shocks!(z0,PATHS,par)
    Simulate_Readmission!(PATHS,par)
    Simulate_EndogenousVariables!(Def0,K0,B0,PATHS,SOLUTION,par)
    return nothing
end

function ExtractFromLongPaths!(t0::Int64,t1::Int64,PATHS::Paths,PATHS_long::Paths)
    PATHS.z .= PATHS_long.z[t0:t1]
    PATHS.zP .= PATHS_long.zP[t0:t1]
    PATHS.Readmission .= PATHS_long.Readmission[t0:t1]

    #Paths of chosen states
    PATHS.Def .= PATHS_long.Def[t0:t1]
    PATHS.K .= PATHS_long.K[t0:t1]
    PATHS.B .= PATHS_long.B[t0:t1]

    #Path of relevant variables
    PATHS.Spreads .= PATHS_long.Spreads[t0:t1]
    PATHS.GDP .= PATHS_long.GDP[t0:t1]
    PATHS.Cons .= PATHS_long.Cons[t0:t1]
    PATHS.Inv .= PATHS_long.Inv[t0:t1]
    PATHS.AdjCost .= PATHS_long.AdjCost[t0:t1]
    PATHS.TB .= PATHS_long.TB[t0:t1]
    PATHS.CA .= PATHS_long.CA[t0:t1]
    return nothing
end

function Simulate_Paths_Ergodic(T::Int64,MODEL::Model)
    @unpack SOLUTION, par = MODEL
    Random.seed!(1234)

    Tlong=MODEL.par.drp+T
    PATHS_long=InitiateEmptyPaths(Tlong)
    PATHS=InitiateEmptyPaths(T)
    t0=MODEL.par.drp+1; t1=Tlong

    Def0=0.0; z0=1.0; K0=0.5*(par.klow+par.khigh); B0=0.0
    Fill_Path_Simulation!(PATHS_long,Def0,z0,K0,B0,MODEL)
    ExtractFromLongPaths!(t0,t1,PATHS,PATHS_long)

    return PATHS
end

@with_kw mutable struct Moments
    #Initiate them at 0.0 to facilitate average across samples
    #Default, spreads, and Debt
    DefaultPr::Float64 = 0.0
    MeanSpreads::Float64 = 0.0
    StdSpreads::Float64 = 0.0
    #Stocks
    Debt_GDP::Float64 = 0.0
    k_GDP::Float64 = 0.0
    #Volatilities
    σ_GDP::Float64 = 0.0
    σ_con::Float64 = 0.0
    σ_inv::Float64 = 0.0
    σ_TB_y::Float64 = 0.0
    #Cyclicality
    Corr_con_GDP::Float64 = 0.0
    Corr_inv_GDP::Float64 = 0.0
    Corr_Spreads_GDP::Float64 = 0.0
    Corr_TB_GDP::Float64 = 0.0
end

function hp_filter(y::Vector{Float64}, lambda::Float64)
    #Returns trend component
    n = length(y)
    @assert n >= 4

    diag2 = lambda*ones(n-2)
    diag1 = [ -2lambda; -4lambda*ones(n-3); -2lambda ]
    diag0 = [ 1+lambda; 1+5lambda; (1+6lambda)*ones(n-4); 1+5lambda; 1+lambda ]

    #D = spdiagm((diag2, diag1, diag0, diag1, diag2), (-2,-1,0,1,2))
    D = spdiagm(-2 => diag2, -1 => diag1, 0 => diag0, 1 => diag1, 2 => diag2)

    D\y
end

function MomentsIntoStructure!(t0::Int64,t1::Int64,PATHS::Paths,PATHS_long::Paths,MOM::Moments,MOD::Model)
    @unpack par = MOD
    #Will fill values into structures PATHS_long, PATHS, and MOM
    Def0=0.0; z0=1.0; K0=0.5*(par.klow+par.khigh); B0=0.0
    Fill_Path_Simulation!(PATHS_long,Def0,z0,K0,B0,MOD)
    ExtractFromLongPaths!(t0,t1,PATHS,PATHS_long)

    #Compute default probability
    Def_Ev=0.0
    for t in 2:length(PATHS_long.Def)
        if PATHS_long.Def[t]==1.0 && PATHS_long.Def[t-1]==0.0
            Def_Ev=Def_Ev+1
        end
    end
    pr_q=Def_Ev/length(PATHS_long.Def)
    pr_ndq=1-pr_q
    pr_ndy=pr_ndq^4
    MOM.DefaultPr=100*(1-pr_ndy)

    #Compute other easy moments
    if sum(PATHS.Def .== 0.0)==0.0
        MOM.MeanSpreads=mean(PATHS.Spreads)
        MOM.StdSpreads=std(PATHS.Spreads)
        MOM.Debt_GDP=mean(100 .* (PATHS.B ./ (4 .* PATHS.GDP)))
    else
        MOM.MeanSpreads=sum((PATHS.Spreads) .* (PATHS.Def .== 0.0))/sum(PATHS.Def .== 0.0)
        VarSpr=sum(((PATHS.Spreads .- MOM.MeanSpreads) .^ 2) .* (PATHS.Def .== 0.0))/sum(PATHS.Def .== 0.0)
        MOM.StdSpreads=sqrt(VarSpr)
        MOM.Debt_GDP=sum((100 .* (PATHS.B ./ (4 .* PATHS.GDP))) .* (PATHS.Def .== 0.0))/sum(PATHS.Def .== 0.0)
    end
    #Compute stocks
    MOM.k_GDP=mean(PATHS.K)

    ###Hpfiltering
    #GDP
    log_GDP=log.(abs.(PATHS.GDP))
    GDP_trend=hp_filter(log_GDP,par.HPFilter_Par)
    GDP_cyc=100.0*(log_GDP .- GDP_trend)
    #Consumption
    log_con=log.(abs.(PATHS.Cons))
    con_trend=hp_filter(log_con,par.HPFilter_Par)
    con_cyc=100.0*(log_con .- con_trend)
    #Investment
    inv_TS=PATHS.Inv
    log_inv=log.(abs.(inv_TS))
    inv_trend=hp_filter(log_inv,par.HPFilter_Par)
    inv_cyc=100.0*(log_inv .- inv_trend)
    #Trade balance
    TB_y_TS=100 * PATHS.TB ./ PATHS.GDP
    #Volatilities
    MOM.σ_GDP=std(GDP_cyc)
    MOM.σ_con=std(con_cyc)
    MOM.σ_inv=std(inv_cyc)
    MOM.σ_TB_y=std(TB_y_TS)
    #Correlations with GDP
    MOM.Corr_con_GDP=cor(GDP_cyc,con_cyc)
    MOM.Corr_inv_GDP=cor(GDP_cyc,inv_cyc)
    MOM.Corr_Spreads_GDP=cor(GDP_cyc .* (PATHS.Def .== 0.0),PATHS.Spreads .* (PATHS.Def .== 0.0))
    MOM.Corr_TB_GDP=cor(GDP_cyc .* (PATHS.Def .== 0.0),TB_y_TS .* (PATHS.Def .== 0.0))

    if isnan(MOM.Corr_TB_GDP)
        println(PATHS.Readmission)
    end

    return nothing
end

function AverageMomentsManySamples(Tmom::Int64,NSamplesMoments::Int64,MOD::Model)
    T=MOD.par.drp+Tmom
    PATHS_long=InitiateEmptyPaths(T)
    PATHS=InitiateEmptyPaths(Tmom)
    t0=MOD.par.drp+1; t1=T

    Random.seed!(1234)
    MOMENTS=Moments(); MOMS=Moments()
    for i in 1:NSamplesMoments
        MomentsIntoStructure!(t0,t1,PATHS,PATHS_long,MOMS,MOD)
        #Default, spreads, and Debt
        MOMENTS.DefaultPr=MOMENTS.DefaultPr+MOMS.DefaultPr/NSamplesMoments
        MOMENTS.MeanSpreads=MOMENTS.MeanSpreads+MOMS.MeanSpreads/NSamplesMoments
        MOMENTS.StdSpreads=MOMENTS.StdSpreads+MOMS.StdSpreads/NSamplesMoments
        #Stocks
        MOMENTS.Debt_GDP=MOMENTS.Debt_GDP+MOMS.Debt_GDP/NSamplesMoments
        MOMENTS.k_GDP=MOMENTS.k_GDP+MOMS.k_GDP/NSamplesMoments
        #Volatilities
        MOMENTS.σ_GDP=MOMENTS.σ_GDP+MOMS.σ_GDP/NSamplesMoments
        MOMENTS.σ_con=MOMENTS.σ_con+MOMS.σ_con/NSamplesMoments
        MOMENTS.σ_inv=MOMENTS.σ_inv+MOMS.σ_inv/NSamplesMoments
        MOMENTS.σ_TB_y=MOMENTS.σ_TB_y+MOMS.σ_TB_y/NSamplesMoments
        #Cyclicality
        MOMENTS.Corr_con_GDP=MOMENTS.Corr_con_GDP+MOMS.Corr_con_GDP/NSamplesMoments
        MOMENTS.Corr_inv_GDP=MOMENTS.Corr_inv_GDP+MOMS.Corr_inv_GDP/NSamplesMoments
        MOMENTS.Corr_Spreads_GDP=MOMENTS.Corr_Spreads_GDP+MOMS.Corr_Spreads_GDP/NSamplesMoments
        MOMENTS.Corr_TB_GDP=MOMENTS.Corr_TB_GDP+MOMS.Corr_TB_GDP/NSamplesMoments
    end
    return MOMENTS
end

###############################################################################
#Functions to calibrate capital adjustment cost
###############################################################################
function SolveClosed_VFI(PrintProg::Bool,PrintAll::Bool,GRIDS::Grids,par::Pars)
    @unpack Tol_q, Tol_V, cnt_max = par
    parClosed=Pars(par,θ=0.0,d0=0.0,d1=0.0)
    if PrintProg
        println("Preparing solution guess")
    end
    SOLUTION_CURRENT=InitiateEmptySolution(GRIDS,parClosed)
    SOLUTION_NEXT=deepcopy(SOLUTION_CURRENT)
    dst_V=1.0
    dst_D=1.0
    dst_P=1.0
    NotConvPct_P=1.0
    dst_q=1.0
    NotConvPct=100.0
    cnt=0
    if PrintProg
        println("Starting VFI")
    end
    while cnt<cnt_max && (dst_V>Tol_V || dst_q>Tol_q)
        UpdateSolution_Closed!(SOLUTION_NEXT,GRIDS,parClosed)
        dst_q, NotConvPct, Ix=ComputeDistance_q(SOLUTION_CURRENT,SOLUTION_NEXT,parClosed)
        dst_D, dst_P, Iv, NotConvPct_P=ComputeDistanceV(SOLUTION_CURRENT,SOLUTION_NEXT,parClosed)
        dst_V=max(dst_D,dst_P)
        cnt=cnt+1
        SOLUTION_CURRENT=deepcopy(SOLUTION_NEXT)
        if PrintProg
            if PrintAll
                println("cnt=$cnt, dst_D=$dst_D, dst_P=$dst_P at $Iv, $NotConvPct_P% of V not converged, dst_q=$dst_q")
            else
                if mod(cnt,100)==0
                    println("cnt=$cnt, dst_D=$dst_D, dst_P=$dst_P at $Iv, $NotConvPct_P% of V not converged, dst_q=$dst_q")
                end
            end
        end
    end
    if PrintProg
        println("cnt=$cnt, dst_D=$dst_D, dst_P=$dst_P, $NotConvPct_P% of V not converged, dst_q=$dst_q")
    end
    return Model(SOLUTION_NEXT,GRIDS,parClosed)
end

function Evaluate_AdjCost(φtry::Float64,GRIDS::Grids,par::Pars)
    par_φ=Pars(par,φ=φtry)
    MOD_CLOSED=SolveClosed_VFI(false,false,GRIDS,par_φ)
    MOM_CLOSED=AverageMomentsManySamples(par.Tmom,par.NSamplesMoments,MOD_CLOSED)
    σi_σy=MOM_CLOSED.σ_inv/MOM_CLOSED.σ_GDP
    println("φtry=$φtry, σi_σy=$σi_σy")
    return σi_σy
end

function Calibrate_AdjCost(GRIDS::Grids,par::Pars)
    φlow=0.0
    foo(φtry::Float64)=Evaluate_AdjCost(φtry,GRIDS,par)-2.0
    φhigh=15.0
    while foo(φhigh)>0.0
        φhigh=2*φhigh
    end
    return MyBisection(foo,φlow,φhigh)
end

###############################################################################
#Functions to calibrate decentralized
###############################################################################
function SolveCalibration_DEC(GRIDS::Grids,par::Pars)
    @unpack Tmom, NSamplesMoments = par
    PrintProg=true
    MOD=SolveModel_VFI(PrintProg,GRIDS,par)
    return AverageMomentsManySamples(Tmom,NSamplesMoments,MOD)
end

function CheckMomentsForTry_DEC(PARS_TRY::Array{Float64,1},GRIDS::Grids,par::Pars)
    knk=PARS_TRY[1]
    d1=PARS_TRY[2]
    d0=-d1*knk
    # φ=PARS_TRY[3]
    parTry=Pars(par,d0=d0,d1=d1)

    MOM=SolveCalibration_DEC(GRIDS,parTry)

    MOM_VEC=Array{Float64,1}(undef,13)

    MOM_VEC[1]=MOM.DefaultPr
    MOM_VEC[2]=MOM.MeanSpreads
    MOM_VEC[3]=MOM.StdSpreads
    MOM_VEC[4]=MOM.Debt_GDP
    MOM_VEC[5]=MOM.k_GDP
    MOM_VEC[6]=MOM.σ_GDP
    MOM_VEC[7]=MOM.σ_con/MOM.σ_GDP
    MOM_VEC[8]=MOM.σ_inv/MOM.σ_GDP
    MOM_VEC[9]=MOM.σ_TB_y
    MOM_VEC[10]=MOM.Corr_con_GDP
    MOM_VEC[11]=MOM.Corr_inv_GDP
    MOM_VEC[12]=MOM.Corr_Spreads_GDP
    MOM_VEC[13]=MOM.Corr_TB_GDP

    return MOM_VEC
end

function CalibrateMatchingMoments_DEC(N::Int64,lb::Vector{Float64},ub::Vector{Float64})
    XX=readdlm("Setup.csv",',')
    VEC_PAR=XX[2:end,2]*1.0
    par, GRIDS=Setup_From_Vector(VEC_PAR)

    #Here N is the number of grid points for each parameter
    #Total computations will be N^2
    nn=N^2
    MAT_TRY=Array{Float64,2}(undef,nn,2)
    gr_knk=collect(range(lb[1],stop=ub[1],length=N))
    gr_d1=collect(range(lb[2],stop=ub[2],length=N))
    # gr_φ=collect(range(lb[3],stop=ub[3],length=N))
    i=1
    for knk_ind in 1:N
        for d1_ind in 1:N
            MAT_TRY[i,1]=gr_knk[knk_ind]
            MAT_TRY[i,2]=gr_d1[d1_ind]
            i=i+1
        end
    end

    #Loop paralelly over all parameter tries
    #There are 13 moments plus dst plus dst_ar, columns should be 13+2 parameters
    PARAMETER_MOMENTS_MATRIX=SharedArray{Float64,2}(nn,15)
    COL_NAMES=["knk" "d1" "DefPr" "Av spreads" "Std spreads" "debt_GDP" "k_GDP" "vol_y" "vol_c/vol_y" "vol_i/vol_y" "vol_tb" "cor(c,y)" "cor(i,y)" "cor(r-r*,y)" "cor(tb/y,y)"]
    # COL_NAMES=["knk" "d1" "phi" "DefPr" "Av spreads" "Std spreads" "debt_GDP" "k_GDP" "vol_y" "vol_c/vol_y" "vol_i/vol_y" "vol_tb" "cor(c,y)" "cor(i,y)" "cor(r-r*,y)" "cor(tb/y,y)"]
    @sync @distributed for i in 1:nn
        PARAMETER_MOMENTS_MATRIX[i,1:2]=MAT_TRY[i,:]
        PARAMETER_MOMENTS_MATRIX[i,3:end]=CheckMomentsForTry_DEC(MAT_TRY[i,:],GRIDS,par)
        MAT=[COL_NAMES; PARAMETER_MOMENTS_MATRIX]
        writedlm("TriedCalibrations.csv",MAT,',')
        println("Done with $i of $nn")
    end
    return nothing
end

function ReadCalibrationOutput(FILE::String,Nknk::Int64,Nd1::Int64)
    XX=readdlm(FILE,',')
    MAT=XX[2:end,:]*1.0

    #knk grid
    knk_low=minimum(MAT[:,1])
    knk_high=maximum(MAT[:,1])
    range_knk=range(knk_low,stop=knk_high,length=Nknk)
    GR_knk=collect(range_knk)

    #d1 grid
    d1_low=minimum(MAT[:,2])
    d1_high=maximum(MAT[:,2])
    range_d1=range(d1_low,stop=d1_high,length=Nd1)
    GR_d1=collect(range_d1)

    #Matrix of moments
    II=(Nd1,Nknk)
    MAT_SPREADS=reshape(MAT[:,4],II)
    MAT_DEBT=reshape(MAT[:,6],II)

    #Interpolation objects
    # ORDER_PARAMETERS=Linear()
    ORDER_PARAMETERS=Cubic(Line(OnGrid()))
    INT_DIMENSIONS=(BSpline(ORDER_PARAMETERS),BSpline(ORDER_PARAMETERS))
    itp_spreads=extrapolate(Interpolations.scale(interpolate(MAT_SPREADS,INT_DIMENSIONS),range_d1,range_knk),Interpolations.Line())
    itp_debt=extrapolate(Interpolations.scale(interpolate(MAT_DEBT,INT_DIMENSIONS),range_d1,range_knk),Interpolations.Line())

    return MAT_SPREADS, MAT_DEBT, GR_knk, GR_d1, itp_spreads, itp_debt
end

function Calibrate_knk_d1(Target_spreads::Float64,Target_debt::Float64,FILE::String,Nknk::Int64,Nd1::Int64)
    MAT_SPREADS, MAT_DEBT, GR_knk, GR_d1, itp_spreads, itp_debt=ReadCalibrationOutput(FILE,Nknk,Nd1)

    function ObjectiveFunction(X_REAL::Array{Float64,1})
        d1=TransformIntoBounds(X_REAL[1],GR_d1[1],GR_d1[end])
        knk=TransformIntoBounds(X_REAL[2],GR_knk[1],GR_knk[end])
        spr_err=itp_spreads(d1,knk)-Target_spreads
        debt_err=itp_debt(d1,knk)-Target_debt
        return (spr_err^2)+(debt_err^2)
    end
    d1_0=0.5*(GR_d1[1]+GR_d1[end])
    knk_0=0.5*(GR_knk[1]+GR_knk[end])
    X0_BOUNDS=[d1_0; knk_0]

    X0=Array{Float64,1}(undef,2)
    X0[1]=TransformIntoReals(X0_BOUNDS[1],GR_d1[1],GR_d1[end])
    X0[2]=TransformIntoReals(X0_BOUNDS[2],GR_knk[1],GR_knk[end])

    res=optimize(ObjectiveFunction,X0,NelderMead())
    d1_star=TransformIntoBounds(Optim.minimizer(res)[1],GR_d1[1],GR_d1[end])
    knk_star=TransformIntoBounds(Optim.minimizer(res)[2],GR_knk[1],GR_knk[end])

    return d1_star, knk_star, itp_spreads, itp_debt
end

###############################################################################
#Functions to compute welfare
###############################################################################
function WelfareGains_Val(v0::Float64,v1::Float64,par::Pars)
    @unpack σ = par
    return 100*(((v1/v0)^(1/(1-σ)))-1)
end

function AverageWelfareGains(N::Int64,MOD0::Model,MOD1::Model)
    @unpack par = MOD0

    Random.seed!(1234)
    PATHS_long=InitiateEmptyPaths(par.drp+N)
    PATHS=InitiateEmptyPaths(N)

    t0=par.drp+1; t1=par.drp+N
    Def0=0.0; z0=1.0; K0=0.5*(par.klow+par.khigh); B0=0.0
    Fill_Path_Simulation!(PATHS_long,Def0,z0,K0,B0,MOD0)
    ExtractFromLongPaths!(t0,t1,PATHS,PATHS_long)

    wg=0.0
    for i in 1:N
        z=PATHS.z[i]; k=PATHS.K[i]; b=PATHS.B[i]

        v0p=MOD0.SOLUTION.itp_VP(b,k,z)
        v0d=MOD0.SOLUTION.itp_VD(k,z)
        if v0d>v0p
            #Government defaults
            v0=MOD0.SOLUTION.itp_VDhh(k,z)
        else
            #Government repays
            v0=MOD0.SOLUTION.itp_VPhh(b,k,z)
        end

        v1p=MOD1.SOLUTION.itp_VP(b,k,z)
        v1d=MOD1.SOLUTION.itp_VD(k,z)
        if v1d>v1p
            #Government defaults
            v1=MOD1.SOLUTION.itp_VDhh(k,z)
        else
            #Government repays
            v1=MOD1.SOLUTION.itp_VPhh(b,k,z)
        end

        wg=wg+WelfareGains_Val(v0,v1,par)/N
    end
    return wg
end

function AverageWelfareGains_0(N::Int64,MOD0::Model,MOD1::Model)
    @unpack par = MOD0

    Random.seed!(1234)
    PATHS_long=InitiateEmptyPaths(par.drp+N)
    PATHS=InitiateEmptyPaths(N)

    t0=par.drp+1; t1=par.drp+N
    Def0=0.0; z0=1.0; K0=0.5*(par.klow+par.khigh); B0=0.0
    Fill_Path_Simulation!(PATHS_long,Def0,z0,K0,B0,MOD0)
    ExtractFromLongPaths!(t0,t1,PATHS,PATHS_long)

    wg=0.0
    for i in 1:N
        z=PATHS.z[i]; k=PATHS.K[i]; b=0.0#PATHS.B[i]

        v0p=MOD0.SOLUTION.itp_VP(b,k,z)
        v0d=MOD0.SOLUTION.itp_VD(k,z)
        if v0d>v0p
            #Government defaults
            v0=MOD0.SOLUTION.itp_VDhh(k,z)
        else
            #Government repays
            v0=MOD0.SOLUTION.itp_VPhh(b,k,z)
        end

        v1p=MOD1.SOLUTION.itp_VP(b,k,z)
        v1d=MOD1.SOLUTION.itp_VD(k,z)
        if v1d>v1p
            #Government defaults
            v1=MOD1.SOLUTION.itp_VDhh(k,z)
        else
            #Government repays
            v1=MOD1.SOLUTION.itp_VPhh(b,k,z)
        end

        wg=wg+WelfareGains_Val(v0,v1,par)/N
    end
    return wg
end

function WelfareGains_AvState(N::Int64,MOD_BEN::Model,MOD0::Model,MOD1::Model)
    @unpack par = MOD_BEN

    Random.seed!(1234)
    PATHS_long=InitiateEmptyPaths(par.drp+N)
    PATHS=InitiateEmptyPaths(N)

    t0=par.drp+1; t1=par.drp+N
    Def0=0.0; z0=1.0; K0=0.5*(par.klow+par.khigh); B0=0.0
    Fill_Path_Simulation!(PATHS_long,Def0,z0,K0,B0,MOD_BEN)
    ExtractFromLongPaths!(t0,t1,PATHS,PATHS_long)

    z=mean(PATHS.z)
    k=mean(PATHS.K)
    b=mean(PATHS.B)

    v0p=MOD0.SOLUTION.itp_VP(b,k,z)
    v0d=MOD0.SOLUTION.itp_VD(k,z)
    if v0d>v0p
        #Government defaults
        v0=MOD0.SOLUTION.itp_VDhh(k,z)
    else
        #Government repays
        v0=MOD0.SOLUTION.itp_VPhh(b,k,z)
    end

    v1p=MOD1.SOLUTION.itp_VP(b,k,z)
    v1d=MOD1.SOLUTION.itp_VD(k,z)
    if v1d>v1p
        #Government defaults
        v1=MOD1.SOLUTION.itp_VDhh(k,z)
    else
        #Government repays
        v1=MOD1.SOLUTION.itp_VPhh(b,k,z)
    end

    wg=WelfareGains_Val(v0,v1,par)
    return wg
end

###############################################################################
#Functions to find optimal fiscal rule
###############################################################################
function TestOneDebtLimit(DL::Float64,MOD0::Model)
    @unpack par, GRIDS = MOD0
    PrintProg=false
    par_fr=Pars(par,WithFR=true,OnlyBudgetBalance=false,FR=DL)
    MOD1=SolveModel_VFI(PrintProg,GRIDS,par_fr)

    Nwg=10000
    wg=AverageWelfareGains(Nwg,MOD0,MOD1)
    return wg, MOD1
end

function TestOneDeficitLimit(χ::Float64,MOD0::Model)
    @unpack par, GRIDS = MOD0
    PrintProg=false
    par_fr=Pars(par,WithFR=true,OnlyBudgetBalance=true,χ=χ)
    MOD1=SolveModel_VFI(PrintProg,GRIDS,par_fr)

    Nwg=10000
    wg=AverageWelfareGains(Nwg,MOD0,MOD1)
    return wg, MOD1
end

function Save_wg_dl(IsFirst::Bool,NAME_CASE::String,MAT_WG::SharedArray{Float64,2})
    (N_dl, cols)=size(MAT_WG)
    grDL=MAT_WG[:,1]
    wg_vec=MAT_WG[:,2]
    def_pr_vec=MAT_WG[:,3]
    VEC=vcat(NAME_CASE,N_dl)
    VEC=vcat(VEC,grDL)
    VEC=vcat(VEC,wg_vec)
    VEC=vcat(VEC,def_pr_vec)

    GAINS_FLE="WelfareGains_DL.csv"
    if IsFirst
        writedlm(GAINS_FLE,VEC,',')
    else
        XX=readdlm(GAINS_FLE,',')
        MAT=hcat(XX,VEC)
        writedlm(GAINS_FLE,MAT,',')
    end

    return nothing
end

function Save_wg_defl(IsFirst::Bool,NAME_CASE::String,MAT_WG::SharedArray{Float64,2})
    (N_dl, cols)=size(MAT_WG)
    grDL=MAT_WG[:,1]
    wg_vec=MAT_WG[:,2]
    def_pr_vec=MAT_WG[:,3]
    VEC=vcat(NAME_CASE,N_dl)
    VEC=vcat(VEC,grDL)
    VEC=vcat(VEC,wg_vec)
    VEC=vcat(VEC,def_pr_vec)

    GAINS_FLE="WelfareGains_DefL.csv"
    if IsFirst
        writedlm(GAINS_FLE,VEC,',')
    else
        XX=readdlm(GAINS_FLE,',')
        MAT=hcat(XX,VEC)
        writedlm(GAINS_FLE,MAT,',')
    end

    return nothing
end

function FindBestDebtLimit(grDL::Vector,MOD0::Model)
    MOD1=deepcopy(MOD0)

    #Allocate matrices to fill
    N_dl=length(grDL)
    #MAT_WG has columns: [debt_limit welfare_gains default_Pr]
    MAT_WG=SharedArray{Float64,2}(N_dl,3)

    VEC=Create_Model_Vector(MOD0)
    N_size_model=length(VEC)
    ALL_SOLS_DL=SharedArray{Float64,2}(N_size_model,N_dl)
    @sync @distributed for i_dl in 1:N_dl
        println("Trying debt limit, i=$i_dl of $N_dl")
        MAT_WG[i_dl,1]=grDL[i_dl]
        MAT_WG[i_dl,2], MOD1=TestOneDebtLimit(grDL[i_dl],MOD0)
        MOM1=AverageMomentsManySamples(MOD1.par.Tmom,MOD1.par.NSamplesMoments,MOD1)
        MAT_WG[i_dl,3]=MOM1.DefaultPr
        ALL_SOLS_DL[:,i_dl]=Create_Model_Vector(MOD1)
    end

    #Find model with best debt limit
    wg_vec=MAT_WG[:,2]
    wg, i_best=findmax(wg_vec)
    VEC_BEST=ALL_SOLS_DL[:,i_best]

    return UnpackModel_Vector(VEC_BEST), MAT_WG
end

function FindBestDeficitRule(grχ::Vector,MOD0::Model)
    MOD1=deepcopy(MOD0)

    #Allocate matrices to fill
    N_dl=length(grχ)
    #MAT_WG has columns: [debt_limit welfare_gains default_Pr]
    MAT_WG=SharedArray{Float64,2}(N_dl,3)

    VEC=Create_Model_Vector(MOD0)
    N_size_model=length(VEC)
    ALL_SOLS_DL=SharedArray{Float64,2}(N_size_model,N_dl)
    @sync @distributed for i_dl in 1:N_dl
        println("Trying deficit limit, i=$i_dl of $N_dl")
        MAT_WG[i_dl,1]=grχ[i_dl]
        MAT_WG[i_dl,2], MOD1=TestOneDeficitLimit(grχ[i_dl],MOD0)
        MOM1=AverageMomentsManySamples(MOD1.par.Tmom,MOD1.par.NSamplesMoments,MOD1)
        MAT_WG[i_dl,3]=MOM1.DefaultPr
        ALL_SOLS_DL[:,i_dl]=Create_Model_Vector(MOD1)
    end

    #Find model with best debt limit
    wg_vec=MAT_WG[:,2]
    wg, i_best=findmax(wg_vec)
    VEC_BEST=ALL_SOLS_DL[:,i_best]

    return UnpackModel_Vector(VEC_BEST), MAT_WG
end

function TestOnePair_DL_χ(χ::Float64,DL::Float64,MOD0::Model)
    @unpack par, GRIDS = MOD0
    PrintProg=false
    par_fr=Pars(par,WithFR=true,OnlyBudgetBalance=false,FR=DL,χ=χ)
    MOD1=SolveModel_VFI(PrintProg,GRIDS,par_fr)

    Nwg=10000
    wg=AverageWelfareGains(Nwg,MOD0,MOD1)
    return wg, MOD1
end

function Save_wg_pair(NAME_CASE::String,grχ::Vector,grDL::Vector,MAT_WG::SharedArray{Float64,2})
    N_χ=length(grχ)
    N_dl=length(grDL)
    wg_vec=reshape(MAT_WG,(:))
    VEC=vcat(N_χ,N_dl)
    VEC=vcat(VEC,grχ)
    VEC=vcat(VEC,grDL)
    VEC=vcat(VEC,wg_vec)

    writedlm("WelfareGains_Pairs_$NAME_CASE.csv",VEC,',')
    return nothing
end

function TestManyPairs_DL_χ(grχ::Vector,grDL::Vector,MOD0::Model)
    MOD1=deepcopy(MOD0)

    #Allocate matrices to fill
    N_χ=length(grχ)
    N_dl=length(grDL)
    MAT_WG=SharedArray{Float64,2}(N_dl,N_χ)

    VEC=Create_Model_Vector(MOD0)
    N_size_model=length(VEC)
    ALL_SOLS_FR=SharedArray{Float64,3}(N_dl,N_χ,N_size_model)
    @sync @distributed for I in CartesianIndices(MAT_WG)
        (i_dl,i_χ)=Tuple(I)
        println("Trying pair, i_dl=$i_dl, i_χ=$i_χ")
        MAT_WG[I], MOD1=TestOnePair_DL_χ(grχ[i_χ],grDL[i_dl],MOD0)
        ALL_SOLS_FR[i_dl,i_χ,:]=Create_Model_Vector(MOD1)
    end

    #Save model with best fiscal rule with adjustment
    wg, I_best=findmax(MAT_WG)
    VEC_BEST=ALL_SOLS_FR[I_best,:]

    return UnpackModel_Vector(VEC_BEST), MAT_WG
end

function Grids_to_Try_FromFile(setup_coulumn::Int64)
    SETUP_FILE="Setup_FR_grids.csv"
    XX=readdlm(SETUP_FILE,',')
    VEC=1.0*XX[2:end,setup_coulumn]

    dl_low=VEC[1]
    dl_high=VEC[2]
    χ_low=VEC[3]
    χ_high=VEC[4]
    N_DL=convert(Int64,VEC[5])
    N_χ=convert(Int64,VEC[6])

    grDL=collect(range(dl_low,stop=dl_high,length=N_DL))
    grχ=collect(range(χ_low,stop=χ_high,length=N_χ))

    return grDL, grχ
end

function TestAll_DebtLimits_And_FiscalRules(DoPairs::Bool,IsFirst::Bool,setup_coulumn::Int64,SETUP_FILE::String)
    grDL, grχ=Grids_to_Try_FromFile(setup_coulumn)

    MOD0, NAME_CASE=Model_FromSetup(setup_coulumn,SETUP_FILE)
    MOD_DL, MAT_WG_DL=FindBestDebtLimit(grDL,MOD0)
    if DoPairs
        MOD_FRpair, MAT_WG_FRpair=TestManyPairs_DL_χ(grχ,grDL,MOD0)
    end
    MOD_DefL, MAT_WG_DefL=FindBestDeficitRule(grχ,MOD0)

    #Save models
    VEC0=Create_Model_Vector(MOD0)
    VEC_DL=Create_Model_Vector(MOD_DL)
    MAT=hcat(VEC0,VEC_DL)
    VEC_DefL=Create_Model_Vector(MOD_DefL)
    MAT=hcat(MAT,VEC_DefL)
    if DoPairs
        VEC_FRpair=Create_Model_Vector(MOD_FRpair)
        MAT=hcat(MAT,VEC_FRpair)
    end
    writedlm("Models $NAME_CASE.csv",MAT,',')

    #Save matrices of tried rules and their welfare gains
    Save_wg_dl(IsFirst,NAME_CASE,MAT_WG_DL)
    Save_wg_defl(IsFirst,NAME_CASE,MAT_WG_DefL)
    if DoPairs
        Save_wg_pair(NAME_CASE,grχ,grDL,MAT_WG_FRpair)
    end

    return nothing
end

###############################################################################
#Compute transfers to avoid deviations
###############################################################################
function Optimizer_One_Shot_Deviation(x::State,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack GR_z, GR_k, GR_b = GRIDS
    #Only makes sense in repayment
    #Do bprime unconstrained by the fiscal rule, just this one time
    kprime=SOLUTION.itp_kprime(x.b,x.k,x.z)
    blowOpt, bhighOpt=Search_bprime_unconstrained(x,kprime,SOLUTION,GRIDS,par)

    foo(bprime::Float64)=-Evaluate_ValueFunction(false,x,kprime,bprime,SOLUTION,par)
    res=optimize(foo,blowOpt,bhighOpt,GoldenSection())

    bprime=Optim.minimizer(res)
    Tr=Calculate_Tr(x,kprime,bprime,SOLUTION,par)
    cons=Evaluate_cons_state(x,kprime,Tr,par)
    return -Optim.minimum(res), kprime, bprime, cons
end

function Avoid_One_Shot_Deviation(x::State,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    v_dev, kprime_dev, bprime_dev, c_dev=Optimizer_One_Shot_Deviation(x,SOLUTION,GRIDS,par)

    @unpack itp_VP, itp_kprime, itp_bprime = SOLUTION
    v_no_dev=itp_VP(x.b,x.k,x.z)
    kprime=itp_kprime(x.b,x.k,x.z)
    bprime=itp_bprime(x.b,x.k,x.z)
    Tr=Calculate_Tr(x,kprime,bprime,SOLUTION,par)
    c_no_dev=Evaluate_cons_state(x,kprime,Tr,par)

    @unpack σ = par
    τ=((((1-σ)*(Utility(c_no_dev,par)+(v_dev-v_no_dev)))^(1/(1-σ)))/c_no_dev)-1
    return τ, v_dev, v_no_dev, c_dev, c_no_dev
end

function Time_Series_Transfers_to_avoid_deviation(TS::Paths,MODEL::Model)
    T=length(TS.z)
    TAU=Array{Float64,1}(undef,T)
    V_DEV=Array{Float64,1}(undef,T)
    V_NO_DEV=Array{Float64,1}(undef,T)
    C_DEV=Array{Float64,1}(undef,T)
    C_NO_DEV=Array{Float64,1}(undef,T)
    for t in 1:T
        if TS.Def[t]==0.0
            x=State(false,TS.z[t],TS.K[t],TS.B[t])
            τ, v_dev, v_no_dev, c_dev, c_no_dev=Avoid_One_Shot_Deviation(x,MODEL)
            TAU[t]=τ
            V_DEV[t]=v_dev
            V_NO_DEV[t]=v_no_dev
            C_DEV[t]=c_dev
            C_NO_DEV[t]=c_no_dev
        else
            TAU[t]=0.0
        end
    end
    return TAU, V_DEV, V_NO_DEV, C_DEV, C_NO_DEV
end

function Moments_Deviation_Transfers(MOD_DL::Model)
    T=10000
    TS_rule=Simulate_Paths_Ergodic(T,MOD_DL)
    TAU, V_DEV, V_NO_DEV, C_DEV, C_NO_DEV=Time_Series_Transfers_to_avoid_deviation(TS_rule,MOD_DL)
    av_τ=mean(100*TAU)
    std_τ=std(100*TAU)
    cor_τ_GDP=cor(TS_rule.GDP,TAU)
    cor_τ_Spreads=cor(TS_rule.Spreads,TAU)
    percentage_positive=100*sum(TAU .> 0.0)/T
    av_τ, std_τ, cor_τ_GDP, cor_τ_Spreads, percentage_positive
end

###############################################################################
#Compute average duration of rule enforced
###############################################################################
function DurationInRule(T::Int64,MOD::Model,MOD_FR::Model)
    @unpack par = MOD
    #Get initial ergodic starting point
    TS_LR=InitiateEmptyPaths(par.drp+1)
    Def00=0.0; z00=1.0; K00=0.5*(MOD.par.klow+MOD.par.khigh); B00=0.0
    Fill_Path_Simulation!(TS_LR,Def00,z00,K00,B00,MOD)
    Def0=TS_LR.Def[end]; z0=TS_LR.z[end]
    K0=TS_LR.K[end]; B0=TS_LR.B[end]

    #Simulate in fiscal rule regime
    TS_FR=InitiateEmptyPaths(T)
    Fill_Path_Simulation!(TS_FR,Def0,z0,K0,B0,MOD_FR)

    Duration=0
    for t in 1:T
        if TS_FR.Def[t]==0
            #Chose not to default today, check if want to deviate
            z=TS_FR.z[t]; k=TS_FR.K[t]; b=TS_FR.B[t]
            if MOD.SOLUTION.itp_VP(b,k,z)>=MOD_FR.SOLUTION.itp_VP(b,k,z)
                #Deviate from rule
                break
            else
                #Not deviated yet, keep counting
                Duration=Duration+1
            end
        else
            #In default, not deviated yet, keep counting
            Duration=Duration+1
        end
    end
    return Duration
end

function AverageDurationInRule(N::Int64,MOD::Model,MOD_FR::Model)
    Random.seed!(1234)
    T=20000
    Duration=0.0
    for i in 1:N
        Di=DurationInRule(T,MOD,MOD_FR)
        Duration=Duration+Di/N
    end
    return Duration
end

function DurationInGoodStanding(T::Int64,MOD_FR::Model)
    @unpack par = MOD_FR
    #Get initial ergodic starting point
    TS_LR=InitiateEmptyPaths(par.drp+1)
    Def00=0.0; z00=1.0; K00=0.5*(MOD_FR.par.klow+MOD_FR.par.khigh); B00=0.0
    Fill_Path_Simulation!(TS_LR,Def00,z00,K00,B00,MOD_FR)
    Def0=TS_LR.Def[end]; z0=TS_LR.z[end]
    K0=TS_LR.K[end]; B0=TS_LR.B[end]

    #Simulate in fiscal rule regime
    TS_FR=InitiateEmptyPaths(T)
    Fill_Path_Simulation!(TS_FR,Def0,z0,K0,B0,MOD_FR)

    Duration=0
    for t in 1:T
        #Check if want to default
        z=TS_FR.z[t]; k=TS_FR.K[t]; b=TS_FR.B[t]
        if MOD_FR.SOLUTION.itp_VD(k,z)>MOD_FR.SOLUTION.itp_VP(b,k,z)
            #Default
            break
        else
            #Not default
            Duration=Duration+1
        end
    end
    return Duration
end

function AverageDurationGoodStanding(N::Int64,MOD_FR::Model)
    Random.seed!(1234)
    T=20000
    Duration=0.0
    for i in 1:N
        Di=DurationInGoodStanding(T,MOD_FR)
        Duration=Duration+Di/N
    end
    return Duration
end

function FrequencyOfRuleBinding(T::Int64,MOD_FR::Model)
    @unpack par = MOD_FR
    @unpack FR, χ, γ = par
    Random.seed!(1234)
    #Get initial ergodic starting point
    PATHS_long=InitiateEmptyPaths(par.drp+T)
    TS_LR=InitiateEmptyPaths(T)
    Def0=0.0; z0=1.0
    K0=0.5*(par.klow+par.khigh); B0=0.0
    Fill_Path_Simulation!(PATHS_long,Def0,z0,K0,B0,MOD_FR)
    ExtractFromLongPaths!(par.drp+1,par.drp+T,TS_LR,PATHS_long)

    TS_Bind=Array{Float64,1}(undef,T)
    for t=1:T
        z=TS_LR.z[t]
        k=TS_LR.K[t]
        b=TS_LR.B[t]
        y=FinalOutput(z,k,par)
        gdp=4*y
        DL=FR*gdp
        if b<DL
            TS_Bind[t]=0.0
        else
            TS_Bind[t]=1.0
        end
    end
    return TS_Bind, TS_LR
end

function FrequencyOfRuleBinding_Bench(T::Int64,MOD::Model,MOD_FR::Model)
    @unpack par = MOD_FR
    @unpack FR, χ, γ = par
    Random.seed!(1234)
    #Get initial ergodic starting point
    PATHS_long=InitiateEmptyPaths(par.drp+T)
    TS_LR=InitiateEmptyPaths(T)
    Def0=0.0; z0=1.0
    K0=0.5*(par.klow+par.khigh); B0=0.0
    Fill_Path_Simulation!(PATHS_long,Def0,z0,K0,B0,MOD)
    ExtractFromLongPaths!(par.drp+1,par.drp+T,TS_LR,PATHS_long)

    TS_Bind=Array{Float64,1}(undef,T)
    for t=1:T
        z=TS_LR.z[t]
        k=TS_LR.K[t]
        b=TS_LR.B[t]
        y=FinalOutput(z,k,par)
        gdp=4*y
        DL=FR*gdp
        if b<DL
            TS_Bind[t]=0.0
        else
            TS_Bind[t]=1.0
        end
    end
    return TS_Bind, TS_LR
end
