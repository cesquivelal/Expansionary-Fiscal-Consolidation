
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
    α::Float64 = 0.33            #Capital share
    δ::Float64 = 0.05            #Capital depreciation rate
    φ::Float64 = 21.0             #Capital adjustment cost parameter
    #Fiscal rules and government expenditure
    WithFR::Bool = false         #Define whether to apply fiscal rule
    FR::Float64 = 0.30           #Maximum b'/GDP
    χ::Float64 = 0.03               #Fiscal consolidation parameter
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
    cmin::Float64 = 1e-2
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
### Functions to save solution in CSV
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
    ### Interpolation objects
    #Input values for end of time
    return Solution(VD,VP,V,EVD,EV,q1,kprime_D,kprime,bprime,Tr,DEV_RULE,itp_VD,itp_VP,itp_V,itp_EVD,itp_EV,itp_q1,itp_kprime_D,itp_kprime,itp_bprime)
end

function StackSolution_Vector(SOLUTION::Solution)
    #Stack vectors of repayment first
    @unpack VP, V, EV, q1 = SOLUTION
    @unpack kprime, bprime, Tr, DEV_RULE = SOLUTION
    VEC=reshape(VP,(:))
    VEC=vcat(VEC,reshape(V,(:)))
    VEC=vcat(VEC,reshape(EV,(:)))
    VEC=vcat(VEC,reshape(q1,(:)))
    VEC=vcat(VEC,reshape(kprime,(:)))
    VEC=vcat(VEC,reshape(bprime,(:)))
    VEC=vcat(VEC,reshape(Tr,(:)))
    VEC=vcat(VEC,reshape(DEV_RULE,(:)))

    #Then stack vectors of default
    @unpack VD, EVD, kprime_D = SOLUTION
    VEC=vcat(VEC,reshape(VD,(:)))
    VEC=vcat(VEC,reshape(EVD,(:)))
    VEC=vcat(VEC,reshape(kprime_D,(:)))

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

    return VEC
end

function Create_Model_Vector(SOLUTION::Solution,par::Pars)
    VEC_PAR=VectorOfRelevantParameters(par)
    N_parameters=length(VEC_PAR)
    VEC=vcat(N_parameters,VEC_PAR)

    #Stack SOLUTION in one vector
    VEC_SOL=StackSolution_Vector(SOLUTION)

    return vcat(VEC,VEC_SOL)
end

function SaveModel_Vector(NAME::String,SOLUTION::Solution,par::Pars)
    VEC=Create_Model_Vector(SOLUTION,par)
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

    #Default
    start=start+size_repayment
    VD=ExtractMatrixFromSolutionVector(start,size_default,true,VEC,par)
    start=start+size_default
    EVD=ExtractMatrixFromSolutionVector(start,size_default,true,VEC,par)
    start=start+size_default
    kprime_D=ExtractMatrixFromSolutionVector(start,size_default,true,VEC,par)
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

    ### Interpolation objects
    #Input values for end of time
    return Solution(VD,VP,V,EVD,EV,q1,kprime_D,kprime,bprime,Tr,DEV_RULE,itp_VD,itp_VP,itp_V,itp_EVD,itp_EV,itp_q1,itp_kprime_D,itp_kprime,itp_bprime)
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
    for j in 1:length(ϵz_weights)
        v, q=foo(ZPRIME[z_ind,j])
        int_v=int_v+ϵz_weights[j]*PDFz[j]*v
        int_q=int_q+ϵz_weights[j]*PDFz[j]*q
    end
    return int_v/FacQz, int_q/FacQz
end

function SDF_Lenders(par::Pars)
    @unpack r_star = par
    return 1/(1+r_star)
end

function Calculate_Covenant_b(itp_q1,z::Float64,k::Float64,b::Float64,bprime::Float64,kprime::Float64,par::Pars)
    @unpack γ, Covenants = par
    if Covenants
        #Compute compensation per remaining bond (1-γ)b for any dilution
        # qq0=itp_q1((1-γ)*b,kprime,z)
        qq0=itp_q1((1-γ)*b,k,z)
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
        return vd, 0.0
    else
        @unpack γ, κ, qmax = par
        SDF=SDF_Lenders(par)
        if γ==1.0
            return vp, SDF
        else
            @unpack itp_q1, itp_kprime, itp_bprime = SOLUTION
            kk=itp_kprime(bprime,kprime,zprime)
            bb=itp_bprime(bprime,kprime,zprime)
            qq=min(qmax,max(0.0,itp_q1(bb,kk,zprime)))
            CC=Calculate_Covenant_b(itp_q1,zprime,kprime,bprime,bb,kk,par)
            return vp, SDF*(γ+(1-γ)*(κ+qq+CC))
        end
    end
end

function ValueAndBondsPayoff_GRIM(zprime::Float64,kprime::Float64,bprime::Float64,SOLUTION::Solution,SOLUTION_NO_RULE::Solution,par::Pars)
    vd=min(0.0,SOLUTION.itp_VD(kprime,zprime))
    vp=min(0.0,SOLUTION.itp_VP(bprime,kprime,zprime))
    vpNoRule=min(0.0,SOLUTION_NO_RULE.itp_VP(bprime,kprime,zprime))
    if vd>max(vp,vpNoRule)
        #Will default for sure
        return vd, 0.0
    else
        if vpNoRule>vp
            #Will deviate from rule
            @unpack γ, κ, qmax = par
            SDF=SDF_Lenders(par)
            if γ==1.0
                return vpNoRule, SDF
            else
                kk=SOLUTION_NO_RULE.itp_kprime(bprime,kprime,zprime)
                bb=SOLUTION_NO_RULE.itp_bprime(bprime,kprime,zprime)
                qq=min(qmax,max(0.0,SOLUTION_NO_RULE.itp_q1(bb,kk,zprime)))
                CC=Calculate_Covenant_b(SOLUTION_NO_RULE.itp_q1,zprime,kprime,bprime,bb,kk,par)
                return vpNoRule, SDF*(γ+(1-γ)*(κ+qq+CC))
            end
        else
            #Will not deviate from the rule
            @unpack γ, κ, qmax = par
            SDF=SDF_Lenders(par)
            if γ==1.0
                return vp, SDF
            else
                kk=SOLUTION.itp_kprime(bprime,kprime,zprime)
                bb=SOLUTION.itp_bprime(bprime,kprime,zprime)
                qq=min(qmax,max(0.0,SOLUTION.itp_q1(bb,kk,zprime)))
                return vp, SDF*(γ+(1-γ)*(κ+qq))
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
    end

    #Repayment and bond price
    for I in CartesianIndices(SOLUTION.q1)
        (b_ind,k_ind,z_ind)=Tuple(I)
        kprime=GRIDS.GR_k[k_ind]
        bprime=GRIDS.GR_b[b_ind]
        foo_mat_vq(zprime::Float64)=ValueAndBondsPayoff(zprime,kprime,bprime,SOLUTION,par)
        SOLUTION.EV[I], SOLUTION.q1[I]=Expectation_over_zprime_2(foo_mat_vq,z_ind,GRIDS)
    end

    return nothing
end

function UpdateExpectations_GRIM!(SOLUTION::Solution,SOLUTION_NO_RULE::Solution,GRIDS::Grids,par::Pars)
    #Loop over all states to compute expectations over p and z

    #Default
    for I in CartesianIndices(SOLUTION.EVD)
        (k_ind,z_ind)=Tuple(I)
        foo_mat_D=CreateInterpolation_ForExpectations(SOLUTION.VD[k_ind,:],GRIDS)
        SOLUTION.EVD[I]=Expectation_over_zprime(foo_mat_D,z_ind,GRIDS)
    end

    #Repayment and bond price
    for I in CartesianIndices(SOLUTION.q1)
        (b_ind,k_ind,z_ind)=Tuple(I)
        kprime=GRIDS.GR_k[k_ind]
        bprime=GRIDS.GR_b[b_ind]
        foo_mat_vq(zprime::Float64)=ValueAndBondsPayoff_GRIM(zprime,kprime,bprime,SOLUTION,SOLUTION_NO_RULE,par)
        SOLUTION.EV[I], SOLUTION.q1[I]=Expectation_over_zprime_2(foo_mat_vq,z_ind,GRIDS)
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
    @unpack FR, χ, γ = par
    #Maximum b/gdp
    y=FinalOutput(z,k,par)
    gdp=4*y    #annualized GDP
    DL=FR*gdp  #debt limit
    # DR=χ*y   #minimum debt reduction if b>DL
    DR=χ*y     #maximum primary deficit if b>DL
    #the parameter χ controls the maximum forced debt reduction
    #if shock is so bad that b is way above DL then the
    #government is not forced to pay all the way down to DL,
    #just pay, at most, DR
    return max(DL,(1-γ)*b+DR)
end

function ComputeSpreadWithQ(qq::Float64,par::Pars)
    @unpack r_star, γ, κ = par
    ib=(((γ+(1-γ)*(κ+qq))/qq)^4)-1
    rf=((1+r_star)^4)-1
    return 100*(ib-rf)
end

function Calculate_Tr(z::Float64,k::Float64,b::Float64,kprime::Float64,bprime::Float64,SOLUTION::Solution,par::Pars)
    @unpack γ, κ = par
    @unpack itp_q1 = SOLUTION
    #Compute net borrowing from the rest of the world
    Covenant=Calculate_Covenant_b(itp_q1,z,k,b,bprime,kprime,par)
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
function ConsNet(y::Float64,k::Float64,kprime::Float64,T::Float64,par::Pars)
    #Resource constraint for the final good
    @unpack δ = par
    inv=kprime-(1-δ)*k
    AdjCost=CapitalAdjustment(kprime,k,par)
    return y-inv-AdjCost+T
end

function Evaluate_cons_state(Default::Bool,I::CartesianIndex,kprime::Float64,Tr::Float64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    if Default
        @unpack GR_z, GR_k = GRIDS
        (k_ind,z_ind)=Tuple(I)
        z=GR_z[z_ind]; k=GR_k[k_ind]
        zD=zDefault(z,par)
        y=FinalOutput(zD,k,par)
        return ConsNet(y,k,kprime,Tr,par)
    else
        @unpack GR_z, GR_k, GR_b = GRIDS
        @unpack itp_q1 = SOLUTION
        (b_ind,k_ind,z_ind)=Tuple(I)
        z=GR_z[z_ind]; k=GR_k[k_ind]; b=GR_b[b_ind]
        y=FinalOutput(z,k,par)
        return ConsNet(y,k,kprime,Tr,par)
    end
end

function Evaluate_VD(I::CartesianIndex,kprime::Float64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, θ, cmin = par
    @unpack itp_EV, itp_EVD = SOLUTION
    (k_ind,z_ind)=Tuple(I)
    z=GRIDS.GR_z[z_ind]
    Default=true; Tr=0.0
    cons=Evaluate_cons_state(Default,I,kprime,Tr,SOLUTION,GRIDS,par)
    if cons>cmin
        return Utility(cons,par)+β*θ*itp_EV(0.0,kprime,z)+β*(1.0-θ)*itp_EVD(kprime,z)
    else
        return Utility(cmin,par)+cons
    end
end

function Evaluate_VP(I::CartesianIndex,kprime::Float64,bprime::Float64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, cmin = par
    @unpack itp_EV, itp_q1 = SOLUTION
    @unpack GR_z, GR_k, GR_b = GRIDS

    #Compute consumption
    #Unpack state
    (b_ind,k_ind,z_ind)=Tuple(I)
    z=GR_z[z_ind]; k=GR_k[k_ind]; b=GR_b[b_ind]
    Default=false
    Tr=Calculate_Tr(z,k,b,kprime,bprime,SOLUTION,par)
    cons=Evaluate_cons_state(Default,I,kprime,Tr,SOLUTION,GRIDS,par)
    if cons>0.0
        qq=itp_q1(bprime,kprime,z)
        if bprime>0.0 && qq==0.0
            #Small penalty for larger debt positions
            #wrong side of laffer curve, it is decreasing
            return Utility(cons,par)+β*itp_EV(bprime,kprime,z)-abs(bprime)*sqrt(eps(Float64))
        else
            return Utility(cons,par)+β*itp_EV(bprime,kprime,z)
        end
    else
        return Utility(cmin,par)+cons
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

function HH_k_Returns_D(I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack cmin, β = par
    @unpack kprime_D = SOLUTION
    @unpack GR_z, GR_k = GRIDS
    #Unpack state and compute output
    (k_ind,z_ind)=Tuple(I)
    z=GR_z[z_ind]; k=GR_k[k_ind]
    zD=zDefault(z,par)

    Default=true
    kprimef=kprime_D[I]; Tr=0.0
    cons=Evaluate_cons_state(Default,I,kprimef,Tr,SOLUTION,GRIDS,par)
    if cons<=0.0
        cons=cmin
    end

    return β*U_c(cons,par)*R_dec(kprimef,zD,k,par)
end

function HH_k_Returns_P(I::CartesianIndex,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack cmin, β = par
    @unpack GR_z, GR_k, GR_b = GRIDS
    @unpack kprime, Tr = SOLUTION
    @unpack VP, VD = SOLUTION
    #Unpack state index
    (b_ind,k_ind,z_ind)=Tuple(I)
    #Unpack state and policies
    z=GR_z[z_ind]; k=GR_k[k_ind]; b=GR_b[b_ind]
    kprimef=kprime[I]

    #Compute consumption
    Default=false
    kprimef=kprime[I]; T=Tr[I]
    cons=Evaluate_cons_state(Default,I,kprimef,T,SOLUTION,GRIDS,par)
    NegConDef=0.0

    if cons>0.0
        #Calculate return
        return β*U_c(cons,par)*R_dec(kprimef,z,k,par), NegConDef
    else
        #Use return in default and flag if default is somehow not chosen
        #this avoids overshooting of interpolation of RN and RT
        @unpack dR_Def = HH_OBJ
        if VP[I]>VD[k_ind,z_ind]
            #Problem, somewhow repay with negative consumption
            #Flag it
            NegConDef=1.0
        end
        return dR_Def[k_ind,z_ind], NegConDef
    end
end

function UpdateHH_Obj!(HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Nz, Nk, Nb, cmin, β, θ = par
    @unpack GR_z, GR_k, GR_b = GRIDS
    @unpack kprime_D = SOLUTION
    @unpack kprime, bprime, Tr = SOLUTION
    #Loop over all states to fill matrices in default
    for I in CartesianIndices(HH_OBJ.dR_Def)
        HH_OBJ.dR_Def[I]=HH_k_Returns_D(I,SOLUTION,GRIDS,par)
    end
    HH_OBJ.itp_dR_Def=CreateInterpolation_HouseholdObjects(HH_OBJ.dR_Def,true,GRIDS)

    #Loop over all states to fill matrices in repayment
    for I in CartesianIndices(HH_OBJ.dR_Rep)
        HH_OBJ.dR_Rep[I], NegConDef=HH_k_Returns_P(I,HH_OBJ,SOLUTION,GRIDS,par)
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

function UpdateHH_Obj_GRIM!(HH_OBJ::HH_itpObjects,SOLUTION::Solution,SOLUTION_NO_RULE::Solution,GRIDS::Grids,par::Pars)
    @unpack Nz, Nk, Nb, cmin, β, θ = par
    @unpack GR_z, GR_k, GR_b = GRIDS
    @unpack kprime_D = SOLUTION
    @unpack kprime, bprime, Tr = SOLUTION
    #Loop over all states to fill matrices in default
    for I in CartesianIndices(HH_OBJ.dR_Def)
        HH_OBJ.dR_Def[I]=HH_k_Returns_D(I,SOLUTION,GRIDS,par)
    end
    HH_OBJ.itp_dR_Def=CreateInterpolation_HouseholdObjects(HH_OBJ.dR_Def,true,GRIDS)

    #Loop over all states to fill matrices in repayment
    for I in CartesianIndices(HH_OBJ.dR_Rep)
        if SOLUTION.DEV_RULE[I]==1.0
            #Will definitely break the rule
            HH_OBJ.dR_Rep[I], NegConDef=HH_k_Returns_P(I,HH_OBJ,SOLUTION_NO_RULE,GRIDS,par)
        else
            #Will not break the rule, may default
            HH_OBJ.dR_Rep[I], NegConDef=HH_k_Returns_P(I,HH_OBJ,SOLUTION,GRIDS,par)
        end
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
function HH_FOC_Def(I::CartesianIndex,kprime::Float64,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack cmin = par
    @unpack GR_z, GR_k = GRIDS
    @unpack itp_ER_Def = HH_OBJ
    (k_ind,z_ind)=Tuple(I)
    z=GR_z[z_ind]; k=GR_k[k_ind]
    #Compute present consumption
    Default=true; Tr=0.0
    c=max(cmin,Evaluate_cons_state(Default,I,kprime,Tr,SOLUTION,GRIDS,par))

    # compute expectation over z and yC
    Ev=itp_ER_Def(kprime,z)
    #compute extra term and return FOC
    ψ1=dΨ_d1(kprime,k,par)
    return Ev-U_c(c,par)*(1.0+ψ1)
end

function HH_FOC_Rep(I::CartesianIndex,kprime::Float64,bprime::Float64,
                    HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack cmin = par
    @unpack GR_z, GR_k, GR_b = GRIDS
    @unpack itp_q1 = SOLUTION
    @unpack itp_ER_Rep = HH_OBJ
    (b_ind,k_ind,z_ind)=Tuple(I)
    z=GR_z[z_ind]; k=GR_k[k_ind]; b=GR_b[b_ind]

    #Compute present consumption
    Default=false; Tr=Calculate_Tr(z,k,b,kprime,bprime,SOLUTION,par)
    c=max(cmin,Evaluate_cons_state(Default,I,kprime,Tr,SOLUTION,GRIDS,par))

    #compute expectation over z and yC
    Ev=itp_ER_Rep(bprime,kprime,z)
    #compute extra term and return FOC
    ψ1=dΨ_d1(kprime,k,par)
    return Ev-U_c(c,par)*(1.0+ψ1)
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

function HHOptim_Def(I::CartesianIndex,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack GR_k = GRIDS
    (k_ind,z_ind)=Tuple(I)
    k=GR_k[k_ind]

    foo(kprime::Float64)=HH_FOC_Def(I,kprime,HH_OBJ,SOLUTION,GRIDS,par)
    ν=0.05
    kkL, kkH=FindBracketing(foo,ν,k,k-ν,k+ν)
    kk=MyBisection(foo,kkL,kkH;xatol=1e-6)

    Default=true; Tr=0.0
    cons=Evaluate_cons_state(Default,I,kk,Tr,SOLUTION,GRIDS,par)
    while cons<0.0
        kk=0.5*(k+kk)
        cons=Evaluate_cons_state(Default,I,kk,Tr,SOLUTION,GRIDS,par)
    end
    return kk
end

function HHOptim_Rep(I::CartesianIndex,bprime::Float64,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack GR_k = GRIDS
    (b_ind,k_ind,z_ind)=Tuple(I)
    k=GR_k[k_ind]

    foo(kprime::Float64)=HH_FOC_Rep(I,kprime,bprime,HH_OBJ,SOLUTION,GRIDS,par)
    ν=0.05
    kkL, kkH=FindBracketing(foo,ν,k,k-ν,k+ν)
    kk=MyBisection(foo,kkL,kkH;xatol=1e-6)

    return kk
end

function GridSearchOverK_D(I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    kpol=0
    val=-Inf
    for ktry in 1:length(GRIDS.GR_k)
        kprime=GRIDS.GR_k[ktry]
        vv=Evaluate_VD(I,kprime,SOLUTION,GRIDS,par)
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

function OptimInDefault_PLA(I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack GR_z, GR_k = GRIDS
    (k_ind,z_ind)=Tuple(I)

    klow, khigh=GridSearchOverK_D(I,SOLUTION,GRIDS,par)
    foo(kprime::Float64)=-Evaluate_VD(I,kprime,SOLUTION,GRIDS,par)
    res=optimize(foo,klow,khigh,GoldenSection())

    return -Optim.minimum(res), Optim.minimizer(res)
end

function OptimInDefault_DEC(I::CartesianIndex,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    kprime=HHOptim_Def(I,HH_OBJ,SOLUTION,GRIDS,par)
    val=Evaluate_VD(I,kprime,SOLUTION,GRIDS,par)
    return val, kprime
end

function InitialPolicyGuess_P(PreviousPolicy::Bool,I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    (b_ind,k_ind,z_ind)=Tuple(I)
    if PreviousPolicy
        #Choose last policy as initial guess for capital
        if SOLUTION.kprime[I]==0.0
            #It is the first attempt, use current kN
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

function ValueInRepayment_PLA(blowOpt::Float64,bhighOpt::Float64,I::CartesianIndex,X_REAL::Array{Float64,1},SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack klowOpt, khighOpt = par
    #transform policy tries into interval
    kprime=TransformIntoBounds(X_REAL[1],klowOpt,khighOpt)
    bprime=TransformIntoBounds(X_REAL[2],blowOpt,bhighOpt)
    vv=Evaluate_VP(I,kprime,bprime,SOLUTION,GRIDS,par)
    return vv
end

function ValueInRepayment_DEC(I::CartesianIndex,bprime::Float64,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #transform policy tries into interval
    kprime=HHOptim_Rep(I,bprime,HH_OBJ,SOLUTION,GRIDS,par)
    vv=Evaluate_VP(I,kprime,bprime,SOLUTION,GRIDS,par)
    return vv
end

function Search_bprime_unconstrained(I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack GR_k, GR_b = GRIDS
    (b_ind,k_ind,z_ind)=Tuple(I)
    if SOLUTION.kprime[I]==0.0
        kprime=GR_k[k_ind]
    else
        kprime=SOLUTION.kprime[I]
    end
    val=-Inf
    bpol=0
    for btry in 1:par.Nb
        vv=Evaluate_VP(I,kprime,GR_b[btry],SOLUTION,GRIDS,par)
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
    blowOpt, bhighOpt=Search_bprime_unconstrained(I,SOLUTION,GRIDS,par)
    if par.WithFR
        @unpack GR_z, GR_k, GR_b = GRIDS
        (b_ind,k_ind,z_ind)=Tuple(I)
        z=GR_z[z_ind]; k=GR_k[k_ind]; b=GR_b[b_ind]
        bbar=DebtLimit(z,k,b,par)
        if bbar<blowOpt
            return par.blowOpt, bbar
        else
            return blowOpt, min(bbar,bhighOpt)
        end
    else
        return blowOpt, bhighOpt
    end
end

function OptimInRepayment_PLA(I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack klowOpt, khighOpt = par
    blowOpt, bhighOpt=BoundsForBprime(I,SOLUTION,GRIDS,par)
    if blowOpt==bhighOpt
        blowOpt=par.blowOpt
    end

    PreviousPolicy=true
    X0_BOUNDS=InitialPolicyGuess_P(PreviousPolicy,I,SOLUTION,GRIDS,par)
    X0=Array{Float64,1}(undef,2)
    X0[1]=TransformIntoReals(X0_BOUNDS[1],klowOpt,khighOpt)
    if X0_BOUNDS[2]<bhighOpt && X0_BOUNDS[2]>blowOpt
        X0[2]=TransformIntoReals(X0_BOUNDS[2],blowOpt,bhighOpt)
    else
        X0[2]=TransformIntoReals(0.5*(blowOpt+bhighOpt),blowOpt,bhighOpt)
    end

    foo(X::Array{Float64,1})=-ValueInRepayment_PLA(blowOpt,bhighOpt,I,X,SOLUTION,GRIDS,par)
    res=optimize(foo,X0,NelderMead())
    vv=-Optim.minimum(res)
    kprime=TransformIntoBounds(Optim.minimizer(res)[1],klowOpt,khighOpt)
    bprime=TransformIntoBounds(Optim.minimizer(res)[2],blowOpt,bhighOpt)

    @unpack itp_q1 = SOLUTION
    @unpack GR_z, GR_k, GR_b = GRIDS
    (b_ind,k_ind,z_ind)=Tuple(I)
    z=GR_z[z_ind]; k=GR_k[k_ind]; b=GR_b[b_ind]
    Tr=Calculate_Tr(z,k,b,kprime,bprime,SOLUTION,par)
    return vv, kprime, bprime, Tr
end

function OptimInRepayment_DEC(I::CartesianIndex,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    blowOpt, bhighOpt=BoundsForBprime(I,SOLUTION,GRIDS,par)

    foo(bprime::Float64)=-ValueInRepayment_DEC(I,bprime,HH_OBJ,SOLUTION,GRIDS,par)
    res=optimize(foo,blowOpt,bhighOpt,GoldenSection())
    bprime=Optim.minimizer(res)
    kprime=HHOptim_Rep(I,bprime,HH_OBJ,SOLUTION,GRIDS,par)

    @unpack itp_q1 = SOLUTION
    @unpack GR_z, GR_k, GR_b = GRIDS
    (b_ind,k_ind,z_ind)=Tuple(I)
    z=GR_z[z_ind]; k=GR_k[k_ind]; b=GR_b[b_ind]
    Tr=Calculate_Tr(z,k,b,kprime,bprime,SOLUTION,par)
    return -Optim.minimum(res), kprime, bprime, Tr
end

###############################################################################
#Update solution
###############################################################################
function DefaultUpdater_PLA!(I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    SOLUTION.VD[I], SOLUTION.kprime_D[I]=OptimInDefault_PLA(I,SOLUTION,GRIDS,par)
    return nothing
end

function DefaultUpdater_DEC!(I::CartesianIndex,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    SOLUTION.VD[I], SOLUTION.kprime_D[I]=OptimInDefault_DEC(I,HH_OBJ,SOLUTION,GRIDS,par)
    return nothing
end

function UpdateDefault_PLA!(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Loop over all states to fill array of VD
    for I in CartesianIndices(SOLUTION.VD)
        DefaultUpdater_PLA!(I,SOLUTION,GRIDS,par)
    end
    IsDefault=true
    SOLUTION.itp_VD=CreateInterpolation_ValueFunctions(SOLUTION.VD,IsDefault,GRIDS)
    SOLUTION.itp_kprime_D=CreateInterpolation_Policies(SOLUTION.kprime_D,IsDefault,GRIDS)

    return nothing
end

function UpdateDefault_DEC!(HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Loop over all states to fill array of VD
    for I in CartesianIndices(SOLUTION.VD)
        DefaultUpdater_DEC!(I,HH_OBJ,SOLUTION,GRIDS,par)
    end
    IsDefault=true
    SOLUTION.itp_VD=CreateInterpolation_ValueFunctions(SOLUTION.VD,IsDefault,GRIDS)
    SOLUTION.itp_kprime_D=CreateInterpolation_Policies(SOLUTION.kprime_D,IsDefault,GRIDS)

    return nothing
end

function UpdateDefault_GRIM!(RuleAfterDefault::Bool,HH_OBJ::HH_itpObjects,SOLUTION::Solution,SOLUTION_NO_RULE::Solution,GRIDS::Grids)
    #Loop over all states to fill array of VD
    if RuleAfterDefault
        for I in CartesianIndices(SOLUTION.VD)
            DefaultUpdater_DEC!(I,HH_OBJ,SOLUTION,GRIDS,par)
        end
    else
        for I in CartesianIndices(SOLUTION.VD)
            SOLUTION.VD[I]=SOLUTION_NO_RULE.VD[I]
            SOLUTION.kprime_D[I]=SOLUTION_NO_RULE.kprime_D[I]
        end
    end

    IsDefault=true
    SOLUTION.itp_VD=CreateInterpolation_ValueFunctions(SOLUTION.VD,IsDefault,GRIDS)
    SOLUTION.itp_kprime_D=CreateInterpolation_Policies(SOLUTION.kprime_D,IsDefault,GRIDS)

    return nothing
end

function RepaymentUpdater_PLA!(I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    SOLUTION.VP[I], SOLUTION.kprime[I], SOLUTION.bprime[I], SOLUTION.Tr[I]=OptimInRepayment_PLA(I,SOLUTION,GRIDS,par)
    (b_ind,k_ind,z_ind)=Tuple(I)
    if SOLUTION.VP[I]<SOLUTION.VD[k_ind,z_ind]
        SOLUTION.V[I]=SOLUTION.VD[k_ind,z_ind]
    else
        SOLUTION.V[I]=SOLUTION.VP[I]
    end
    return nothing
end

function RepaymentUpdater_DEC!(I::CartesianIndex,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    SOLUTION.VP[I], SOLUTION.kprime[I], SOLUTION.bprime[I], SOLUTION.Tr[I]=OptimInRepayment_DEC(I,HH_OBJ,SOLUTION,GRIDS,par)
    (b_ind,k_ind,z_ind)=Tuple(I)
    if SOLUTION.VP[I]<SOLUTION.VD[k_ind,z_ind]
        SOLUTION.V[I]=SOLUTION.VD[k_ind,z_ind]
    else
        SOLUTION.V[I]=SOLUTION.VP[I]
    end
    return nothing
end

function RepaymentUpdater_GRIM!(I::CartesianIndex,HH_OBJ::HH_itpObjects,SOLUTION::Solution,SOLUTION_NO_RULE::Solution,GRIDS::Grids,par::Pars)
    SOLUTION.VP[I], SOLUTION.kprime[I], SOLUTION.bprime[I], SOLUTION.Tr[I]=OptimInRepayment_DEC(I,HH_OBJ,SOLUTION,GRIDS,par)
    (b_ind,k_ind,z_ind)=Tuple(I)
    if SOLUTION.VP[I]<SOLUTION_NO_RULE.VP[I]
        #Prefer to deviate than uphold, check if would rather default
        if SOLUTION_NO_RULE.VP[I]<SOLUTION.VD[k_ind,z_ind]
            #Default instead of deviating from rule
            SOLUTION.DEV_RULE[I]=0.0
            SOLUTION.V[I]=SOLUTION.VD[k_ind,z_ind]
        else
            #Deviate from rule
            SOLUTION.DEV_RULE[I]=1.0
            SOLUTION.V[I]=SOLUTION_NO_RULE.VP[I]
        end
    else
        #Prefer to uphold than to deviate, check if want to default
        SOLUTION.DEV_RULE[I]=0.0
        if SOLUTION.VP[I]<SOLUTION.VD[k_ind,z_ind]
            SOLUTION.V[I]=SOLUTION.VD[k_ind,z_ind]
        else
            SOLUTION.V[I]=SOLUTION.VP[I]
        end
    end
    return nothing
end

function UpdateRepayment_PLA!(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Loop over all states to fill array of VD
    for I in CartesianIndices(SOLUTION.VP)
        RepaymentUpdater_PLA!(I,SOLUTION,GRIDS,par)
    end

    IsDefault=false
    SOLUTION.itp_VP=CreateInterpolation_ValueFunctions(SOLUTION.VP,IsDefault,GRIDS)
    SOLUTION.itp_V=CreateInterpolation_ValueFunctions(SOLUTION.V,IsDefault,GRIDS)
    SOLUTION.itp_kprime=CreateInterpolation_Policies(SOLUTION.kprime,IsDefault,GRIDS)
    SOLUTION.itp_bprime=CreateInterpolation_Policies(SOLUTION.bprime,IsDefault,GRIDS)

    return nothing
end

function UpdateRepayment_DEC!(HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Loop over all states to fill array of VD
    for I in CartesianIndices(SOLUTION.VP)
        RepaymentUpdater_DEC!(I,HH_OBJ,SOLUTION,GRIDS,par)
    end

    IsDefault=false
    SOLUTION.itp_VP=CreateInterpolation_ValueFunctions(SOLUTION.VP,IsDefault,GRIDS)
    SOLUTION.itp_V=CreateInterpolation_ValueFunctions(SOLUTION.V,IsDefault,GRIDS)
    SOLUTION.itp_kprime=CreateInterpolation_Policies(SOLUTION.kprime,IsDefault,GRIDS)
    SOLUTION.itp_bprime=CreateInterpolation_Policies(SOLUTION.bprime,IsDefault,GRIDS)

    return nothing
end

function UpdateRepayment_GRIM!(HH_OBJ::HH_itpObjects,SOLUTION::Solution,SOLUTION_NO_RULE::Solution,GRIDS::Grids,par::Pars)
    #Loop over all states to fill array of VD
    for I in CartesianIndices(SOLUTION.VP)
        RepaymentUpdater_GRIM!(I,HH_OBJ,SOLUTION,SOLUTION_NO_RULE,GRIDS,par)
    end

    IsDefault=false
    SOLUTION.itp_VP=CreateInterpolation_ValueFunctions(SOLUTION.VP,IsDefault,GRIDS)
    SOLUTION.itp_V=CreateInterpolation_ValueFunctions(SOLUTION.V,IsDefault,GRIDS)
    SOLUTION.itp_kprime=CreateInterpolation_Policies(SOLUTION.kprime,IsDefault,GRIDS)
    SOLUTION.itp_bprime=CreateInterpolation_Policies(SOLUTION.bprime,IsDefault,GRIDS)

    return nothing
end

function UpdateSolution_PLA!(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    UpdateDefault_PLA!(SOLUTION,GRIDS,par)
    UpdateRepayment_PLA!(SOLUTION,GRIDS,par)
    UpdateExpectations!(SOLUTION,GRIDS,par)

    #Compute expectation interpolations
    IsDefault=true
    SOLUTION.itp_EVD=CreateInterpolation_ValueFunctions(SOLUTION.EVD,IsDefault,GRIDS)
    IsDefault=false
    SOLUTION.itp_EV=CreateInterpolation_ValueFunctions(SOLUTION.EV,IsDefault,GRIDS)
    SOLUTION.itp_q1=CreateInterpolation_Price(SOLUTION.q1,GRIDS)
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

function UpdateSolution_DEC!(HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    UpdateDefault_DEC!(HH_OBJ,SOLUTION,GRIDS,par)
    UpdateRepayment_DEC!(HH_OBJ,SOLUTION,GRIDS,par)
    UpdateExpectations!(SOLUTION,GRIDS,par)

    #Compute expectation interpolations
    IsDefault=true
    SOLUTION.itp_EVD=CreateInterpolation_ValueFunctions(SOLUTION.EVD,IsDefault,GRIDS)
    IsDefault=false
    SOLUTION.itp_EV=CreateInterpolation_ValueFunctions(SOLUTION.EV,IsDefault,GRIDS)
    SOLUTION.itp_q1=CreateInterpolation_Price(SOLUTION.q1,GRIDS)

    UpdateHH_Obj!(HH_OBJ,SOLUTION,GRIDS,par)
    return nothing
end

function UpdateSolution_GRIM!(RuleAfterDefault::Bool,HH_OBJ_FR::HH_itpObjects,SOLUTION::Solution,SOLUTION_NO_RULE::Solution,GRIDS::Grids,par::Pars)
    UpdateDefault_GRIM!(RuleAfterDefault,HH_OBJ_FR,SOLUTION,SOLUTION_NO_RULE,GRIDS)
    UpdateRepayment_GRIM!(HH_OBJ_FR,SOLUTION,SOLUTION_NO_RULE,GRIDS,par)
    UpdateExpectations_GRIM!(SOLUTION,SOLUTION_NO_RULE,GRIDS,par)

    #Compute expectation interpolations
    IsDefault=true
    SOLUTION.itp_EVD=CreateInterpolation_ValueFunctions(SOLUTION.EVD,IsDefault,GRIDS)
    IsDefault=false
    SOLUTION.itp_EV=CreateInterpolation_ValueFunctions(SOLUTION.EV,IsDefault,GRIDS)
    SOLUTION.itp_q1=CreateInterpolation_Price(SOLUTION.q1,GRIDS)

    UpdateHH_Obj_GRIM!(HH_OBJ_FR,SOLUTION,SOLUTION_NO_RULE,GRIDS,par)
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

function SolveModel_VFI(Decentralized::Bool,PrintProg::Bool,PrintAll::Bool,GRIDS::Grids,par::Pars)
    @unpack Tol_q, Tol_V, cnt_max = par
    if PrintProg
        println("Preparing solution guess")
    end
    SOLUTION_CURRENT=InitiateEmptySolution(GRIDS,par)
    if Decentralized
        HH_OBJ=InitiateHH_Obj(SOLUTION_CURRENT,GRIDS,par)
    end
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
        if Decentralized
            UpdateSolution_DEC!(HH_OBJ,SOLUTION_NEXT,GRIDS,par)
        else
            UpdateSolution_PLA!(SOLUTION_NEXT,GRIDS,par)
        end
        dst_q, NotConvPct, Ix=ComputeDistance_q(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        dst_D, dst_P, Iv, NotConvPct_P=ComputeDistanceV(SOLUTION_CURRENT,SOLUTION_NEXT,par)
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
    return SOLUTION_NEXT
end

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

function SolveModel_VFI_GRIM(RuleAfterDefault::Bool,PrintProg::Bool,PrintAll::Bool,GRIDS::Grids,par::Pars,parFR::Pars)
    @unpack Tol_q, Tol_V, cnt_max = par
    if PrintProg
        println("Preparing solution guess")
    end
    SOLUTION_CURRENT=InitiateEmptySolution(GRIDS,par)
    SOLUTION_CURRENT_FR=InitiateEmptySolution(GRIDS,parFR)

    HH_OBJ=InitiateHH_Obj(SOLUTION_CURRENT,GRIDS,par)
    HH_OBJ_FR=InitiateHH_Obj(SOLUTION_CURRENT_FR,GRIDS,parFR)

    SOLUTION_NEXT=deepcopy(SOLUTION_CURRENT)
    SOLUTION_NEXT_FR=deepcopy(SOLUTION_CURRENT_FR)

    dst_V=1.0
    rdts_V=100.0
    dst_q=1.0
    NotConvPct=100.0
    cnt=0
    if PrintProg
        println("Starting VFI")
    end
    while cnt<cnt_max && (dst_V>Tol_V || dst_q>Tol_q)
        #Update solutions, first no rule
        UpdateSolution_DEC!(HH_OBJ,SOLUTION_NEXT,GRIDS,par)
        UpdateSolution_GRIM!(RuleAfterDefault,HH_OBJ_FR,SOLUTION_NEXT_FR,SOLUTION_NEXT,GRIDS,parFR)

        #Compute distances no rule
        dst_q, NotConvPct=ComputeDistance_q(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        dst_D, dst_P, Iv, NotConvPct_P=ComputeDistanceV(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        dst_V=max(dst_D,dst_P)

        #Compute distances fiscal rule
        dst_qFR, NotConvPctFR=ComputeDistance_q(SOLUTION_CURRENT_FR,SOLUTION_NEXT_FR,parFR)
        dst_DFR, dst_PFR, IvFR, NotConvPct_PFR=ComputeDistanceV(SOLUTION_CURRENT_FR,SOLUTION_NEXT_FR,parFR)
        dst_VFR=max(dst_DFR,dst_PFR)

        cnt=cnt+1
        SOLUTION_CURRENT=deepcopy(SOLUTION_NEXT)
        SOLUTION_CURRENT_FR=deepcopy(SOLUTION_NEXT_FR)
        if PrintProg
            if PrintAll
                println("cnt=$cnt, dst_D=$dst_D, dst_P=$dst_P at $Iv, $NotConvPct_P% of V not converged, dst_q=$dst_q")
                println("          dsDFR=$dst_DFR, dsPFR=$dst_PFR at $IvFR, $NotConvPct_PFR% of V not converged, d_qFR=$dst_qFR")
            else
                if mod(cnt,100)==0
                    println("cnt=$cnt, dst_D=$dst_D, dst_P=$dst_P at $Iv, $NotConvPct_P% of V not converged, dst_q=$dst_q")
                    println("          dsDFR=$dst_DFR, dsPFR=$dst_PFR at $IvFR, $NotConvPct_PFR% of V not converged, d_qFR=$dst_qFR")
                end
            end
        end
    end
    return SOLUTION_NEXT, SOLUTION_NEXT_FR
end

function SolveAndSaveModel(Decentralized::Bool,NAME::String,GRIDS::Grids,par::Pars)
    PrintProg=true; PrintAll=true
    SOL=SolveModel_VFI(Decentralized,PrintProg,PrintAll,GRIDS,par)
    SaveModel_Vector(NAME,SOL,par)
    return nothing
end

function SolveAndSaveModel_GRIM(RuleAfterDefault::Bool,NAME_FR::String,GRIDS::Grids,par::Pars,parFR::Pars)
    PrintProg=true; PrintAll=true
    SOL, SOL_FR=SolveModel_VFI_GRIM(RuleAfterDefault,PrintProg,PrintAll,GRIDS,par,parFR)
    SaveModel_Vector(NAME_FR,SOL_FR,parFR)
    return nothing
end

function SolveAndSaveModel_GRIMfast(NAME::String,MOD0::Model,parFR::Pars)
    PrintProg=true; PrintAll=true
    SOL_GRIM=SolveModel_VFI_GRIMfast(PrintProg,PrintAll,MOD0,parFR)
    SaveModel_Vector(NAME,SOL_GRIM,parFR)
    return nothing
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
                PATHS.zP[t]=z
                PATHS.B[t]=0.0
                b=0.0
                kprime=itp_kprime(b,k,z)
                bprime=itp_bprime(b,k,z)
                y=FinalOutput(z,k,par)
                Tr=Calculate_Tr(z,k,b,kprime,bprime,SOLUTION,par)
                PATHS.Def[t]=0.0
                PATHS.Spreads[t]=ComputeSpreads(z,kprime,bprime,SOLUTION,par)
            else
                #Remain in default
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
                PATHS.zP[t]=zD
                kprime=itp_kprime_D(k,z)
                bprime=b
                y=FinalOutput(zD,k,par)
                Tr=0.0
                PATHS.Def[t]=1.0
                PATHS.Spreads[t]=0.0
            else
                #Repayment
                PATHS.zP[t]=z
                kprime=itp_kprime(b,k,z)
                bprime=itp_bprime(b,k,z)
                y=FinalOutput(z,k,par)
                Tr=Calculate_Tr(z,k,b,kprime,bprime,SOLUTION,par)
                PATHS.Def[t]=0.0
                PATHS.Spreads[t]=ComputeSpreads(z,kprime,bprime,SOLUTION,par)
            end
        end
        PATHS.GDP[t]=y
        PATHS.Cons[t]=ConsNet(y,k,kprime,Tr,par)
        PATHS.AdjCost[t]=CapitalAdjustment(kprime,k,par)
        PATHS.Inv[t]=kprime-(1-δ)*k+PATHS.AdjCost[t]
        PATHS.TB[t]=-Tr
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

    Tlong=MOD.par.drp+T
    PATHS_long=InitiateEmptyPaths(Tlong)
    PATHS=InitiateEmptyPaths(T)
    t0=MOD.par.drp+1; t1=Tlong

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
    Decentralized=true; PrintProg=true; PrintAll=false
    SOL=SolveModel_VFI(Decentralized,PrintProg,PrintAll,GRIDS,par)
    MOD=Model(SOL,GRIDS,par)
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

function AverageWelfareGains(ZeroDebt::Bool,N::Int64,MOD0::Model,MOD1::Model)
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
        z=PATHS.z[i]; k=PATHS.K[i]
        if ZeroDebt
            b=0.0
        else
            b=PATHS.B[i]
        end

        v0p=MOD0.SOLUTION.itp_VP(b,k,z)
        v0d=MOD0.SOLUTION.itp_VD(k,z)
        v0=max(v0p,v0d)

        v1p=MOD1.SOLUTION.itp_VP(b,k,z)
        v1d=MOD1.SOLUTION.itp_VD(k,z)
        v1=max(v1p,v1d)

        wg=wg+WelfareGains_Val(v0,v1,par)/N
    end
    return wg
end

function WelfareGainsMatrixFine(N::Int64,zat_f::Float64,MOD0::Model,MOD1::Model)
    @unpack par = MOD0
    WG=Array{Float64,2}(undef,N,N)
    kk=collect(range(par.klow,stop=par.khigh,length=N))
    bb=collect(range(par.blow,stop=par.bhigh,length=N))
    for I in CartesianIndices(WG)
        (b_ind,k_ind)=Tuple(I)
        k=kk[k_ind]; b=bb[b_ind]
        v0=MOD0.SOLUTION.itp_V(b,k,zat_f)
        v1=MOD1.SOLUTION.itp_V(b,k,zat_f)
        WG[I]=WelfareGains_Val(v0,v1,par)
    end
    return WG
end

function PlotWelfareGainsMatrix(N::Int64,zat_f::Float64,MOD_DEC::Model,MOD_DEC_FR::Model)
    @unpack par = MOD_DEC
    WGf=WelfareGainsMatrixFine(N,zat_f,MOD_DEC,MOD_DEC_FR)
    kk=collect(range(par.klow,stop=par.khigh,length=N))
    bb=collect(range(par.blow,stop=par.bhigh,length=N))
    zrnd=round(zat_f,digits=2)
    TITLE="Welfare gains, z=$zrnd"
    plt=heatmap(kk,bb,WGf,c=:hot,title=TITLE,ylabel="B",xlabel="K")
    foo(k::Float64)=parFR.FR*FinalOutput(zat_f,k,par)
    plot!(kk,foo.(kk),ylims=[bb[1],bb[end]],label="debt limit B'<0.63*Y",
          ylabel="B",xlabel="K",linecolor=:blue,legend=:bottomright)

    savefig(plt,"Graphs\\Heatmap_z_$zrnd.pdf")
    return plt
end

function PlotWelfareGainsMatrixPositive(N::Int64,zat_f::Float64,MOD_DEC::Model,MOD_DEC_FR::Model)
    WGf=WelfareGainsMatrixFine(N,zat_f,MOD_DEC,MOD_DEC_FR)
    kk=collect(range(par.klow,stop=par.khigh,length=N))
    bb=collect(range(par.blow,stop=par.bhigh,length=N))
    TITLE="Positive gains from fiscal rule"
    plt=heatmap(kk,bb,WGf .> 0,c=:hot,title=TITLE,ylabel="B",xlabel="K",legend=false)
    foo(k::Float64)=parFR.FR*FinalOutput(zat_f,k,par)
    plot!(kk,foo.(kk),ylims=[bb[1],bb[end]],label="debt limit",
          ylabel="B",xlabel="K",linecolor=:blue)

    savefig(plt,"Graphs\\HeatmapPos_z_$zat_f.pdf")
    return plt
end

###############################################################################
#Functions to find optimal fiscal rule
###############################################################################
function TestOneFiscalRule(IsPlanner::Bool,IsGrim::Bool,FR::Float64,Nwg::Int64,MOD0::Model)
    @unpack par, GRIDS = MOD0
    PrintProg=false; PrintAll=false
    par_fr=Pars(par,WithFR=true,FR=FR)
    if IsPlanner
        #Testing fiscal rule in centralized economy with commitment
        Decentralized=false
        SOL1=SolveModel_VFI(Decentralized,PrintProg,PrintAll,GRIDS,par_fr)
    else
        Decentralized=true
        if IsGrim
            #Testing fiscal rule in decentralized economy with grim strategy
            RuleAfterDefault=false
            SOL0, SOL1=SolveModel_VFI_GRIM(RuleAfterDefault,PrintProg,PrintAll,GRIDS,par,par_fr)
        else
            #Testing fiscal rule in decentralized economy with commitment
            SOL1=SolveModel_VFI(Decentralized,PrintProg,PrintAll,GRIDS,par_fr)
        end
    end
    MOD1=Model(SOL1,GRIDS,par_fr)
    ZeroDebt=false
    wg=AverageWelfareGains(ZeroDebt,Nwg,MOD0,MOD1)
    return wg, MOD1
end

function TestOneTransitionSmoother(IsPlanner::Bool,IsGrim::Bool,FR::Float64,χ::Float64,Nwg::Int64,MOD0::Model)
    @unpack par, GRIDS = MOD0
    PrintProg=false; PrintAll=false
    par_fr=Pars(par,WithFR=true,FR=FR,χ=χ)
    if IsPlanner
        #Testing fiscal rule in centralized economy with commitment
        Decentralized=false
        SOL1=SolveModel_VFI(Decentralized,PrintProg,PrintAll,GRIDS,par_fr)
    else
        Decentralized=true
        if IsGrim
            #Testing fiscal rule in decentralized economy with grim strategy
            RuleAfterDefault=false
            SOL0, SOL1=SolveModel_VFI_GRIM(RuleAfterDefault,PrintProg,PrintAll,GRIDS,par,par_fr)
        else
            #Testing fiscal rule in decentralized economy with commitment
            SOL1=SolveModel_VFI(Decentralized,PrintProg,PrintAll,GRIDS,par_fr)
        end
    end
    MOD1=Model(SOL1,GRIDS,par_fr)
    ZeroDebt=false
    wg=AverageWelfareGains(ZeroDebt,Nwg,MOD0,MOD1)
    return wg, MOD1
end

function TestManyFiscalRules_Dec(grFR::Vector,MOD0_DEC::Model)
    IsPlanner=false; IsGrim=false
    #Allocate matrices to fill
    N=length(grFR); Nwg=10000
    MAT_WG=SharedArray{Float64,2}(N,2)
    VEC=Create_Model_Vector(MOD0_DEC.SOLUTION,MOD0_DEC.par)

    SOLS_DEC=SharedArray{Float64,2}(length(VEC),N)
    @sync @distributed for i_fr in 1:N
        println("Doing decentralized, i=$i_fr of $N")
        #Do decentralized with commitment
        MAT_WG[i_fr,1]=grFR[i_fr]
        MAT_WG[i_fr,2], MOD1=TestOneFiscalRule(IsPlanner,IsGrim,grFR[i_fr],Nwg,MOD0_DEC)
        SOLS_DEC[:,i_fr]=Create_Model_Vector(MOD1.SOLUTION,MOD1.par)
    end

    #Save matrices of tried rules and their welfare gains
    NAMES=["FR" "DEC"]
    MAT=vcat(NAMES,MAT_WG)
    writedlm("WelfareGains_DEC.csv",MAT,',')

    #Save matrices with all solutions
    writedlm("ALL_SOLS_DEC_FR_0.csv",SOLS_DEC,',')

    wg, i_fr_best=findmax(MAT_WG[:,2])

    return grFR[i_fr_best]
end

function TestManyTransitionSmoothers_Dec(FR::Float64,grχ::Vector,MOD0_DEC::Model)
    IsPlanner=false; IsGrim=false
    #Allocate matrices to fill
    N=length(grχ); Nwg=10000
    MAT_WG=SharedArray{Float64,2}(N,2)
    VEC=Create_Model_Vector(MOD0_DEC.SOLUTION,MOD0_DEC.par)

    SOLS_DEC=SharedArray{Float64,2}(length(VEC),N)
    #Span to N*2 workers
    @sync @distributed for i_χ in 1:N
        println("Doing smoother, iχ=$i_χ of $N, FR=$FR")
        #Do decentralized with commitment
        MAT_WG[i_χ,1]=grχ[i_χ]
        MAT_WG[i_χ,2], MOD1=TestOneTransitionSmoother(IsPlanner,IsGrim,FR,grχ[i_χ],Nwg,MOD0_DEC)
        SOLS_DEC[:,i_χ]=Create_Model_Vector(MOD1.SOLUTION,MOD1.par)
    end

    #Save matrices of tried rules and their welfare gains
    NAMES=["chi" "DEC"]
    MAT=vcat(NAMES,MAT_WG)
    writedlm("WelfareGainsTransition_DEC.csv",MAT,',')

    #Save matrices with all solutions
    writedlm("ALL_SOLS_DEC_FR_CHI.csv",SOLS_DEC,',')

    return nothing
end

function StackSolutions_PairsOfPolicies(MAT_SOLS::Array{Float64,3})
    (N_FR, N_χ, Lsol)=size(MAT_SOLS)
    VEC_SOLS=reshape(MAT_SOLS,(:))
    VEC=vcat(N_FR,N_χ)
    VEC=vcat(VEC,Lsol)
    VEC=vcat(VEC,VEC_SOLS)
    return VEC
end

function UnpackArrayOfSolutions(VEC)
    N_FR=convert(Int64,VEC[1])
    N_χ=convert(Int64,VEC[2])
    Lsol=convert(Int64,VEC[3])
    Ix=(N_FR,N_χ,Lsol)

    ARR=reshape(VEC[4:end],Ix)
    return ARR
end

function TestOneFiscalRulePair(IsPlanner::Bool,IsGrim::Bool,FR::Float64,χ::Float64,Nwg::Int64,MOD0::Model)
    @unpack par, GRIDS = MOD0
    PrintProg=false; PrintAll=false
    par_fr=Pars(par,WithFR=true,FR=FR,χ=χ)
    if IsPlanner
        #Testing fiscal rule in centralized economy with commitment
        Decentralized=false
        SOL1=SolveModel_VFI(Decentralized,PrintProg,PrintAll,GRIDS,par_fr)
    else
        Decentralized=true
        if IsGrim
            #Testing fiscal rule in decentralized economy with grim strategy
            RuleAfterDefault=false
            SOL0, SOL1=SolveModel_VFI_GRIM(RuleAfterDefault,PrintProg,PrintAll,GRIDS,par,par_fr)
        else
            #Testing fiscal rule in decentralized economy with commitment
            SOL1=SolveModel_VFI(Decentralized,PrintProg,PrintAll,GRIDS,par_fr)
        end
    end
    MOD1=Model(SOL1,GRIDS,par_fr)
    wg=AverageWelfareGains(false,Nwg,MOD0,MOD1)
    return wg, MOD1
end

function TestManyFiscalRulePairs(IsPlanner::Bool,IsGrim::Bool,grFR::Vector,grχ::Vector,MOD0::Model)
    #Allocate arrays to fill
    #Total computations will be N_FR*N_χ
    N_FR=length(grFR); N_χ=length(grχ); Nwg=10000
    N=N_FR*N_χ

    ARR_WG=SharedArray{Float64,2}(N_FR,N_χ)

    VEC_MOD=Create_Model_Vector(MOD0.SOLUTION,MOD0.par)
    SOLS=SharedArray{Float64,3}(N_FR,N_χ,length(VEC_MOD))

    #Span to N workers
    @sync @distributed for I in CartesianIndices(ARR_WG)
        (i_fr,i_χ)=Tuple(I)
        ARR_WG[i_fr,i_χ], MOD1=TestOneFiscalRulePair(IsPlanner,IsGrim,grFR[i_fr],grχ[i_χ],Nwg,MOD0)
        SOLS[i_fr,i_χ,:]=Create_Model_Vector(MOD1.SOLUTION,MOD1.par)
        if IsPlanner
            println("Done with planner i_fr=$i_fr of $N_FR, i_χ=$i_χ of $N_χ")
        else
            if IsGrim
                println("Done with grim i_fr=$i_fr of $N_FR, i_χ=$i_χ of $N_χ")
            else
                println("Done with decentralized i_fr=$i_fr of $N_FR, i_χ=$i_χ of $N_χ")
            end
        end
    end

    #Create vector with all info of welfare gains
    VEC=vcat(N_FR,N_χ)
    VEC=vcat(VEC,grFR[1])
    VEC=vcat(VEC,grFR[end])
    VEC=vcat(VEC,grχ[1])
    VEC=vcat(VEC,grχ[end])
    VEC_WG=reshape(ARR_WG,(:))
    VEC=vcat(VEC,VEC_WG)

    #Pick model with best fiscal rule
    wg, I_best=findmax(ARR_WG)
    (iFR_best,iχ_best)=Tuple(I_best)
    FR=grFR[iFR_best]; χ=grχ[iχ_best]

    if IsPlanner
        writedlm("TriedRules_PLA.csv",VEC,',')
        NAME_SOLS="ALL_SOLS_PLA_FR.csv"
    else
        if IsGrim
            writedlm("TriedRules_GRIM.csv",VEC,',')
            NAME_SOLS="ALL_SOLS_GRIM_FR.csv"
        else
            writedlm("TriedRules_DEC.csv",VEC,',')
            NAME_SOLS="ALL_SOLS_DEC_FR.csv"
        end
    end

    MAT_SOLS=convert(Array{Float64,3},SOLS)
    VEC_SOLS=StackSolutions_PairsOfPolicies(MAT_SOLS)
    writedlm(NAME_SOLS,VEC_SOLS,',')

    return FR, χ
end

function Map_T_to_χ(FR::Float64,T::Float64,MOD::Model)
    @unpack par = MOD
    TT=par.drp+par.Tmom
    PATHS_long=InitiateEmptyPaths(TT)
    PATHS=InitiateEmptyPaths(par.Tmom)
    t0=par.drp+1; t1=TT

    Def0=0.0; z0=1.0; K0=0.5*(par.klow+par.khigh); B0=0.0
    Fill_Path_Simulation!(PATHS_long,Def0,z0,K0,B0,MOD)
    ExtractFromLongPaths!(t0,t1,PATHS,PATHS_long)

    bbar=mean(PATHS.B)
    AvGDP=4*mean(PATHS.GDP)
    Fbar=FR*AvGDP

    #Assume T is expressed in years
    Tquarters=4*T

    return 1-((Fbar/bbar)^Tquarters)
end

function TestManyFiscalRulePairsT(IsPlanner::Bool,IsGrim::Bool,grFR::Vector,grT::Vector,MOD0::Model)
    #Allocate arrays to fill
    #Total computations will be N_FR*N_χ
    N_FR=length(grFR); N_T=length(grT); Nwg=10000
    N=N_FR*N_T

    ARR_WG=SharedArray{Float64,2}(N_FR,N_T)

    VEC_MOD=Create_Model_Vector(MOD0.SOLUTION,MOD0.par)
    SOLS=SharedArray{Float64,3}(N_FR,N_T,length(VEC_MOD))

    #Span to N workers
    @sync @distributed for I in CartesianIndices(ARR_WG)
        (i_fr,i_T)=Tuple(I)
        FR=grFR[i_fr]; T=grT[i_T]
        χ=Map_T_to_χ(FR,T,MOD0)
        ARR_WG[i_fr,i_T], MOD1=TestOneFiscalRulePair(IsPlanner,IsGrim,FR,χ,Nwg,MOD0)
        SOLS[i_fr,i_T,:]=Create_Model_Vector(MOD1.SOLUTION,MOD1.par)
        println("Done with i_fr=$i_fr of $N_FR, i_T=$i_T of $N_T")
    end

    #Create vector with all info of welfare gains
    VEC=vcat(N_FR,N_T)
    VEC=vcat(VEC,grFR[1])
    VEC=vcat(VEC,grFR[end])
    VEC=vcat(VEC,grT[1])
    VEC=vcat(VEC,grT[end])
    VEC_WG=reshape(ARR_WG,(:))
    VEC=vcat(VEC,VEC_WG)

    if IsPlanner
        writedlm("TriedRules_PLA.csv",VEC,',')
        NAME="SOL_PLA_FR"
    else
        if IsGrim
            writedlm("TriedRules_GRIM.csv",VEC,',')
            NAME="SOL_GRIM"
        else
            writedlm("TriedRules_DEC.csv",VEC,',')
            NAME="SOL_DEC_FR"
        end
    end

    #Save models with best fiscal rule for each T
    MAT_BEST=Array{Float64,2}(undef,length(grT),4)
    for t in 1:length(grT)
        T=grT[t]
        wg, iFR_best=findmax(ARR_WG[:,t])
        FR=grFR[iFR_best]
        χ=Map_T_to_χ(FR,T,MOD0)
        MAT_BEST[t,1]=T
        MAT_BEST[t,2]=FR
        MAT_BEST[t,3]=wg
        MAT_BEST[t,4]=χ
        MOD1=UnpackModel_Vector(SOLS[iFR_best,t,:])
        NAME_t="$NAME$t.csv"
        SaveModel_Vector(NAME_t,MOD1.SOLUTION,MOD1.par)
        println("Best FR for T=$T is FR=$FR with wg=$wg")
    end
    COL_NAMES=["T" "FR" "wg" "chi"]
    writedlm("BestRules_T.csv",vcat(COL_NAMES,MAT_BEST),',')

    return nothing
end

function TestManyFiscalRulesPairs_All(grFR::Vector,grχ::Vector,MOD0_DEC::Model,MOD0_PLA::Model)
    #Allocate arrays to fill
    #Total computations will be N_FR*N_χ
    N_FR=length(grFR); N_χ=length(grχ); Nwg=10000
    N=N_FR*N_χ

    ARR_WG=SharedArray{Float64,3}(N_FR,N_χ,3) #Third dimension is DEC, GRIM, PLA

    VEC_DEC=Create_Model_Vector(MOD0_DEC.SOLUTION,MOD0_DEC.par)
    SOLS_DEC=SharedArray{Float64,3}(N_FR,N_χ,length(VEC_DEC))
    SOLS_GRIM=SharedArray{Float64,3}(N_FR,N_χ,length(VEC_DEC))
    SOLS_PLA=SharedArray{Float64,3}(N_FR,N_χ,length(VEC_DEC))

    #Span to N workers (do PLA with all workers first because it's faster)
    IsPlanner=true; IsGrim=false
    @sync @distributed for I in CartesianIndices(ARR_WG[:,:,3])
        (i_fr,i_χ)=Tuple(I)
        ARR_WG[i_fr,i_χ,3], MOD1=TestOneFiscalRulePair(IsPlanner,IsGrim,grFR[i_fr],grχ[i_χ],Nwg,MOD0_PLA)
        SOLS_PLA[i_fr,i_χ,:]=Create_Model_Vector(MOD1.SOLUTION,MOD1.par)
        println("Planner: done with i_fr=$i_fr of $N_FR, i_χ=$i_χ of $N_χ")
    end

    #Span to N*2 workers (now do DEC and GRIM, which are slower)
    IsPlanner=false
    @sync @distributed for I in CartesianIndices(ARR_WG[:,:,1:2])
        (i_fr,i_χ,i_case)=Tuple(I)
        if i_case==1
            #Do decentralized with commitment
            IsGrim=false
            ARR_WG[I], MOD1=TestOneFiscalRulePair(IsPlanner,IsGrim,grFR[i_fr],grχ[i_χ],Nwg,MOD0_DEC)
            SOLS_DEC[i_fr,i_χ,:]=Create_Model_Vector(MOD1.SOLUTION,MOD1.par)
            println("Decentralized: done with i_fr=$i_fr of $N_FR, i_χ=$i_χ of $N_χ")
        else
            #Do decentralized with grim strategy
            IsGrim=true
            ARR_WG[I], MOD1=TestOneFiscalRulePair(IsPlanner,IsGrim,grFR[i_fr],grχ[i_χ],Nwg,MOD0_DEC)
            SOLS_GRIM[i_fr,i_χ,:]=Create_Model_Vector(MOD1.SOLUTION,MOD1.par)
            println("Grim: done with i_fr=$i_fr of $N_FR, i_χ=$i_χ of $N_χ")
        end
    end
    #Save vector with all info of welfare gains
    VEC=vcat(N_FR,N_χ)
    VEC=vcat(VEC,grFR[1])
    VEC=vcat(VEC,grFR[end])
    VEC=vcat(VEC,grχ[1])
    VEC=vcat(VEC,grχ[end])
    VEC_WG=reshape(ARR_WG,(:))
    VEC=vcat(VEC,VEC_WG)
    writedlm("TriedRules_All.csv",VEC,',')

    #Save models with best fiscal rules
    wg_DEC, I_best_DEC=findmax(ARR_WG[:,:,1])
    wg_GRIM, I_best_GRIM=findmax(ARR_WG[:,:,2])
    wg_PLA, I_best_PLA=findmax(ARR_WG[:,:,3])

    (iFR_best_DEC,iχ_best_DEC)=Tuple(I_best_DEC)
    (iFR_best_GRIM,iχ_best_GRIM)=Tuple(I_best_GRIM)
    (iFR_best_PLA,iχ_best_PLA)=Tuple(I_best_PLA)

    FR_DEC=grFR[iFR_best_DEC]; χ_DEC=grχ[iχ_best_DEC]
    FR_GRIM=grFR[iFR_best_GRIM]; χ_GRIM=grχ[iχ_best_GRIM]
    FR_PLA=grFR[iFR_best_PLA]; χ_PLA=grχ[iχ_best_PLA]

    NAME="SOL_DEC_FR.csv"
    MOD1=UnpackModel_Vector(SOLS_DEC[iFR_best_DEC,iχ_best_DEC,:])
    SaveModel_Vector(NAME,MOD1.SOLUTION,MOD1.par)

    NAME="SOL_GRIM.csv"
    MOD1=UnpackModel_Vector(SOLS_GRIM[iFR_best_GRIM,iχ_best_GRIM,:])
    SaveModel_Vector(NAME,MOD1.SOLUTION,MOD1.par)

    NAME="SOL_PLA_FR.csv"
    MOD1=UnpackModel_Vector(SOLS_PLA[iFR_best_PLA,iχ_best_PLA,:])
    SaveModel_Vector(NAME,MOD1.SOLUTION,MOD1.par)

    println("Best FR for decentralized is FR=$FR_DEC, χ=$χ_DEC with wg=$wg_DEC")
    println("Best FR for planner is FR=$FR_PLA, χ=$χ_PLA with wg=$wg_PLA")
    println("Best FR for grim is FR=$FR_GRIM, χ=$χ_GRIM with wg=$wg_GRIM")

    return nothing
end

function UnpackWelfareGains(FOLDER_FILE::String)
    VEC=readdlm(FOLDER_FILE,',')

    N_FR=convert(Int64,VEC[1])
    N_χ=convert(Int64,VEC[2])
    fr_low=convert(Float64,VEC[3])
    fr_high=convert(Float64,VEC[4])
    χ_low=convert(Float64,VEC[5])
    χ_high=convert(Float64,VEC[6])

    gr_fr=collect(range(fr_low,stop=fr_high,length=N_FR))
    gr_χ=collect(range(χ_low,stop=χ_high,length=N_χ))

    ARR=reshape(1*VEC[7:end,1],(N_FR,N_χ,3))

    return ARR, gr_fr, gr_χ
end

function UnpackWelfareGainsT(FOLDER_FILE::String)
    VEC=readdlm(FOLDER_FILE,',')

    N_FR=convert(Int64,VEC[1])
    N_T=convert(Int64,VEC[2])
    fr_low=convert(Float64,VEC[3])
    fr_high=convert(Float64,VEC[4])
    T_low=convert(Float64,VEC[5])
    T_high=convert(Float64,VEC[6])

    gr_fr=collect(range(fr_low,stop=fr_high,length=N_FR))
    gr_T=collect(range(T_low,stop=T_high,length=N_T))

    ARR=reshape(1*VEC[7:end,1],(N_FR,N_T))

    return ARR, gr_fr, gr_T
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
