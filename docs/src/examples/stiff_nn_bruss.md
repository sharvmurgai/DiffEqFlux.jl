# Learning the Brusselator Equation with a Universal Differential Equation (UDE)

This document walks through an example of using a Universal Differential Equation (UDE) to learn the dynamics of a reaction-diffusion system known as the Brusselator.

### Context: The Brusselator and UDEs

The **Brusselator** is a partial differential equation (PDE) that models a theoretical autocatalytic chemical reaction. It describes how the concentrations of two chemical species evolve over space and time, governed by two main processes:
1.  **Reaction**: The species interact with each other, changing their concentrations locally.
2.  **Diffusion**: The species spread out over the spatial domain.

A **Universal Differential Equation (UDE)** is a hybrid modeling approach that merges known physical laws with machine learning. The core idea is to encode the parts of the system you understand (e.g., diffusion) directly into the equations and use a neural network to learn the parts you don't (e.g., the complex reaction kinetics).

In this example, we will:
1.  Generate "ground truth" data by solving the full, known Brusselator PDE.
2.  Define a UDE with explicitly coded diffusion and partial Brusselator kinetics, then use a neural residual to learn the missing contribution.
3.  Train the neural network's parameters by requiring the UDE's solution to match the ground truth data.
4.  Visualize the results to confirm that our UDE has successfully learned the unknown reaction dynamics.

This showcases the power of scientific machine learning (SciML) to discover governing equations from data.

---

### 1. Problem Setup and Dependencies

First, we import the necessary Julia libraries. `OrdinaryDiffEq.jl` provides the ODE solvers, alongside `Lux.jl`, `Optimization.jl`, and `Plots.jl`. The rendered example uses a bounded grid and horizon; set `STIFF_FULL=true` to select the larger configuration for a local run.


```@example stiff_bruss
using OrdinaryDiffEq
using LinearSolve
using SciMLBase
using ADTypes: AutoForwardDiff
using Lux
using Optimization, OptimizationOptimisers
using Optimisers
using NNlib
using Random
using Plots
using Base.Threads
using ComponentArrays
using Dates
# We disable the default plot saving and display them directly.

# 1. Problem Setup: Constants, Grid, and Initial Conditions

# -- Simulation Parameters --
const FULL_RUN  = get(ENV, "STIFF_FULL", "false") == "true"
const N         = FULL_RUN ? 16 : 8
const TEND      = FULL_RUN ? 12.0f0 : 2.0f0
const MAXITERS  = FULL_RUN ? 200 : 20
const RTOL_REF  = 1f-6
const ATOL_REF  = 1f-6

# -- Grid and Discretization --
const xyd = range(0f0; step=1f0 / N, length=N)
const dx = step(xyd)
const α = 10.0f0
const αdx2 = α / (dx*dx)

@inline limit(a::Int, N::Int) = a == N + 1 ? 1 : (a == 0 ? N : a)
@inline function periodic_laplacian(u, i, j, k, N)
    return u[limit(i-1,N),j,k] + u[limit(i+1,N),j,k] +
           u[i,limit(j-1,N),k] + u[i,limit(j+1,N),k] - 4f0*u[i,j,k]
end
const cosine_field = repeat(
    [cos(2f0 * Float32(π) * Float32(i-1) / N) for i in 1:N, j in 1:N, k in 1:1],
    1, 1, 2,
)
const discrete_cosine_laplacian = 2f0 * (cos(2f0 * Float32(π) / N) - 1f0)
for (i,j) in ((1,1), (cld(N,2), cld(N,2)))
    for species in 1:2
        measured_diffusion = αdx2 * periodic_laplacian(cosine_field, i, j, species, N)
        expected_diffusion = αdx2 * discrete_cosine_laplacian * cosine_field[i,j,species]
        @assert isapprox(measured_diffusion, expected_diffusion; rtol=8f-6, atol=8f-6)
    end
end
@inline brusselator_f(x, y, t) =
    (((x - 0.3f0)^2 + (y - 0.6f0)^2) <= 0.1f0^2) * (t >= 1.1f0) * 5.0f0

function init_u0(xyd)
    N = length(xyd)
    u = zeros(Float32, N, N, 2)
    @inbounds for I in CartesianIndices((N,N))
        x = Float32(xyd[I[1]]); y = Float32(xyd[I[2]])
        u[I,1] = 22f0 * (y*(1f0-y))^(3f0/2f0)
        u[I,2] = 27f0 * (x*(1f0-x))^(3f0/2f0)
    end
    return u
end
const u0   = init_u0(xyd)
```

### 2. Generating the Reference Solution (Ground Truth)
To train our UDE, we need data to learn from. We generate this by solving the full Brusselator PDE with its known equations. This solution will serve as our "ground truth" that we will try to replicate with the UDE. The rhs_ref! function defines the complete dynamics, including both diffusion and reaction terms.

```@example stiff_bruss
println("Stage 1/4: Ground truth not found. Generating..."); flush(stdout)

function rhs_ref!(du, u, p, t)
    A, B, αval = p
    αval_dx2 = αval / (dx*dx)
    @inbounds for I in CartesianIndices((N, N))
        i, j = Tuple(I)
        x, y = xyd[i], xyd[j]
        ip1, im1 = limit(i+1,N), limit(i-1,N)
        jp1, jm1 = limit(j+1,N), limit(j-1,N)
        u1 = u[i,j,1]; v1 = u[i,j,2]
        lap_u = periodic_laplacian(u, i, j, 1, N)
        lap_v = periodic_laplacian(u, i, j, 2, N)
        du[i,j,1] = αval_dx2*lap_u + B + u1^2*v1 - (A+1f0)*u1 + brusselator_f(x,y,t)
        du[i,j,2] = αval_dx2*lap_v + A*u1 - u1^2*v1
    end
    return nothing
end

p_ref    = (3.4f0, 1.0f0, 10.0f0)
prob_ref = ODEProblem(rhs_ref!, u0, (0.0f0, TEND), p_ref)
sol_ref  = solve(prob_ref, KenCarp47(linsolve=LinearSolve.KrylovJL_GMRES());
                    saveat=0.0f0:0.5f0:TEND, reltol=RTOL_REF, abstol=ATOL_REF)
@assert SciMLBase.successful_retcode(sol_ref)
@assert sol_ref.t == collect(0.0f0:0.5f0:TEND)

global Yref = Array(sol_ref)
global ts   = sol_ref.t

println("Ground truth loaded. Size: ", size(Yref))
```

### 3. Defining the Neural Network
The model places a trainable diagonal scaling between two dense subnetworks. Its initial weights decrease with the mode index. `ComponentArray` exposes the structured Lux parameters as a vector for the optimizer.

```@example stiff_bruss
# 3. Neural Network (UDE Component)

# This section defines the neural network architecture that will learn the
# unknown reaction term.

import LuxCore: initialparameters, initialstates

"""
Nonnegative diagonal scaling with mode-indexed weights.
"""
struct SigmaDiag <: Lux.AbstractLuxLayer
    kind::Symbol  # :logistic or :exp
end
function initialparameters(::AbstractRNG, Σ::SigmaDiag)
    if Σ.kind === :exp
        return (β = 1.0f0,)
    else
        return (a = -3.0f0, b = 4.0f0)  # logistic: σ_k = σ(a*k + b)
    end
end
initialstates(::AbstractRNG, ::SigmaDiag) = NamedTuple()
function (Σ::SigmaDiag)(x::AbstractVector{T}, ps, st) where {T}
    k = T.(1:length(x))
    σ = Σ.kind === :exp      ? exp.(-ps.β .* k) :
        Σ.kind === :logistic ? one(T) ./(one(T) .+ exp.(-(ps.a .* k .+ ps.b))) :
        error("Unknown Σ.kind=$(Σ.kind)")
    return σ .* x, st
end

"""
StiffUVSigma: F(x) = U( Σ( V(x) ) )   with U,V: ℝ^m→ℝ^m and diagonal Σ.
"""
struct StiffUVSigma{V,U,S} <: Lux.AbstractLuxLayer
    V::V
    Σ::S
    U::U
end
function initialparameters(rng::AbstractRNG, M::StiffUVSigma)
    return (V = initialparameters(rng, M.V),
            Σ = initialparameters(rng, M.Σ),
            U = initialparameters(rng, M.U))
end
function initialstates(rng::AbstractRNG, M::StiffUVSigma)
    return (V = initialstates(rng, M.V),
            Σ = initialstates(rng, M.Σ),
            U = initialstates(rng, M.U))
end
function (M::StiffUVSigma)(x, ps, st)
    yV, stV = M.V(x, ps.V, st.V)
    yΣ, stΣ = M.Σ(yV, ps.Σ, st.Σ)
    yU, stU = M.U(yΣ, ps.U, st.U)
    return yU, (V = stV, Σ = stΣ, U = stU)
end

# Model: U( Σ V(x) ) with m=2 state; H is width in U/V
const H = 16
const STIFF_KIND = Symbol(get(ENV, "STIFF_KIND", "logistic"))  # :logistic | :exp

Random.seed!(1234)
Vnet = Chain(Dense(2 => H, tanh), Dense(H => 2))
Σlay = SigmaDiag(STIFF_KIND)
Unet = Chain(Dense(2 => H, tanh), Dense(H => 2))
model = StiffUVSigma(Vnet, Σlay, Unet)

rng   = Random.default_rng()
ps0, st0 = Lux.setup(rng, model)
const ST = st0
θ0 = ComponentArray(ps0)
```

### 4. Constructing the Universal Differential Equation (UDE)
Here is the core of the UDE. `rhs_ude!` keeps the known diffusion and Brusselator reaction structure, but starts with incomplete kinetics: `B0 = 0.8` instead of the reference production term `B = 1.0`. The residual network must learn the missing `+0.2` contribution to the first species. This function defines the ODEProblem used for training and evaluation.

```@example stiff_bruss
const RESIDUAL   = true
const COUT       = 8.0
const RESID_COUT = 0.7
const A0         = 3.4
const B0         = 0.8f0

function rhs_ude!(du, u, θ_vec, t)
    Tz = eltype(u)
    @inbounds for I in CartesianIndices((N,N))
        i, j = Tuple(I); x = xyd[i]; y = xyd[j]
        u1 = u[i,j,1]; v1 = u[i,j,2]
        lap_u = periodic_laplacian(u, i, j, 1, N)
        lap_v = periodic_laplacian(u, i, j, 2, N)
        ŷ, _ = model(Tz[u1, v1], θ_vec, ST)
        if RESIDUAL
            r1 = B0 + u1^2*v1 - (A0 + 1f0)*u1
            r2 = A0*u1 - u1^2*v1
            δ1 = clamp(ŷ[1], -RESID_COUT, RESID_COUT)
            δ2 = clamp(ŷ[2], -RESID_COUT, RESID_COUT)
            du[i,j,1] = αdx2*lap_u + (r1 + δ1) + brusselator_f(x,y,t)
            du[i,j,2] = αdx2*lap_v + (r2 + δ2)
        else
            y1 = clamp(ŷ[1], -COUT, COUT); y2 = clamp(ŷ[2], -COUT, COUT)
            du[i,j,1] = αdx2*lap_u + y1 + brusselator_f(x,y,t)
            du[i,j,2] = αdx2*lap_v + y2
        end
    end
    return nothing
end

prob_ude = ODEProblem(rhs_ude!, u0, (0.0f0, TEND), θ0)

# Exercise the reference and UDE RHS stencils on the discrete cosine mode.
# Subtract each known reaction/forcing term to isolate diffusion.
stencil_du_ref = similar(cosine_field)
stencil_du_ude = similar(cosine_field)
rhs_ref!(stencil_du_ref, cosine_field, p_ref, 0.0f0)
rhs_ude!(stencil_du_ude, cosine_field, zero.(θ0), 0.0f0)
for (i,j) in ((1,1), (cld(N,2), cld(N,2))), species in 1:2
    u1 = cosine_field[i,j,1]
    v1 = cosine_field[i,j,2]
    reaction_ref = species == 1 ? p_ref[2] + u1^2*v1 - (p_ref[1] + 1f0)*u1 +
                                 brusselator_f(xyd[i], xyd[j], 0.0f0) :
                                 p_ref[1]*u1 - u1^2*v1
    reaction_ude = species == 1 ? B0 + u1^2*v1 - (A0 + 1f0)*u1 +
                                 brusselator_f(xyd[i], xyd[j], 0.0f0) :
                                 A0*u1 - u1^2*v1
    expected_diffusion = αdx2 * discrete_cosine_laplacian * cosine_field[i,j,species]
    tol = 32f0 * eps(Float32) * max(1f0, abs(expected_diffusion))
    @assert isapprox(stencil_du_ref[i,j,species] - reaction_ref, expected_diffusion;
                     rtol=32f0*eps(Float32), atol=tol)
    @assert isapprox(stencil_du_ude[i,j,species] - reaction_ude, expected_diffusion;
                     rtol=32f0*eps(Float32), atol=tol)
end
```

### 5. Training the UDE
The baseline kinetics deliberately use `B0 = 0.8` while the reference uses `B = 1.0`, leaving a known `+0.2` contribution to the first species for the neural residual to learn. We train on the primary initial condition and evaluate on a distinct held-out initial condition. `AutoForwardDiff` differentiates through this reduced solve directly; this avoids relying on an adjoint gradient that must first agree with an independent reference.

```@example stiff_bruss
println("Stage 2/4: Set loss + optimizer …"); flush(stdout)
const ALG_FINAL   = TRBDF2(autodiff=false)
const FINAL_RTOL  = 1f-4
const FINAL_ATOL  = 1f-4
const FINAL_DTMAX = 0.1f0

# ---- Zygote-safe constants used in loss ----
const LAMBDA = 3e-3

# Curriculum in time for training only
const TRAIN_FRAC = 0.75
const TRAIN_IDX  = 1:clamp(ceil(Int, TRAIN_FRAC * length(ts)), 1, length(ts))
const ts_train   = ts[TRAIN_IDX]
const Yref_train = Yref[:,:,:,TRAIN_IDX]
const W_WEIGHTS  = reshape(1 .+ collect(ts_train)./maximum(ts_train), 1,1,1,length(TRAIN_IDX))

function loss(θ_vec)
    sol = solve(remake(prob_ude; p=θ_vec), TRBDF2(autodiff=false);
                saveat=ts_train, reltol=1f-4, abstol=1f-4,
                save_everystep=false)
    Y = Array(sol)
    if size(Y) != size(Yref_train)
        return Inf32
    end
    data_mse = sum(abs2, (Y .- Yref_train) .* W_WEIGHTS) / length(Yref_train)
    reg = LAMBDA * sum(abs2, θ_vec)
    return data_mse + reg
end

# Check the optimizer's output-bias direction against tolerance-refined
# central differences on the same seeded reduced solve.
import ForwardDiff
bias0 = θ0.U.layer_2.bias[1]
const bias_direction = zero.(θ0)
bias_direction.U.layer_2.bias[1] = 1f0
output_bias(value) = loss(θ0 .+ (value - bias0) .* bias_direction)
grad_ad = ForwardDiff.derivative(output_bias, bias0)
hs = (2f-2, 1f-2, 5f-3)
grad_fd = [(output_bias(bias0 + h) - output_bias(bias0 - h)) / (2h) for h in hs]
fd_error = abs(grad_fd[end] - grad_fd[end-1])
@assert fd_error > 0f0
grad_tol = 4f0 * fd_error
@assert abs(grad_ad - grad_fd[end]) <= grad_tol
println("Output-bias directional derivative: AD=$grad_ad central=$grad_fd tolerance=$grad_tol")

optf    = OptimizationFunction((θ, _)->loss(θ), AutoForwardDiff())
optprob = OptimizationProblem(optf, θ0)

println("Stage 3/4: Training …"); flush(stdout)
k_iter = 0
function cb(state, L)
    global k_iter += 1
    if k_iter % 5 == 0
        println("  it=$k_iter  loss=$(round(L; digits=6))"); flush(stdout)
    end
    return false
end

initial_lr = 0.0010
decay_lr   = 0.00035
switch_it  = 40

function train_with_schedule()
    it1 = min(MAXITERS, switch_it)
    opt1 = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0f0), Optimisers.Adam(initial_lr))
    stg1 = solve(optprob, opt1; maxiters=it1, callback=cb)
    rem = MAXITERS - it1
    if rem > 0
        optprob2 = OptimizationProblem(optf, ComponentArray(stg1.u))
        opt2 = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0f0), Optimisers.Adam(decay_lr))
        stg2 = solve(optprob2, opt2; maxiters=rem, callback=cb)
        return stg2
    else
        return stg1
    end
end

solopt = train_with_schedule()

θ★ = ComponentArray(solopt.u)
```

### 6. Evaluation and Visualization
Finally, we evaluate the optimized UDE against reference data from a held-out initial condition and compare the spatial fields for both species:

The four heatmaps compare both species from the reference and learned solutions at the final time on the held-out initial condition. The printed relative squared field errors include every spatial node and both species. If training succeeds, the held-out error should improve over the zero-residual baseline, and the learned first-species residual should recover the missing `+0.2` at representative states.

```@example stiff_bruss
println("\nStage 4/4: Final evaluation …"); flush(stdout)
println("  Config: N=$(N), TEND=$(TEND), TRAIN_FRAC=$(TRAIN_FRAC)")
println("  Threads during final solve: DISABLED | Algorithm: ", typeof(ALG_FINAL))
t_start = time()

sol_ude_final = solve(remake(prob_ude; p=θ★), ALG_FINAL;
          saveat=ts, reltol=FINAL_RTOL, abstol=FINAL_ATOL,
          save_everystep=false, dtmax=FINAL_DTMAX)

println("  Final solve retcode = ", sol_ude_final.retcode)
println("  Saved points = ", length(sol_ude_final.t))
println("  Final solve wall time = $(round(time()-t_start; digits=3)) s")

u0_holdout = init_u0(xyd) .* reshape(Float32[0.85, 1.1], 1, 1, 2)
prob_holdout = ODEProblem(rhs_ref!, u0_holdout, (0.0f0, TEND), p_ref)
sol_ref_holdout = solve(prob_holdout, KenCarp47(linsolve=LinearSolve.KrylovJL_GMRES());
    saveat=ts, reltol=RTOL_REF, abstol=ATOL_REF)
@assert SciMLBase.successful_retcode(sol_ref_holdout)
prob_ude_holdout = ODEProblem(rhs_ude!, u0_holdout, (0.0f0, TEND), θ0)
θzero = zero.(θ0)
sol_ude_holdout_zero = solve(remake(prob_ude_holdout; p=θzero), ALG_FINAL; saveat=ts,
    reltol=FINAL_RTOL, abstol=FINAL_ATOL, dtmax=FINAL_DTMAX)
sol_ude_holdout = solve(remake(prob_ude_holdout; p=θ★), ALG_FINAL; saveat=ts,
    reltol=FINAL_RTOL, abstol=FINAL_ATOL, dtmax=FINAL_DTMAX)
@assert SciMLBase.successful_retcode(sol_ude_final)
@assert SciMLBase.successful_retcode(sol_ude_holdout_zero)
@assert SciMLBase.successful_retcode(sol_ude_holdout)
@assert sol_ude_final.t == ts && sol_ref_holdout.t == ts && sol_ude_holdout.t == ts
heldout_error(sol) = sum(abs2, Array(sol) .- Array(sol_ref_holdout)) / sum(abs2, Array(sol_ref_holdout))
zero_error = heldout_error(sol_ude_holdout_zero)
trained_error = heldout_error(sol_ude_holdout)
@assert trained_error < zero_error
println("done. Held-out relative squared field error: zero=$zero_error trained=$trained_error")

representative_states = (Float32[1, 1], Float32[2, 0.5], Float32[0.5, 2])
correction = [model(state, θ★, ST)[1][1] for state in representative_states]
@assert all(isapprox.(correction, 0.2f0; atol=0.12f0))
println("Recovered first-species correction at representative states: ", correction)

learned_holdout = Array(sol_ude_holdout)
reference_holdout = Array(sol_ref_holdout)
fields = Any[]
for species in 1:2
    push!(fields, heatmap(xyd, xyd, reference_holdout[:,:,species,end],
                          title="Reference species $species"))
    push!(fields, heatmap(xyd, xyd, learned_holdout[:,:,species,end],
                          title="Learned species $species"))
end
plot(fields..., layout=(2,2))
```
