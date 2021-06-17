include("reach.jl")
include("invariance.jl")


# Test Polytopic ROA #

A = [0.1419 -0.0856; -0.2574 0.6743]
Fₓ = [0. 0.5; 0.8581 1.0856]
Fₓ = vcat(Fₓ, -Fₓ)
Ω01 = vcat([0.1577*8 -0.6586*8; -0.1577*8 0.6586*8], Fₓ)
Ω02 = vcat([1. -2.; -1. 2.], Fₓ)

# λ1 = invariant_polytope(Fₓ, ones(size(Fₓ,1)), [0.1577*8 -0.6586*8; -0.1577*8 0.6586*8], ones(2), A)
# λ2 = invariant_polytope(Fₓ, ones(size(Fₓ,1)), [1. -2.; -1. 2.], ones(2), A)

# Symmetric Algorithm
# λ1_sym = invariant_polytope_sym([0. 0.5; 0.8581 1.0856], ones(2), [0.1577*8 -0.6586*8], [1], A)
λ2_sym = invariant_polytope_sym([0. 0.5; 0.8581 1.0856], ones(2), [1. -2.], [1], A)


# plt = plot(HPolytope(constraints_list(Fₓ, ones(size(Fₓ,1)))))
# plot!(plt, HPolytope(Ω01, ones(size(Ω01, 1))), label="Ω01")
# plot!(plt, HPolytope(Ω02, ones(size(Ω02, 1))), label="Ω02")
