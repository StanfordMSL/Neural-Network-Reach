include("reach.jl")
include("invariance.jl")


# Test Polytopic ROA #

A = [0.1419 -0.0856; -0.2574 0.6743]
Fₓ = [0. 0.5; 0.8581 1.0856]
Fₓ = vcat(Fₓ, -Fₓ)
Ω01 = vcat([0.1577*8 -0.6586*8; -0.1577*8 0.6586*8], Fₓ)
Ω02 = vcat([1. -2.; -1. 2.], Fₓ)
Ω2 = vcat(0.2995*[1. -2.; -1. 2.], Fₓ)
Ω_inf = [ 0.1577 -0.6586;  0.8581  1.0856; 
		 -0.1577  0.6586; -0.8581 -1.0856]

# plot regions #
plt1 = plot(HPolytope(constraints_list(Fₓ, ones(size(Fₓ,1)))))
plot!(plt1, HPolytope(Ω01, ones(size(Ω01, 1))), label="Ω01")
plot!(plt1, HPolytope(Ω_inf, ones(size(Ω_inf, 1))), label="Ω_inf")
# plot!(plt, HPolytope(Ω_inf*inv(A), ones(size(Ω_inf, 1))), label="Ω_inf_+")

plt2 = plot(HPolytope(constraints_list(Fₓ, ones(size(Fₓ,1)))))
plot!(plt2, HPolytope(Ω02, ones(size(Ω02, 1))), label="Ω02")
plot!(plt2, HPolytope(Ω_inf, ones(size(Ω_inf, 1))), label="Ω_inf")


# Check Algorithm 1 #
A1 ,b1 = invariant_polytope(Fₓ, ones(size(Fₓ,1)), [0.1577*8 -0.6586*8; -0.1577*8 0.6586*8], ones(2), A)
plot!(plt1, HPolytope(A1, b1), label="Ω01_opt")
plot!(plt1, HPolytope(A1*inv(A), b1), label="Ω01_opt_+")

A2, b2 = invariant_polytope(Fₓ, ones(size(Fₓ,1)), [1. -2.; -1. 2.], ones(2), A)
plot!(plt2, HPolytope(A2, b2), label="Ω02_opt")
plot!(plt2, HPolytope(A2*inv(A), b2), label="Ω02_opt_+")


# Test Ω_inf is actually invariant #
println("Ω_01 proven invariant? ", is_invariant(Ω01, A))
println("Ω_01_opt proven invariant? ", is_invariant(A1, A))
println("Ω1 = Ω_inf proven invariant? ", is_invariant(Ω_inf, A))
println("Ω_02 proven invariant? ", is_invariant(Ω02, A))
println("Ω_02_opt proven invariant? ", is_invariant(A2, A))
println("Ω_2 proven invariant? ", is_invariant(Ω2, A))