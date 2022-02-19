using Plots, FileIO
include("reach.jl")
include("invariance.jl")

# Returns H-rep of various input sets
function input_constraints_taxinet(weights)
	A = [1. 0.; -1. 0.; 0. 1.; 0. -1.]
	# b = [0.05, 0.05, 0.05, 0.05]
	# b = [5.0, 5.0, 2.3, 2.3] # taxinet_pwa_map.jld
	# b = [10., 10., 2., 2.]   # taxinet_pwa_map_large.jld +- 10 meters, +- 2 degrees
	b = [5., 5., 15., 15.]   # taxinet_pwa_map_5_15.jld +- 5 meters, +- 15 degrees
	# b = [11., 11., 30., 30.] # taxinet_pwa_map_full.jld  +- 10 meters, +- 30 degrees

	# b = [-1.0, 1.11, 0.01, 0.8]
	# b = [11., 11., 30., 30.]
	return A, b
end

# Returns H-rep of various output sets
function output_constraints_taxinet(weights)
	A = [1. 0.; -1. 0.; 0. 1.; 0. -1.]
	b = [1., 1., 1., 1.]
 	return A, b
end


# Plot all polytopes
function plot_hrep_taxinet(state2constraints; type="normal")
	plt = plot(reuse = false, legend=false, xlabel="x₁", ylabel="x₂")
	for state in keys(state2constraints)
		A, b = state2constraints[state]
		reg = HPolytope(A, b)
	
		if isempty(reg)
			@show reg
			error("Empty polyhedron.")
		end

		if type == "normal"
			plot!(plt,  reg, fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
		elseif type == "gif"
			plot!(plt,  reg, xlims=(-3, 3), ylims=(-3, 3), fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
		end
	end
	return plt
end



###########################
######## SCRIPTING ########
###########################
# Given a network representing discrete-time autonomous dynamics and state constraints,
# ⋅ find fixed points
# ⋅ verify the fixed points are stable equilibria
# ⋅ compute invariant polytopes around the fixed points
# ⋅ perform backwards reachability to estimate the maximal region of attraction in the domain

copies = 1 # copies = 1 is original network
weights = taxinet_cl(copies)

Aᵢ, bᵢ = input_constraints_taxinet(weights)
Aₒ, bₒ = output_constraints_taxinet(weights)

# Already found fixed point and seed ROA
# fp = [-1.089927713157323, -0.12567755953751042]
# A_roa = 370*[-0.20814568962857855 0.03271955855771795; 0.2183098663000297 0.12073669880754853; 0.42582101825227686 0.0789033995762251; 0.14480530852927057 -0.05205811047518554; -0.13634673812819695 -0.1155315084750385; 0.04492020060602461 0.09045775648816877; -0.6124506220873154 -0.12811621541510643]
# b_roa = ones(size(A_roa,1)) + A_roa*fp
# reg_roa = HPolytope(A_roa, b_roa)

# Run algorithm
@time begin
state2input, state2output, state2map, state2backward, state2neighbors = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], graph=true)
# state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [A_roa], [b_roa], fp=fp, reach=false, back=true, connected=true)
end
@show length(state2input)
# @show length(state2backward[1])


# save pwa map 
save("models/taxinet/taxinet_pwa_map_5_15.jld2", Dict("ap2map" => state2map, "ap2input" => state2input, "ap2neighbors" => state2neighbors, "Aᵢ" => Aᵢ, "bᵢ" => bᵢ))


# Plot all regions #
# plt_in1 = plot_hrep_taxinet(state2input)
# scatter!(plt_in1, [fp[1]], [fp[2]])
# plot!(plt_in1, reg_roa)

# plt_in2  = plot_hrep_taxinet(state2backward[1])


# determine if function is a homeomorphism
homeomorph = is_homeomorphism(state2map, size(Aᵢ,2))
println("PWA function is a homeomorphism: ", homeomorph)


# find any fixed points if they exist
# fixed_points, fp_dict = find_fixed_points(state2map, state2input, weights)
# @show fixed_points


# find an attractive fixed point and return it's affine function y=Cx+d and polytope Aₓx≤bₓ
# fp, C, d, Aₓ, bₓ = find_attractor(fixed_points, fp_dict)

# Getting mostly suboptimal SDP here
# A_roa, b_roa, fp, state2backward_chain, plt_in2 = find_roa("taxinet", weights, 40, 1)
# @show plt_in2




# Load in saved function #
# pwa_dict = load("models/taxinet/taxinet_pwa_map.jld2")
# state2input = pwa_dict["state2input"]
# plt_in1 = plot_hrep_taxinet(state2input)