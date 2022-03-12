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
# Aₒ, bₒ = output_constraints_taxinet(weights)

# Already found fixed point and seed ROA
# fp = [-1.089927713157323, -0.12567755953751042]
# A_roa = 370*[-0.20814568962857855 0.03271955855771795; 0.2183098663000297 0.12073669880754853; 0.42582101825227686 0.0789033995762251; 0.14480530852927057 -0.05205811047518554; -0.13634673812819695 -0.1155315084750385; 0.04492020060602461 0.09045775648816877; -0.6124506220873154 -0.12811621541510643]
# b_roa = ones(size(A_roa,1)) + A_roa*fp
# reg_roa = HPolytope(A_roa, b_roa)

# Run algorithm
# @time begin
# ap2input, ap2output, ap2map, ap2backward, state2neighbors = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], graph=true)
# # ap2input, ap2output, ap2map, ap2backward = compute_reach(weights, Aᵢ, bᵢ, [A_roa], [b_roa], fp=fp, reach=false, back=true, connected=true)
# end
# @show length(ap2input)
# @show length(ap2backward[1])


# save pwa map
# save("models/taxinet/taxinet_pwa_map_5_15.jld2", Dict("ap2map" => ap2map, "ap2input" => ap2input, "ap2neighbors" => state2neighbors, "Aᵢ" => Aᵢ, "bᵢ" => bᵢ))


# Plot all regions #
# plt_in1 = plot_hrep_taxinet(ap2input)
# scatter!(plt_in1, [fp[1]], [fp[2]])
# plot!(plt_in1, reg_roa)

# plt_in2  = plot_hrep_taxinet(ap2backward[1])

# net_dict = load("models/taxinet/taxinet_pwa_map_5_15.jld2")
# ap2map = net_dict["ap2map"]
# ap2input = net_dict["ap2input"]

# determine if function is a homeomorphism
# homeomorph = is_homeomorphism(ap2map, size(Aᵢ,2))
# println("PWA function is a homeomorphism: ", homeomorph)


# find any fixed points if they exist
# fixed_points, fp_dict = find_fixed_points(ap2map, ap2input, weights)
# @show fixed_points


# find an attractive fixed point and return it's affine function y=Cx+d and polytope Aₓx≤bₓ
# A1, b1 =  fp_dict[fixed_points[1]][1]
# C1, d1 = fp_dict[fixed_points[1]][2]
# fp2, C2, d2, A2, b2 = find_attractor([fixed_points[2]], fp_dict)
# fp3, C3, d3, A3, b3 = find_attractor([fixed_points[3]], fp_dict)


# plt = plot(reuse = false, xlabel="x₁", ylabel="x₂")
# scatter!(plt, [fp2[1]], [fp2[2]], label="fp2")
# scatter!(plt, [fp3[1]], [fp3[2]], label="fp3")
# plot!(HPolytope(A1, b1), label="unstable")
# plot!(HPolytope(A2, b2), label="stable")
# plot!(HPolytope(A3, b3), label="stable")


# num_constraints = 50
# A2_roa, b2_roa = polytope_roa_sdp(A2, b2, C2, fp2, num_constraints)
# A3_roa, b3_roa = polytope_roa_sdp(A3, b3, C3, fp3, num_constraints)

# plot!(HPolytope(A2_roa, b2_roa), label="stable_roa")
# plot!(HPolytope(A3_roa, b3_roa), label="stable_roa")

# save("models/taxinet/5_15/fp_info.jld2", Dict("stable fps" => fixed_points[2:3], "roas" => [(A2_roa, b2_roa), (A3_roa, b3_roa)]))









# Getting mostly suboptimal SDP here
# A_roa, b_roa, fp, ap2backward_chain, plt_in2 = find_roa("taxinet", weights, 40, 1)
# @show plt_in2




# Load in saved function #
# pwa_dict = load("models/taxinet/taxinet_pwa_map.jld2")
# ap2input = pwa_dict["ap2input"]
# plt_in1 = plot_hrep_taxinet(ap2input)