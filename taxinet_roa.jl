using Plots
include("reach.jl")
include("invariance.jl")

# Returns H-rep of various input sets
function input_constraints_taxinet(weights)
	A = [1. 0.; -1. 0.; 0. 1.; 0. -1.]
	# b = [0.05, 0.05, 0.05, 0.05]
	b = [5.0, 5.0, 2.3, 2.3]
	# b = [-1.05, 1.15, -0.1, 0.15]
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
		reg = HPolytope(constraints_list(A, b))
	
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
fp = [-1.089927713157323, -0.12567755953751042]
A_roa = [-0.5989801202026732 -0.18513226080861578; 0.012078943003279052 0.040868423421668826; -0.41121213687204483 -0.040694624447252484; 0.4111192278275168 0.11499869189110568; 0.13786071030857897 -0.026813295697633487; -0.03515382829869035 -0.07641852864614132; -0.6771039664098355 -0.15917355099470454]
b_roa = [0.9867664026962134, 1.0030239594320969, 0.9973298133488417, 1.00818696973037, 0.9978915112240178, 0.9943563303365947, 0.9887604260847158]


# Run algorithm
@time begin
state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ])
# state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [A_roa], [b_roa], fp=fp, reach=false, back=true, connected=true)
end
@show length(state2input)
# @show length(state2backward[1])


# Plot all regions #
# plt_in1  = plot_hrep_taxinet(state2input)
# plt_in2  = plot_hrep_taxinet(state2backward[1])


# determine if function is a homeomorphism
homeomorph = is_homeomorphism(state2map, size(Aᵢ,2))
println("PWA function is a homeomorphism: ", homeomorph)

# save pwa map 
using FileIO
save("models/taxinet/taxinet_pwa_map.jld2", Dict("state2map" => state2map, "state2input" => state2input, "Aᵢ" => Aᵢ, "bᵢ" => bᵢ))


# find any fixed points if they exist
# fixed_points, fp_dict = find_fixed_points(state2map, state2input, weights)
# @show fixed_points


# find an attractive fixed point and return it's affine function y=Cx+d and polytope Aₓx≤bₓ
# fp, C, d, Aₓ, bₓ = find_attractor(fixed_points, fp_dict)




# Getting mostly suboptimal SDP here
# A_roa, b_roa, fp, state2backward_chain, plt_in2 = find_roa("taxinet", weights, 40, 1)
# @show plt_in2