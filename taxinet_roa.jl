using Plots
include("reach.jl")
include("invariance.jl")

# Returns H-rep of various input sets
function input_constraints_taxinet(weights)
	A = [1. 0.; -1. 0.; 0. 1.; 0. -1.]
	b = [0.01, 0.01, 0.01, 0.01]
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
weights = taxinet_cl()


Aᵢ, bᵢ = input_constraints_taxinet(weights)
Aₒ, bₒ = output_constraints_taxinet(weights)

# Run algorithm
@time begin
state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ])
# state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [A_roa], [b_roa], fp=fp, reach=false, back=true, connected=true)
end
@show length(state2input)
# @show length(state2backward[1])


# Plot all regions #
plt_in1  = plot_hrep_taxinet(state2input)
# plt_in2  = plot_hrep_taxinet(state2backward[1])

homeomorph = is_homeomorphism(state2map, size(Aᵢ,2))
println("PWA function is a homeomorphism: ", homeomorph)

fixed_points, fp_dict = find_fixed_points(state2map, state2input, weights)
@show fixed_points


# Getting mostly suboptimal SDP here
# A_roa, b_roa, fp, state2backward_chain, plt_in2 = find_roa("vanderpol", nn_weights, 20, 7, nn_params=nn_params)
# @show plt_in2