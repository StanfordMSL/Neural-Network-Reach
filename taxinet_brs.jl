using Plots, FileIO
include("reach.jl")
include("merge_poly.jl")

# Returns H-rep of various input sets
function input_constraints_taxinet(weights)
	A = [1. 0. 0.; -1. 0. 0.; 0. 1. 0.; 0. -1. 0.; 0. 0. 1.; 0. 0. -1.]
	b = [10., 10., 10., 10., 30., 30.] # deg rudder command, m offset, deg heading
	return A, b
end

# Returns H-rep of various output sets
function output_constraints_taxinet(weights)
	A1 = [1. 0.; -1. 0.; 0. 1.; 0. -1.]
	b1 = [15., -10., 30., 30.] # m offset, deg heading
	A2 = [1. 0.; -1. 0.; 0. 1.; 0. -1.]
	b2 = [-10., 15., 30., 30.] # m offset, deg heading
 	return [A1, A2], [b1, b2]
end


# Plot all polytopes
function plot_hrep_taxinet(state2constraints; type="normal")
	plt = plot(reuse = false, legend=false, xlabel="x₁", ylabel="x₂")
	for state in keys(state2constraints)
		A, b = state2constraints[state]
		reg = LazySets.project(HPolytope(A, b), [2,3])
	
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

weights = pytorch_net("models/taxinet/weights_dynamics.npz", "models/taxinet/norm_params_dynamics.npz", 1) # [u; x] -> x′

Aᵢ, bᵢ = input_constraints_taxinet(weights)
output_As, output_bs = output_constraints_taxinet(weights)


# Run algorithm
@time begin
state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, output_As, output_bs, back=true)
end
@show length(state2input)
@show length(state2backward[1])
@show length(state2backward[2])


# merged_set = merge_polytopes(P, verbose=true)

# Plot all regions #
plt_in1 = plot_hrep_taxinet(state2input)
plt_in2  = plot_hrep_taxinet(state2backward[1])




# Load in saved function #
# pwa_dict = load("models/taxinet/taxinet_pwa_map.jld2")
# state2input = pwa_dict["state2input"]
# plt_in1 = plot_hrep_taxinet(state2input)