using Plots, FileIO, JLD2
include("reach.jl")



### CONSTRAINTS ###
function input_constraints_quadratic(type::String)
	# Each input specification is in the form Ax≤b
	if type == "box"
		in_dim = 2
		A_pos = Matrix{Float64}(I, in_dim, in_dim)
		A_neg = Matrix{Float64}(-I, in_dim, in_dim)
		A = vcat(A_pos, A_neg)
		b = [1.0, 1.0, 1.0, 1.0]
	elseif type == "hexagon"
		A = [1 0; -1 0; 0 1; 0 -1; 1 1; -1 1; 1 -1; -1 -1]
		b = [5, 5, 5, 5, 8, 8, 8, 8]
	else
		error("Invalid input constraint specification.")
	end
	return A, b
end


function output_constraints_quadratic(type::String)
	# Each output specification is in the form Ay≤b
	if type == "origin"
		A = [1. 0.; -1. 0.; 0. 1.; 0. -1.]
		b = [1., 1., 1., 1.]
 	else 
 		error("Invalid input constraint specification.")
 	end
 	return A, b
end




### PLOTTING ###
function plot_hrep_quadratic(ap2input; type="normal")
	# Plot all polytopes
	plt = plot(reuse = false, legend=false)
	for ap in keys(ap2input)
		A, b = ap2input[ap]
		reg = HPolytope(constraints_list(A, b))
	
		# sanity check
		if isempty(reg)
			@show reg
			error("Empty polyhedron.")
		end

		# static or gif plot
		if type == "normal"
			plot!(plt,  reg, xlabel="x₁", ylabel="x₂", xlims=(-1, 1), ylims=(-1, 1), fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
		elseif type == "gif"
			plot!(plt,  reg, xlabel="x₁", ylabel="x₂", xlims=(-1, 1), ylims=(-1, 1), fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
		end
	end
	return plt
end








### SCRIPTING ###

# find explicit PWA representation
# load network weights
nn_weights = "models/quadratic/weights.npz"
nn_params = "models/quadratic/norm_params.npz"
weights = pytorch_net(nn_weights, nn_params, 1)

# set domain, output constraints
Aᵢ, bᵢ = input_constraints_quadratic("box")
Aₒ, bₒ = output_constraints_quadratic("origin")

# solve for explicit PWA representation
@time begin
ap2input, ap2output, ap2map, ap2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ])
end

# plot input space
plt = plot_hrep_quadratic(ap2input)

# save to jld2 for plotting in matlab later
# save("models/quadratic/quadratic_pwa.jld2", Dict("ap2map" => ap2map, "ap2input" => ap2input, "Aᵢ" => Aᵢ, "bᵢ" => bᵢ))