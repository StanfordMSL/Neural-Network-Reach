using Plots, FileIO, MAT
include("reach.jl")

# Returns H-rep of various input sets
function input_constraints_pendulum(type::String)
	# Each input specification is in the form Ax≤b
	if type == "pendulum"
		A = [1. 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 1; 0 0 -1]
		b = [π, π, π, π, 5, 5]
	else
		error("Invalid input constraint specification.")
	end

	return A, b
end

# Returns H-rep of various output sets
function output_constraints_pendulum(type::String)
	# Each output specification is in the form Ayₒᵤₜ≤b
	# The raw network outputs are unnormalized: yₒᵤₜ = Aₒᵤₜy + bₒᵤₜ
	# Thus the output constraints for raw network outputs are: A*Aₒᵤₜ*y ≤ b - A*bₒᵤₜ
	if type == "origin"
		A = [1. 0; -1 0; 0 1; 0 -1]
		b = deg2rad.([5., 5, 2, 2])
 	else 
 		error("Invalid input constraint specification.")
 	end
 	return A, b
end


# Plot all polytopes
function plot_hrep(inv_set)
	plt = plot(reuse = false, legend=false, xlabel="Angle (deg.)", ylabel="Angular Velocity (deg./s.)")
	for (A,b) in inv_set
		reg = (180/π)*HPolytope(constraints_list(A,b)) # Convert from rad to deg for plotting
		
		if isempty(reg)
			@show reg
			error("Empty polyhedron.")
		end
		plot!(plt, reg, fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
	end
	return plt
end


###########################
######## SCRIPTING ########
###########################

# Load network
copies = 1 # copies = 1 is original network
nn_weights = "models/Pendulum/weights_controlled.npz"
nn_params = "models/Pendulum/norm_params_controlled.npz"
weights = pytorch_net(nn_weights, nn_params, copies)

Aᵢ, bᵢ = input_constraints_pendulum("pendulum")
Aₒ, bₒ = output_constraints_pendulum("origin")


# Run algorithm
@time begin
ap2input, ap2output, ap2map, ap2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ])
end
@show length(ap2input)


# Load and plot control invariant set
inv_dict = load("models/Pendulum/pendulum_controlled_inv_set.jld2",)
inv_set = inv_dict["inv_set"]
plt = plot_hrep(inv_set)
