using Plots, FileIO, MAT
include("reach.jl")

# Returns H-rep of various input sets
function input_constraints_pendulum(weights, type::String)
	# Each input specification is in the form Ax≤b
	if type == "pendulum"
		A = [1. 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 1; 0 0 -1]
		b = [π, π, π, π, 5, 5]
	elseif type == "box"
		in_dim = size(weights[1],2) - 1
		A_pos = Matrix{Float64}(I, in_dim, in_dim)
		A_neg = Matrix{Float64}(-I, in_dim, in_dim)
		A = vcat(A_pos, A_neg)
		b = 0.01*ones(2*in_dim)
	elseif type == "hexagon"
		A = [1. 0; -1 0; 0 1; 0 -1; 1 1; -1 1; 1 -1; -1 -1]
		b = [5., 5, 5, 5, 8, 8, 8, 8]
	else
		error("Invalid input constraint specification.")
	end

	return A, b
end

# Returns H-rep of various output sets
function output_constraints_pendulum(weights, type::String)
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
function plot_hrep_pendulum(state2constraints)
	plt = plot(reuse = false, legend=false, xlabel="Angle (deg.)", ylabel="Angular Velocity (deg./s.)")
	for state in keys(state2constraints)
		A, b = state2constraints[state]
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
# Two layers of 16 neurons each: 76 regions
# Can make this 20 for far better accuracy but ~250 regions
copies = 1 # copies = 1 is original network
nn_weights = "models/Pendulum/weights_controlled.npz"
nn_params = "models/Pendulum/norm_params_controlled.npz"
weights = pytorch_net(nn_weights, nn_params, copies)

# Load control invariant set
Ab = matread("models/Pendulum/cntrl_invariant.mat")["Ab"]
A_ctrl, b_ctrl = Ab[:,1:2], Ab[:,3]

Aᵢ, bᵢ = input_constraints_pendulum(weights, "pendulum")
Aₒ, bₒ = output_constraints_pendulum(weights, "origin")

# Run algorithm
# @time begin
# state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ])
# end
# @show length(state2input)

# Plot all regions #
# plt_in1  = plot_hrep_pendulum(state2input)
# plt_in2  = plot_hrep_pendulum(state2backward[1])
# plt_out = plot_hrep_pendulum(state2output)
# plot((180/π)*HPolytope(constraints_list(A_ctrl, b_ctrl)), reuse = false, legend=false, title="Control Invariant Set", xlabel="Angle (deg.)", ylabel="Angular Velocity (deg./s.)")


# save("models/Pendulum/pendulum_controlled_pwa.jld2", Dict("state2input" => state2input, "state2map" => state2map, "Ai" => Aᵢ, "bi" => bᵢ))



