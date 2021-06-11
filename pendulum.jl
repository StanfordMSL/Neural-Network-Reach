using Plots
include("reach.jl")

# Returns H-rep of various input sets
function input_constraints_pendulum(weights, type::String, net_dict)
	# Each input specification is in the form Ax≤b
	# The network takes normalized inputs: xₙ = Aᵢₙx + bᵢₙ
	# Thus the input constraints for raw network inputs are: A*inv(Aᵢₙ)x ≤ b + A*inv(Aᵢₙ)*bᵢₙ
	if type == "pendulum"
		A = [1 0; -1 0; 0 1; 0 -1]
		b = deg2rad.([90, 90, 90, 90])
	elseif type == "box"
		in_dim = size(weights[1],2) - 1
		A_pos = Matrix{Float64}(I, in_dim, in_dim)
		A_neg = Matrix{Float64}(-I, in_dim, in_dim)
		A = vcat(A_pos, A_neg)
		b = 0.01*ones(2*in_dim)
	elseif type == "hexagon"
		A = [1 0; -1 0; 0 1; 0 -1; 1 1; -1 1; 1 -1; -1 -1]
		b = [5, 5, 5, 5, 8, 8, 8, 8]
	else
		error("Invalid input constraint specification.")
	end

	Aᵢₙ, bᵢₙ = net_dict["input_norm_map"]
	return A*inv(Aᵢₙ), b + A*inv(Aᵢₙ)*bᵢₙ
end

# Returns H-rep of various output sets
function output_constraints_pendulum(weights, type::String, net_dict)
	# Each output specification is in the form Ayₒᵤₜ≤b
	# The raw network outputs are unnormalized: yₒᵤₜ = Aₒᵤₜy + bₒᵤₜ
	# Thus the output constraints for raw network outputs are: A*Aₒᵤₜ*y ≤ b - A*bₒᵤₜ
	if type == "origin"
		A = [1 0; -1 0; 0 1; 0 -1]
		b = deg2rad.([5, 5, 2, 2])
 	else 
 		error("Invalid input constraint specification.")
 	end
 	Aₒᵤₜ, bₒᵤₜ = net_dict["output_unnorm_map"]
 	return A*Aₒᵤₜ, b - A*bₒᵤₜ
end


# Plot all polytopes
function plot_hrep_pendulum(state2constraints, net_dict; space = "input")
	plt = plot(reuse = false, legend=false, xlabel="Angle (deg.)", ylabel="Angular Velocity (deg./s.)")
	for state in keys(state2constraints)
		A, b = state2constraints[state]
		if space == "input"	
			# Each cell is in the form Axₙ≤b
			# The network takes normalized inputs: xₙ = Aᵢₙx + bᵢₙ
			# Thus the input constraints are: A*Aᵢₙx ≤ b - A*bᵢₙ
			Aᵢₙ, bᵢₙ = net_dict["input_norm_map"]
			reg = HPolytope(constraints_list(A*Aᵢₙ, b - A*bᵢₙ))
		elseif space == "output"
			# Each cell is in the form Ay≤b
			# The raw network outputs are unnormalized: yₒᵤₜ = Aₒᵤₜy + bₒᵤₜ ⟹ y = inv(Aₒᵤₜ)*(yₒᵤₜ - bₒᵤₜ)
			# Thus the output constraints are: A*inv(Aₒᵤₜ)*y ≤ b + A*inv(Aₒᵤₜ)*bₒᵤₜ
			Aₒᵤₜ, bₒᵤₜ = net_dict["output_unnorm_map"]
			reg = HPolytope(constraints_list(A*inv(Aₒᵤₜ), b + A*inv(Aₒᵤₜ)*bₒᵤₜ))
		else
			error("Invalid arg given for space")
		end
		
		reg = (180/π)*reg # Convert from rad to deg for plotting
		if isempty(reg)
			@show reg
			error("Empty polyhedron.")
		end
		plot!(plt,  reg, fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
	end
	return plt
end


###########################
######## SCRIPTING ########
###########################
copies = 51 # copies = 1 is original network
model = "models/Pendulum/NN_params_pendulum_0_1s_1e7data_a15_12_2_L1.mat"

weights, net_dict = pendulum_net(model, copies)

Aᵢ, bᵢ = input_constraints_pendulum(weights, "pendulum", net_dict)
Aₒ, bₒ = output_constraints_pendulum(weights, "origin", net_dict)

# Run algorithm
@time begin
state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], reach=true, back=true, verification=false)
end
@show length(state2input)

# Plot all regions #
plt_in1  = plot_hrep_pendulum(state2input, net_dict, space="input")
plt_in2  = plot_hrep_pendulum(state2backward[1], net_dict, space="input")
plt_out = plot_hrep_pendulum(state2output, net_dict, space="output")
