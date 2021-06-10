using Plots
include("reach.jl")

# Returns H-rep of various input sets
function input_constraints_pendulum(weights, type::String; net_dict=[])
	# Each input specification is in the form Ax≤b
	# The network takes normalized inputs: xₙ = Aᵢₙx + bᵢₙ
	# Thus the input constraints for raw network inputs is: A*inv(Aᵢₙ)x ≤ b + A*inv(Aᵢₙ)*bᵢₙ
	if type == "pendulum"
		# Square. ⨦1 rad , ⨦ 1 rad/s
		A = [1 0; -1 0; 0 1; 0 -1]
		b = [deg2rad(90), deg2rad(90), deg2rad(90), deg2rad(90)]
		# C = Float64.(Diagonal(vec(net_dict["X_std"])))
		# d = vec(net_dict["X_mean"])
		# Aᵢ = A*C
		# bᵢ = b - A*d	
	elseif type == "box"
		in_dim = size(weights[1],2) - 1
		Aᵢ_pos = Matrix{Float64}(I, in_dim, in_dim)
		Aᵢ_neg = Matrix{Float64}(-I, in_dim, in_dim)
		Aᵢ = vcat(Aᵢ_pos, Aᵢ_neg)
		bᵢ = 0.01*ones(2*in_dim)
	elseif type == "hexagon"
		Aᵢ = [1 0; -1 0; 0 1; 0 -1; 1 1; -1 1; 1 -1; -1 -1]
		bᵢ = [5, 5, 5, 5, 8, 8, 8, 8]
	else
		error("Invalid input constraint specification.")
	end

	Aᵢₙ, bᵢₙ = net_dict["input_norm_map"]
	return A*inv(Aᵢₙ), b + A*inv(Aᵢₙ)*bᵢₙ

	# return Aᵢ, bᵢ
end

# Returns H-rep of various output sets
function output_constraints_pendulum(weights, type::String; net_dict=[])
	# Each output specification is in the form Ay≤b
	# The network takes normalized inputs: yₒᵤₜ = Aₒᵤₜy + bₒᵤₜ
	# Thus the output constraints for raw network outputs is: A*inv(Aₒᵤₜ)y ≤ b + A*inv(Aₒᵤₜ)*bₒᵤₜ
	if type == "origin"
		A = [1 0; -1 0; 0 1; 0 -1]
		b = deg2rad.([5, 5, 2, 2])
		# σ = Diagonal(vec(net_dict["Y_std"]))
		# μ = vec(net_dict["Y_mean"])
		# Aₒ = (180/π)*A*σ
		# bₒ = b - (180/π)*A*μ
 	else 
 		error("Invalid input constraint specification.")
 	end
 	Aₒᵤₜ, bₒᵤₜ = net_dict["output_unnorm_map"]
 	return A*inv(Aₒᵤₜ), b + A*inv(Aₒᵤₜ)*bₒᵤₜ
 	# return Aₒ, bₒ
end


# Plot all polytopes
function plot_hrep_pendulum(state2constraints, net_dict; space = "input")
	plt = plot(reuse = false, legend=false, xlabel="Angle (deg.)", ylabel="Angular Velocity (deg./s.)")
	for state in keys(state2constraints)
		A, b = state2constraints[state]
		if space == "input"				
			# C = Diagonal(vec(net_dict["X_std"]))
			# d = vec(net_dict["X_mean"])
			C, d = net_dict["input_norm_map"]
		elseif space == "output"
			# C = Diagonal(vec(net_dict["Y_std"]))
			# d = vec(net_dict["Y_mean"])
			C, d = net_dict["output_unnorm_map"]
		else
			error("Invalid arg given for space")
		end
		reg = Float64.(C)*HPolytope(constraints_list(A,b)) + Float64.(d)
		reg = (180/π)*reg # Convert from rad to deg
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
# copies = 49 # copies = 0 is original network
# model = "models/Pendulum/NN_params_pendulum_0_1s_1e7data_a15_12_2_L1.mat"

# weights, net_dict = pendulum_net(model, copies)
# weights2, net_dict2 = pendulum_net2(model, copies+1)

# Aᵢ, bᵢ = input_constraints_pendulum(weights, "pendulum", net_dict=net_dict2)
# Aₒ, bₒ = output_constraints_pendulum(weights, "origin", net_dict=net_dict2)

# # Run algorithm
# @time begin
# state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], reach=false, back=true, verification=false)
# end
# @show length(state2input)

# Plot all regions #
# plt_in1  = plot_hrep_pendulum(state2input, net_dict2, space="input")
plt_in2  = plot_hrep_pendulum(state2backward[1], net_dict2, space="input")
# plt_out = plot_hrep_pendulum(state2output, net_dict, space="output")
