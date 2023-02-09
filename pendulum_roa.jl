<<<<<<< Updated upstream
using Plots, MATLAB
include("reach.jl")
include("invariance.jl")

# Returns H-rep of various input sets
function input_constraints_pendulum(weights, type::String)
	# Each input specification is in the form Ax≤b
	# The network takes normalized inputs: xₙ = Aᵢₙx + bᵢₙ
	# Thus the input constraints for raw network inputs are: A*inv(Aᵢₙ)x ≤ b + A*inv(Aᵢₙ)*bᵢₙ
	if type == "pendulum"
		A = [1. 0.; -1. 0.; 0. 1.; 0. -1.]
		b = deg2rad.([90, 90, 90, 90])
	elseif type == "box"
		in_dim = size(weights[1],2) - 1
		A_pos = Matrix{Float64}(I, in_dim, in_dim)
		A_neg = Matrix{Float64}(-I, in_dim, in_dim)
		A = vcat(A_pos, A_neg)
		b = 0.01*ones(2*in_dim)
	elseif type == "hexagon"
		A = [1. 0.; -1. 0.; 0. 1.; 0. -1.; 1. 1.; -1. 1.; 1. -1.; -1. -1.]
		b = [5., 5., 5., 5., 8., 8., 8., 8.]
	else
		error("Invalid input constraint specification.")
	end

	return A, b
end

# Returns H-rep of various output sets
function output_constraints_pendulum(weights, type::String)
	if type == "origin"
		A = [1. 0.; -1. 0.; 0. 1.; 0. -1.]
		b = deg2rad.([5, 5, 2, 2])
 	else 
 		error("Invalid input constraint specification.")
 	end
 	return A, b
end


# Plot all polytopes
function plot_hrep_pendulum(state2constraints; type="normal")
	plt = plot(reuse = false, legend=false, xlabel="Angle (deg.)", ylabel="Angular Velocity (deg./s.)")
	for state in keys(state2constraints)
		A, b = state2constraints[state]
		reg = (180/π)*HPolytope(constraints_list(A,b)) # Convert from rad to deg for plotting
		if isempty(reg)
			@show reg
			error("Empty polyhedron.")
		end

		if type == "normal"
			plot!(plt,  reg, fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
		elseif type == "gif"
			plot!(plt,  reg, xlims=(-90, 90), ylims=(-90, 90), fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
		end
	end
	return plt
end


# make gif of backwards reachable set over time
function BRS_gif(model, Aᵢ, bᵢ, Aₛ, bₛ, steps)
	model = "models/Pendulum/NN_params_pendulum_0_1s_1e7data_a15_12_2_L1.mat"
	plt = plot((180/π)*HPolytope(constraints_list(Aₛ, bₛ)), xlims=(-90, 90), ylims=(-90, 90))
	# Way 1
	anim = @animate for Step in 2:steps
		weights = pendulum_net(model, Step)
		state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₛ], [bₛ], reach=false, back=true, verification=false)
    	plt = plot_hrep_pendulum(state2backward[1], type="gif")
	end
	gif(anim, string("brs_",steps  ,".gif"), fps = 2)
end


###########################
######## SCRIPTING ########
###########################
# Given a network representing discrete-time autonomous dynamics and state constraints,
# ⋅ find fixed points
# ⋅ verify the fixed points are stable equilibria
# ⋅ compute invariant polytopes around the fixed points
# ⋅ perform backwards reachability to estimate the maximal region of attraction in the domain

# Getting mostly suboptimal SDP here
nn_file = "models/Pendulum/NN_params_pendulum_0_1s_1e7data_a15_12_2_L1.mat"
A_roa, b_roa, fp, state2backward_chain[1], plt_in2 = find_roa("pendulum", nn_file, 50, 5)
# matwrite("models/Pendulum/pendulum_seed.mat", Dict("A_roa" => A_roa, "b_roa" => b_roa))

@show plt_in2

# Create gif of backward reachable set
=======
using Plots, MATLAB
include("reach.jl")
include("invariance.jl")

# Returns H-rep of various input sets
function input_constraints_pendulum(weights, type::String)
	# Each input specification is in the form Ax≤b
	# The network takes normalized inputs: xₙ = Aᵢₙx + bᵢₙ
	# Thus the input constraints for raw network inputs are: A*inv(Aᵢₙ)x ≤ b + A*inv(Aᵢₙ)*bᵢₙ
	if type == "pendulum"
		A = [1. 0.; -1. 0.; 0. 1.; 0. -1.]
		b = deg2rad.([90, 90, 90, 90])
	elseif type == "box"
		in_dim = size(weights[1],2) - 1
		A_pos = Matrix{Float64}(I, in_dim, in_dim)
		A_neg = Matrix{Float64}(-I, in_dim, in_dim)
		A = vcat(A_pos, A_neg)
		b = 0.01*ones(2*in_dim)
	elseif type == "hexagon"
		A = [1. 0.; -1. 0.; 0. 1.; 0. -1.; 1. 1.; -1. 1.; 1. -1.; -1. -1.]
		b = [5., 5., 5., 5., 8., 8., 8., 8.]
	else
		error("Invalid input constraint specification.")
	end

	return A, b
end

# Returns H-rep of various output sets
function output_constraints_pendulum(weights, type::String)
	if type == "origin"
		A = [1. 0.; -1. 0.; 0. 1.; 0. -1.]
		b = deg2rad.([5, 5, 2, 2])
 	else 
 		error("Invalid input constraint specification.")
 	end
 	return A, b
end


# Plot all polytopes
function plot_hrep_pendulum(state2constraints; type="normal")
	plt = plot(reuse = false, legend=false, xlabel="Angle (deg.)", ylabel="Angular Velocity (deg./s.)")
	for state in keys(state2constraints)
		A, b = state2constraints[state]
		reg = (180/π)*HPolytope(constraints_list(A,b)) # Convert from rad to deg for plotting
		if isempty(reg)
			@show reg
			error("Empty polyhedron.")
		end

		if type == "normal"
			plot!(plt,  reg, fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
		elseif type == "gif"
			plot!(plt,  reg, xlims=(-90, 90), ylims=(-90, 90), fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
		end
	end
	return plt
end


# make gif of backwards reachable set over time
function BRS_gif(model, Aᵢ, bᵢ, Aₛ, bₛ, steps)
	model = "models/Pendulum/NN_params_pendulum_0_1s_1e7data_a15_12_2_L1.mat"
	plt = plot((180/π)*HPolytope(constraints_list(Aₛ, bₛ)), xlims=(-90, 90), ylims=(-90, 90))
	# Way 1
	anim = @animate for Step in 2:steps
		weights = pendulum_net(model, Step)
		state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₛ], [bₛ], reach=false, back=true, verification=false)
    	plt = plot_hrep_pendulum(state2backward[1], type="gif")
	end
	gif(anim, string("brs_",steps  ,".gif"), fps = 2)
end


###########################
######## SCRIPTING ########
###########################
# Given a network representing discrete-time autonomous dynamics and state constraints,
# ⋅ find fixed points
# ⋅ verify the fixed points are stable equilibria
# ⋅ compute invariant polytopes around the fixed points
# ⋅ perform backwards reachability to estimate the maximal region of attraction in the domain

# Getting mostly suboptimal SDP here
nn_file = "models/Pendulum/NN_params_pendulum_0_1s_1e7data_a15_12_2_L1.mat"
A_roa, b_roa, fp, state2backward_chain[1], plt_in2 = find_roa("pendulum", nn_file, 50, 5)
# matwrite("models/Pendulum/pendulum_seed.mat", Dict("A_roa" => A_roa, "b_roa" => b_roa))

@show plt_in2

# Create gif of backward reachable set
>>>>>>> Stashed changes
# BRS_gif(model, Aᵢ, bᵢ, A_roa, b_roa, 5)