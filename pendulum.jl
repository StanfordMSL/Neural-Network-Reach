using Plots
include("reach.jl")

# Returns H-rep of various input sets
function input_constraints_pendulum(weights, type::String)
	# Each input specification is in the form Ax≤b
	if type == "pendulum"
		A = [1. 0; -1 0; 0 1; 0 -1]
		b = deg2rad.([90., 90, 90, 90])
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
function plot_hrep_pendulum(ap2constraints)
	plt = plot(reuse = false, legend=false, xlabel="Angle (deg.)", ylabel="Angular Velocity (deg./s.)")
	for ap in keys(ap2constraints)
		A, b = ap2constraints[ap]
		reg = (180/π)*HPolytope(constraints_list(A,b)) # Convert from rad to deg for plotting
		
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
copies = 50 # copies = 1 is original network
model = "models/Pendulum/NN_params_pendulum_0_1s_1e7data_a15_12_2_L1.mat"
weights = pendulum_net(model, copies)

Aᵢ, bᵢ = input_constraints_pendulum(weights, "pendulum")
Aₒ, bₒ = output_constraints_pendulum(weights, "origin")

# Run algorithm
@time begin
ap2input, ap2output, ap2map, ap2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], reach=true, back=true, verification=false)
end
@show length(ap2input)

# Plot all regions #
plt_in  = plot_hrep_pendulum(ap2input)
plt_in_brs  = plot_hrep_pendulum(ap2backward[1])
plt_out = plot_hrep_pendulum(ap2output)
