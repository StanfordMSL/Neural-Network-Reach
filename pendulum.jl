include("reach.jl")
pyplot()

bound_r(a,b) = (b-a)*(rand()-1) + b # Generates a uniformly random number on [a,b]


function input_constraints_pendulum(weights, type::String; net_dict=[])
	if type == "pendulum"
		# Square. ⨦1 rad , ⨦ 1 rad/s
		A = [1 0; -1 0; 0 1; 0 -1]
		b = [deg2rad(90), deg2rad(90), deg2rad(90), deg2rad(90)]
		C = Float64.(Diagonal(vec(net_dict["X_std"])))
		d = vec(net_dict["X_mean"])
		Aᵢ = A*C
		bᵢ = b - A*d	
	elseif type == "box"
		in_dim = size(weights[1],2) - 1
		Aᵢ_pos = Matrix{Float64}(I, in_dim, in_dim)
		Aᵢ_neg = Matrix{Float64}(-I, in_dim, in_dim)
		Aᵢ = vcat(Aᵢ_pos, Aᵢ_neg)
		bᵢ = 0.25*ones(2*in_dim)
	elseif type == "hexagon"
		Aᵢ = [1 0; -1 0; 0 1; 0 -1; 1 1; -1 1; 1 -1; -1 -1]
		bᵢ = [5, 5, 5, 5, 8, 8, 8, 8]
	else
		error("Invalid input constraint specification.")
	end
	return Aᵢ, bᵢ
end


function output_constraints_pendulum(weights, type::String; net_dict=[])
	if type == "origin"
		A = [1 0; -1 0; 0 1; 0 -1]
		b = [5, 5, 2, 2]
		σ = Diagonal(vec(net_dict["Y_std"]))
		μ = vec(net_dict["Y_mean"])
		Aₒ = (180/π)*A*σ
		bₒ = b - (180/π)*A*μ
 	else 
 		error("Invalid input constraint specification.")
 	end
 	return Aₒ, bₒ
end


# Plots all polyhedra
function plot_hrep_pendulum(state2constraints, net_dict; space = "input")
	plt = plot(reuse = false, legend=false, xlabel="Angle (deg.)", ylabel="Angular Velocity (deg./s.)")
	for state in keys(state2constraints)
		A, b = state2constraints[state]
		if space == "input"				
			C = Diagonal(vec(net_dict["X_std"]))
			d = vec(net_dict["X_mean"])
		elseif space == "output"
			C = Diagonal(vec(net_dict["Y_std"]))
			d = vec(net_dict["Y_mean"])
		else
			error("Invalid arg given for space")
		end
		reg = Float64.(C)*HPolytope(constraints_list(A,b)) + Float64.(d)
		reg = (180/π)*reg # Convert from rad to deg
		if isempty(reg)
			@show reg
			error("Empty polyhedron.")
		end
		plot!(plt,  reg)
	end
	return plt
end


function damped_plt(init, steps::Int64, net_dict)
	plt = plot(reuse = false)
	t = collect(0:0.1:0.1*steps)
	state_traj = zeros(2,length(t))
	state_traj[:,1] = init
	for i in 2:length(t)
		state_traj[:,i] = eval_net(state_traj[:,i-1], net_dict, 0)
	end
	plot!(plt, t, rad2deg.(state_traj[1,:]), linewidth=3, legend=false, xlabel="Time (s.)", ylabel="Angle (deg.)")
	return plt
end

######################################################################

# Pendulum Examples ##
copies = 150 # copies = 0 is original network
model = "models/Pendulum/NN_params_pendulum_0_1s_1e7data_a15_12_2_L1.mat"

weights, net_dict = pendulum_net(model, copies)
Aᵢ, bᵢ = input_constraints_pendulum(weights, "box", net_dict=net_dict)
Aₒ, bₒ = output_constraints_pendulum(weights, "origin", net_dict=net_dict)

@time begin
state2input, state2output, state2map, state2backward = forward_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], reach=false, back=false, verification=false)
end
@show length(state2input)

# Plot all regions #
plt_in1  = plot_hrep_pendulum(state2input, net_dict, space="input")
# plt_in2  = plot_hrep_pendulum(state2backward, net_dict, space="input")
# plt_out = plot_hrep_pendulum(state2output, net_dict, space="output")


# Overlay samples on forward reach plot #
# n = 1000
# box = deg2rad(90)
# in_dat = [[bound_r(-box,box), bound_r(-box,box)] for _ in 1:n]
# out_dat = [eval_net(pnt, net_dict, copies) for pnt in in_dat]
# in_dat = hcat(in_dat...)'
# out_dat = hcat(out_dat...)'
# scatter!(plt_in1, rad2deg.(in_dat[:,1]), rad2deg.(in_dat[:,2]), legend=false)
# scatter!(plt_out, rad2deg.(out_dat[:,1]), rad2deg.(out_dat[:,2]), legend=false)


## Generate damped sine plot ##
# init = [deg2rad(30), deg2rad(0)]
# plt_sin = damped_plt(init, 50, net_dict)
