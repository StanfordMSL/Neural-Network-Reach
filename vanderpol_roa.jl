using Plots
include("reach.jl")
include("invariance.jl")

# Returns H-rep of various input sets
function input_constraints_vanderpol(weights, type::String)
	# Each input specification is in the form Ax≤b
	# The network takes normalized inputs: xₙ = Aᵢₙx + bᵢₙ
	# Thus the input constraints for raw network inputs are: A*inv(Aᵢₙ)x ≤ b + A*inv(Aᵢₙ)*bᵢₙ
	if type == "box"
		in_dim = size(weights[1],2) - 1
		A_pos = Matrix{Float64}(I, in_dim, in_dim)
		A_neg = Matrix{Float64}(-I, in_dim, in_dim)
		A = vcat(A_pos, A_neg)
		b = [2.5, 3.0, 2.5, 3.0]
	elseif type == "hexagon"
		A = [1 0; -1 0; 0 1; 0 -1; 1 1; -1 1; 1 -1; -1 -1]
		b = [5, 5, 5, 5, 8, 8, 8, 8]
	else
		error("Invalid input constraint specification.")
	end
	return A, b
end

# Returns H-rep of various output sets
function output_constraints_vanderpol(weights, type::String)
	# Each output specification is in the form Ayₒᵤₜ≤b
	# The raw network outputs are unnormalized: yₒᵤₜ = Aₒᵤₜy + bₒᵤₜ
	# Thus the output constraints for raw network outputs are: A*Aₒᵤₜ*y ≤ b - A*bₒᵤₜ
	if type == "origin"
		A = [1. 0.; -1. 0.; 0. 1.; 0. -1.]
		b = [1., 1., 1., 1.]
 	else 
 		error("Invalid input constraint specification.")
 	end
 	return A, b
end


# Plot all polytopes
function plot_hrep_vanderpol(state2constraints; type="normal")
	plt = plot(reuse = false, legend=false)
	for state in keys(state2constraints)
		A, b = state2constraints[state]
		reg = HPolytope(constraints_list(A, b))
	
		if isempty(reg)
			@show reg
			error("Empty polyhedron.")
		end

		if type == "normal"
			plot!(plt,  reg, xlabel="x₁", ylabel="x₂", xlims=(-2.5, 2.5), ylims=(-3, 3), fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
		elseif type == "gif"
			plot!(plt,  reg, xlabel="x₁", ylabel="x₂", xlims=(-3, 3), ylims=(-3, 3), fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
		end
	end
	return plt
end



# overplot van der Pol limit cycle
RK_f(S) = [-S[2], S[1] + S[2]*(S[1]^2 - 1)] # reverse time. ROA is a nonconvex subset of a square of +- 3 around the origin

function RK_update(S, dt)
	k1 = RK_f(S)
	k2 = RK_f(S + dt*0.5*k1)
	k3 = RK_f(S + dt*0.5*k2)
	k4 = RK_f(S + dt*k3)
	return S + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
end

function rollout(x0, steps, dt)
	traj = Matrix{Float64}(undef, steps, 2)
	traj[1,:] = x0
	for i in 2:steps
		traj[i,:] = RK_update(traj[i-1,:], dt)
	end
	return traj
end

function add_limit_cycle(plt)
	traj = rollout([-2.0086212, 0.0], 135, 0.05)
	plot!(plt, traj[:,1], traj[:,2], label=false, color="blue")
end


# make gif of backwards reachable set over time
function BRS_gif(nn_weights, nn_params, Aᵢ, bᵢ, Aₛ, bₛ, steps)
	plt = plot(HPolytope(constraints_list(Aₛ, bₛ)), xlims=(-2.5, 2.5), ylims=(-3, 3))
	anim = @animate for Step in 2:steps
		weights = pytorch_net(nn_weights, nn_params, Step)
		state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₛ], [bₛ], reach=false, back=true, verification=false)
    	plt = plot_hrep_pendulum(state2backward[1], type="gif")
	end
	gif(anim, string("vanderpol_brs_",steps  ,".gif"), fps = 2)
end




function get_BRSs(steps)
	state2constraints_vec = []
	Aᵢ, bᵢ = input_constraints_vanderpol(weights, "box")
	Aₒ, bₒ = output_constraints_vanderpol(weights, "origin")
	A_roa = Matrix{Float64}(matread("models/vanderpol/vanderpol_seed.mat")["A_roa"])
	b_roa = Vector{Float64}(matread("models/vanderpol/vanderpol_seed.mat")["b_roa"])
	fp = matread("models/vanderpol/vanderpol_seed.mat")["fp"] 

	for copies in steps
		nn_weights = "models/vanderpol/weights.npz"
		nn_params = "models/vanderpol/norm_params.npz"
		weights = pytorch_net(nn_weights, nn_params, copies)

		# Run algorithm
		@time begin
		state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [A_roa], [b_roa], fp=fp, reach=false, back=true, connected=true)
		end
		@show length(state2input)
		@show length(state2backward[1])
		push!(state2constraints_vec, state2backward[1])
	end
	return state2constraints_vec
end


function vanderpol_fig(steps, brs_dict)
	plots = []
	for step in steps
		plt = plot_hrep_vanderpol(brs_dict[string(step)])
		plt = add_limit_cycle(plt)
		push!(plots, plt)
	end
	subplots = plot(plots..., layout=(3, 2), xlims=(-2.5, 2.5), ylims=(-3, 3), size=(4*3*96, 3*4*4*96/3))
	return subplots
end


###########################
######## SCRIPTING ########
###########################
# save("models/vanderpol/vanderpol_pwa.jld2", Dict("state2input" => state2input, "state2map" => state2map, "Ai" => Aᵢ, "bi" => bᵢ))
# matwrite("models/vanderpol/vanderpol_seed.mat", Dict("A_roa" => A_roa, "b_roa" => b_roa, "fp" => fp))

# Given a network representing discrete-time autonomous dynamics and state constraints,
# ⋅ find fixed points
# ⋅ verify the fixed points are stable equilibria
# ⋅ compute invariant polytopes around the fixed points
# ⋅ perform backwards reachability to estimate the maximal region of attraction in the domain

# copies = 5 # copies = 1 is original network
# nn_weights = "models/vanderpol/weights.npz"
# nn_params = "models/vanderpol/norm_params.npz"
# weights = pytorch_net(nn_weights, nn_params, copies)


# Aᵢ, bᵢ = input_constraints_vanderpol(weights, "box")
# Aₒ, bₒ = output_constraints_vanderpol(weights, "origin")
# A_roa = Matrix{Float64}(matread("models/vanderpol/vanderpol_seed.mat")["A_roa"])
# b_roa = Vector{Float64}(matread("models/vanderpol/vanderpol_seed.mat")["b_roa"])
# fp = matread("models/vanderpol/vanderpol_seed.mat")["fp"]

# Run algorithm
# @time begin
# state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [A_roa], [b_roa], fp=fp, reach=false, back=true, connected=true)
# end
# @show length(state2input)
# @show length(state2backward[1])


# Plot all regions #
# plt_in1  = plot_hrep_vanderpol(state2input)
# plt_in2  = plot_hrep_vanderpol(state2backward[1])


# homeomorph = is_homeomorphism(state2map, size(Aᵢ,2))
# println("PWA function is a homeomorphism: ", homeomorph)

# fixed_points, fp_dict = find_fixed_points(state2map, state2input, weights)
# fp = fixed_points[1]
# @show fp


# Getting mostly suboptimal SDP here
# A_roa, b_roa, fp, state2backward_chain, plt_in2 = find_roa("vanderpol", nn_weights, 20, 7, nn_params=nn_params)
# @show plt_in2
# 10 steps is ~35k polytopes with ~300 polytopes in the BRS
# 15 steps is 88,500 polytopes with 895 polytopes in the BRS
# algorithm does ~1000 polytopes per minute.
# Create gif of backward reachable set
# BRS_gif(nn_weights, nn_params, Aᵢ, bᵢ, A_roa, b_roa, 5)
# nothing





# Make Figure in Paper
steps = [5, 10, 15, 20, 25, 30]

# state2constraints_vec = get_BRSs(steps)
# save("models/vanderpol/BRSs.jld2", Dict("5" => state2constraints_vec[1], "10" => state2constraints_vec[2], "15" => state2constraints_vec[3],
										# "20" => state2constraints_vec[4], "25" => state2constraints_vec[5], "30" => state2constraints_vec[6]))

brs_dict = load("models/vanderpol/BRSs.jld2")
subplots = vanderpol_fig([5, 10, 15, 20, 25, 30], brs_dict)

nothing

