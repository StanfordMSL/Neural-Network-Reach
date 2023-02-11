using Plots, FileIO, JLD2
include("reach.jl")
include("invariance.jl")

# Returns H-rep of various input sets
function input_constraints_vanderpol(type::String)
	# Each input specification is in the form Ax≤b
	# The network takes normalized inputs: xₙ = Aᵢₙx + bᵢₙ
	# Thus the input constraints for raw network inputs are: A*inv(Aᵢₙ)x ≤ b + A*inv(Aᵢₙ)*bᵢₙ
	if type == "box"
		in_dim = 2
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
function output_constraints_vanderpol(type::String)
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
function plot_hrep_vanderpol(ap2constraints; type="normal")
	plt = plot(reuse = false, legend=false)
	for ap in keys(ap2constraints)
		A, b = ap2constraints[ap]
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
		ap2input, ap2output, ap2map, ap2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₛ], [bₛ], reach=false, back=true, verification=false)
    	plt = plot_hrep_pendulum(ap2backward[1], type="gif")
	end
	gif(anim, string("vanderpol_brs_",steps  ,".gif"), fps = 2)
end




function get_BRSs(steps)
	ap2constraints_vec = []
	Aᵢ, bᵢ = input_constraints_vanderpol("box")
	A_roa = Matrix{Float64}(matread("models/vanderpol/vanderpol_seed.mat")["A_roa"])
	b_roa = Vector{Float64}(matread("models/vanderpol/vanderpol_seed.mat")["b_roa"])
	fp = matread("models/vanderpol/vanderpol_seed.mat")["fp"] 

	for copies in steps
		nn_weights = "models/vanderpol/weights.npz"
		nn_params = "models/vanderpol/norm_params.npz"
		weights = pytorch_net(nn_weights, nn_params, copies)

		# Run algorithm
		@time begin
		ap2input, ap2output, ap2map, ap2backward = compute_reach(weights, Aᵢ, bᵢ, [A_roa], [b_roa], fp=fp, reach=false, back=true, connected=true)
		end
		@show length(ap2input)
		@show length(ap2backward[1])
		push!(ap2constraints_vec, ap2backward[1])
	end
	return ap2constraints_vec
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




# pick # steps for BRSs
steps = [5, 10]


# Find BRSs
ap2constraints_vec = get_BRSs(steps)
plt_5 = add_limit_cycle(plot_hrep_vanderpol(ap2constraints_vec[1]))
plt_10 = add_limit_cycle(plot_hrep_vanderpol(ap2constraints_vec[2]))


# Make Figure in Paper
# brs_dict = load("models/vanderpol/BRSs.jld2")
# subplots = vanderpol_fig(steps, brs_dict)


