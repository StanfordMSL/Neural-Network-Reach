using Plots, FileIO, JLD2
include("reach.jl")
include("invariance.jl")



### VAN DER POL LIMIT CYCLE ###
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
	


### CONSTRAINTS ###
function input_constraints_vanderpol(type::String)
	# Each input specification is in the form Ax≤b
	if type == "box"
		in_dim = 2
		A_pos = Matrix{Float64}(I, in_dim, in_dim)
		A_neg = Matrix{Float64}(-I, in_dim, in_dim)
		A = vcat(A_pos, A_neg)
		b = [2.5, 3.0, 2.5, 3.0]
	elseif type == "large_box"
		in_dim = 2
		A_pos = Matrix{Float64}(I, in_dim, in_dim)
		A_neg = Matrix{Float64}(-I, in_dim, in_dim)
		A = vcat(A_pos, A_neg)
		b = [7.0, 7.0, 7.0, 7.0]
	elseif type == "hexagon"
		A = [1 0; -1 0; 0 1; 0 -1; 1 1; -1 1; 1 -1; -1 -1]
		b = [5, 5, 5, 5, 8, 8, 8, 8]
	else
		error("Invalid input constraint specification.")
	end
	return A, b
end


function output_constraints_vanderpol(type::String)
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
function plot_hrep_vanderpol(ap2constraints; type="normal")
	# Plot all polytopes
	plt = plot(reuse = false, legend=false)
	for ap in keys(ap2constraints)
		A, b = ap2constraints[ap]
		reg = HPolytope(constraints_list(A, b))
	
		# sanity check
		if isempty(reg)
			@show reg
			error("Empty polyhedron.")
		end

		# static or gif plot
		if type == "normal"
			plot!(plt,  reg, xlabel="x₁", ylabel="x₂", xlims=(-2.5, 2.5), ylims=(-3, 3), fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
		elseif type == "gif"
			plot!(plt,  reg, xlabel="x₁", ylabel="x₂", xlims=(-3, 3), ylims=(-3, 3), fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
		end
	end
	return plt
end


# make gif of backwards reachable set over time
function BRS_gif(nn_weights, nn_params, Aᵢ, bᵢ, Aₛ, bₛ, steps)
	plt = plot(HPolytope(constraints_list(Aₛ, bₛ)), xlims=(-2.5, 2.5), ylims=(-3, 3))
	anim = @animate for Step in 2:steps
		weights = pytorch_net(nn_weights, nn_params, Step)
		ap2input, ap2output, ap2map, ap2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₛ], [bₛ], back=true)
    	plt = plot_hrep_vanderpol(ap2backward[1], type="gif")
	end
	gif(anim, string("vanderpol_brs_",steps  ,".gif"), fps = 2)
end


# make figure of BRSs from paper
function vanderpol_fig(steps, brs_dict)
	plots = []
	for step in steps
		plt = plot_hrep_vanderpol(brs_dict[string(step)])
		plt = add_limit_cycle(plt)
		push!(plots, plt)
	end
	subplots = plot(plots..., layout=(3, 2), xlims=(-7, 7), ylims=(-7, 7), size=(4*3*96, 3*4*4*96/3))
	return subplots
end




### REGION OF ATTRACTION ###
function get_BRSs(steps; connected=true)
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
		ap2input, ap2output, ap2map, ap2backward = compute_reach(weights, Aᵢ, bᵢ, [A_roa], [b_roa], fp=fp, back=true, connected=connected, check_aps=false)
		end
		@show length(ap2input)
		@show length(ap2backward[1])
		push!(ap2constraints_vec, ap2backward[1])
	end
	return ap2constraints_vec
end










### SCRIPTING ###

# Make Figure in Paper
# brs_dict = load("models/vanderpol/BRSs.jld2")
# steps = [5, 10, 15, 20, 25, 30]
# subplots = vanderpol_fig(steps, brs_dict)


# or compute one of the ROAs
# find explicit PWA representation
# load network weights
nn_weights = "models/vanderpol/weights.npz"
nn_params = "models/vanderpol/norm_params.npz"
weights = pytorch_net(nn_weights, nn_params, 1)

# set domain, output constraints
Aᵢ, bᵢ = input_constraints_vanderpol("box")
Aₒ, bₒ = output_constraints_vanderpol("origin")

# solve for explicit PWA representation
@time begin
ap2input, ap2output, ap2map, ap2backward, ap2neighbors = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], graph=true)
end
@show length(ap2input)


# plot input space
plt_in  = plot_hrep_vanderpol(ap2input)


# check homeomorphic property
@time begin
print("NN is homeomorphic: ", is_homeomorphism(ap2map, 2))
end

# load seed ROA used in paper
A_roa = Matrix{Float64}(matread("models/vanderpol/vanderpol_seed.mat")["A_roa"])
b_roa = Vector{Float64}(matread("models/vanderpol/vanderpol_seed.mat")["b_roa"])

# or find new seed ROA
# @time begin
# fixed_points, fp_dict = find_fixed_points(ap2map, ap2input, weights)
# fp, C, d, Aₓ, bₓ = find_attractor(fixed_points, fp_dict)
# end
# @time begin
# A_roa, b_roa = polytope_roa_sdp(Aₓ, bₓ, C, fp, 25)
# end


# plot seed ROA
plt_seed = plot(HPolyhedron(constraints_list(A_roa, b_roa)))


# solve for t-step ROAs via backward reachability
steps = [10]
ap2constraints_vec = get_BRSs(steps, connected=true)
plt_brs = add_limit_cycle(plot_hrep_vanderpol(ap2constraints_vec[1]))

# save BRSs
# save("models/vanderpol/BRSs.jld2", Dict("5" => ap2constraints_vec[1], "10" => ap2constraints_vec[2],
# 										"15" => ap2constraints_vec[3], "20" => ap2constraints_vec[4],
# 										"25" => ap2constraints_vec[5], "30" => ap2constraints_vec[6]))

# save pwa_dict
# save("models/vanderpol/vanderpol_pwa_large.jld2", Dict("Ai" => Aᵢ, "bi" => bᵢ,
# 	"ap2map" => ap2map, "ap2input" => ap2input, "ap2neighbors" => ap2neighbors,
# 	"ap_fp" => ap_fp, "seed_roa" => (A_roa, b_roa)))