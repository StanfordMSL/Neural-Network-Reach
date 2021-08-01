using Plots
include("reach.jl")
include("invariance.jl")

# Returns H-rep of various input sets
function input_constraints_vanderpol(weights, type::String, net_dict)
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

	Aᵢₙ, bᵢₙ = net_dict["input_norm_map"]
	return A*inv(Aᵢₙ), b + A*inv(Aᵢₙ)*bᵢₙ
end

# Returns H-rep of various output sets
function output_constraints_vanderpol(weights, type::String, net_dict)
	# Each output specification is in the form Ayₒᵤₜ≤b
	# The raw network outputs are unnormalized: yₒᵤₜ = Aₒᵤₜy + bₒᵤₜ
	# Thus the output constraints for raw network outputs are: A*Aₒᵤₜ*y ≤ b - A*bₒᵤₜ
	if type == "origin"
		A = [1. 0.; -1. 0.; 0. 1.; 0. -1.]
		b = [1., 1., 1., 1.]
 	else 
 		error("Invalid input constraint specification.")
 	end
 	Aₒᵤₜ, bₒᵤₜ = net_dict["output_unnorm_map"]
 	return A*Aₒᵤₜ, b - A*bₒᵤₜ
end


# Plot all polytopes
function plot_hrep_vanderpol(state2constraints, net_dict; space = "input", type="normal")
	plt = plot(reuse = false, legend=false, xlabel="x₁", ylabel="x₂")
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
		
		if isempty(reg)
			@show reg
			error("Empty polyhedron.")
		end

		if type == "normal"
			plot!(plt,  reg, xlims=(-2.5, 2.5), ylims=(-3, 3), fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
		elseif type == "gif"
			plot!(plt,  reg, xlims=(-3, 3), ylims=(-3, 3), fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
		end
	end
	return plt
end


# make gif of backwards reachable set over time
function BRS_gif(model, Aᵢ, bᵢ, Aₛ, bₛ, steps)
	plt = plot(HPolytope(constraints_list(Aₛ, bₛ)), xlims=(-2.5, 2.5), ylims=(-3, 3))
	anim = @animate for Step in 2:steps
		weights, net_dict = pytorch_net("vanderpol", Step)
		Aₒᵤₜ, bₒᵤₜ = net_dict["output_unnorm_map"]
		Aₒ, bₒ = Aₛ*Aₒᵤₜ, bₛ - Aₛ*bₒᵤₜ
		state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], reach=false, back=true, verification=false)
    	plt = plot_hrep_pendulum(state2backward[1], net_dict, space="input", type="gif")
	end
	gif(anim, string("vanderpol_brs_",steps  ,".gif"), fps = 2)
end


###########################
######## SCRIPTING ########
###########################
# save("models/vanderpol/vanderpol_pwa.jld2", Dict("state2input" => state2input, "state2map" => state2map, "net_dict" => net_dict, "Ai" => Aᵢ, "bi" => bᵢ))
# matwrite("models/vanderpol/vanderpol_seed.mat", Dict("A_roa" => A_roa*A_norm_in, "b_roa" => b_roa- A_roa*b_norm_in))

# Given a network representing discrete-time autonomous dynamics and state constraints,
# ⋅ find fixed points
# ⋅ verify the fixed points are stable equilibria
# ⋅ compute invariant polytopes around the fixed points
# ⋅ perform backwards reachability to estimate the maximal region of attraction in the domain

copies = 2 # copies = 1 is original network
weights, net_dict = pytorch_net("vanderpol", copies)


Aᵢ, bᵢ = input_constraints_vanderpol(weights, "box", net_dict)
Aₒ, bₒ = output_constraints_vanderpol(weights, "origin", net_dict)
A_roa = Matrix{Float64}(matread("models/vanderpol/vanderpol_seed.mat")["A_roa"])
b_roa = Vector{Float64}(matread("models/vanderpol/vanderpol_seed.mat")["b_roa"])

# Run algorithm
@time begin
state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [A_roa], [b_roa], fp=fp, reach=false, back=true, connected=true)
end
@show length(state2input)
@show length(state2backward[1])


# Plot all regions #
# plt_in1  = plot_hrep_vanderpol(state2input, net_dict, space="input")
plt_in2  = plot_hrep_vanderpol(state2backward[1], net_dict, space="input")

# homeomorph = is_homeomorphism(state2map, size(Aᵢ,2))
# println("PWA function is a homeomorphism: ", homeomorph)

# fixed_points, fp_dict = find_fixed_points(state2map, state2input, weights, net_dict)
# fp = fixed_points[1]
# @show fp


# Getting mostly suboptimal SDP here
# A_roa, b_roa, state2backward_chain, net_dict_chain, plt_in2 = find_roa("vanderpol", 20, 3)
# 10 steps is ~35k polytopes with ~300 polytopes in the BRS
# 15 steps is 88,500 polytopes with 895 polytopes in the BRS
# algorithm does ~1000 polytopes per minute.
# Create gif of backward reachable set
# BRS_gif(model, Aᵢ, bᵢ, A_roa, b_roa, 5)
# nothing
















# old stuff #
# Find Polytopic ROA via Sylvester method # 
# A_roa, b_roa = polytope_roa_slyvester(Aₓ, bₓ, C, fp; constraints=10)

# Sample P s.t. P .≥ 0 && P*1 ≤ 1
# constraints = 10
# λ_c, vecs_c = eigen(C)
# P = sample_P(λ_c, constraints)

# Solve Sylvester equation PF + F(-C) = 0
# F = sylvc(P, -C, zeros(constraints, size(C,2)))


# Check local stability and find local Lyapunov function #
# Q = local_stability(fp, fp_dict)

# Find max ellipsoidal ROA in polytope #
# α = max_ellipsoid(Q, fp, fp_dict)
# α = 3.0 # scaling. α ↑ ⟹ volume ↑
# plot!(plt_in1, (180/π)*Ellipsoid(fp, α*inv(Q)), check_posdef=false)

# Check that it is actually a ROA #
# xₒ = deg2rad.([0., 25.])
# state_traj = compute_traj(xₒ, 100, weights, net_dict)
# scatter!(plt_in1, rad2deg.(state_traj[1,:]), rad2deg.(state_traj[2,:]))

# Find one step reachable set from max ellipsoidal ROA in polytpe #
# Q̄ = forward_reach_ellipse(Q, fp, fp_dict)
# plot!(plt_in1, (180/π)*Ellipsoid(fp, α*inv(Q̄), check_posdef=false))


# i, state, A_, b_ = i_step_invariance(fixed_points[1], 1)
# plt_seed = plot_hrep_pendulum(Dict(state => (A_, b_)), net_dict; space = "input")






























# # Export to matlab data
# A_dat = Vector{Matrix{Float64}}(undef, length(state2input))
# b_dat = Vector{Vector{Float64}}(undef, length(state2input))
# C_dat = Vector{Matrix{Float64}}(undef, length(state2input))
# d_dat = Vector{Vector{Float64}}(undef, length(state2input))

# for (i,key) in enumerate(keys(state2input))
# 	C_dat[i], d_dat[i] = state2map[key]
# 	A_dat[i], b_dat[i] = state2input[key]
# end


# A = mxcellarray(A_dat)  # creates a MATLAB cell array
# b = mxcellarray(b_dat)  # creates a MATLAB cell array
# C = mxcellarray(C_dat)  # creates a MATLAB cell array
# d = mxcellarray(d_dat)  # creates a MATLAB cell array
# write_matfile("region_dat.mat"; A = A, b = b, C = C, d = d)



# mat"""
# Q = Polyhedron([1 -1;0.5 -2;-1 0.4; -1 -2],[1;2;3;4])

# sys1 = LTISystem('A', 0.5, 'f', 0);
# sys1.setDomain('x', Polyhedron('lb',0, 'ub', 1));

# sys2 = LTISystem('A', -0.5, 'f', 0);
# sys2.setDomain('x', Polyhedron('lb',-1, 'ub', 0));

# bb = 33


# pwa = PWASystem([sys1, sys2])
# S = pwa.invariantSet()
# size(S)
# $ii

# """












# for region in regions, search each one for locally invariant set
# for ap in keys(state2map)
# 	C, d = state2map[ap]
# 	A, b = state2input[ap]
# 	println(rank(I - C))
# 	p = (I - C) \ d
# 	inside = true
# 	for i in 1:length(b)
# 		A[i,:]⋅p > b[i] ? inside = false : nothing
# 	end
# 	if inside
# 		@show p
# 		scatter!(plt_in1, [p[1]], [p[2]])
# 	end
# end


# fixed_point = [-0.001989079271247683, 0.0174667758051234] # in the normalized input space

# Then compose network 50 times and find all inputs that lead to the fixed point
# for region in regions, search each one for locally invariant set
# for ap in keys(state2map)
# 	feasible = false
# 	C, d = state2map[ap]
# 	A, b = state2input[ap]
# 	rank(C) > 2 ? println(rank(C)) : nothing

# 	# solve feasibility LP
# 	model = Model(GLPK.Optimizer)
# 	@variable(model, x[1:size(C,2)])
# 	@objective(model, Max, 0)
# 	@constraint(model, A*x .≤ b)
# 	@constraint(model, C*x + d .== fixed_point)
# 	optimize!(model)
# 	if termination_status(model) == MOI.OPTIMAL
# 		A_ = vcat(A, C, -C)
# 		b_ = vcat(b, d, -d)
# 		state2invariant[ap] = (A_, b_) # feasible
# 	end
# end



# plt_in3  = plot_hrep_pendulum(state2invariant, net_dict, space="input")






















# mat"""
# Q = Polyhedron([1 -1;0.5 -2;-1 0.4; -1 -2],[1;2;3;4])

# sys1 = LTISystem('A', 1, 'f', 1);
# sys1.setDomain('x', Polyhedron('lb',0));

# sys2 = LTISystem('A', -2, 'f', 1);
# sys2.setDomain('x', Polyhedron('ub',0));

# pwa = PWASystem([sys1,sys2]);
# S = pwa.invariantSet();
# """
