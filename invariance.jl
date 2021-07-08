using LinearAlgebra, MatrixEquations, JuMP, Convex, SCS, COSMO, GLPK, Distributions

# Return true if x ∈ {x | Ax ≤ b}, otherwise return false
function in_polytope(x, A, b)
	for i in 1:length(b)
		A[i,:]⋅x > b[i] ? (return false) : nothing
	end
	return true
end

# Return trajectory of discrete time system where each column is the state at step i
# Julia arrays are stored in column-major order so it's faster to do matrix[:,i] = data rather than matrix[i,:] = data
function compute_traj(init, steps::Int64, weights, net_dict; type="normal")
	state_traj = Matrix{Float64}(undef, net_dict["input_size"], steps+1)
	state_traj[:,1] = init
	for i in 1:steps
		state_traj[:,i+1] = eval_net(state_traj[:,i], weights, net_dict, 1, type=type)
	end
	return state_traj
end

# Specific for the pendulum
function plot_traj(state_traj)
	plt = plot(reuse = false)
	plot!(plt, t, rad2deg.(state_traj[1,:]), linewidth=3, legend=false, xlabel="Time (s.)", ylabel="Angle (deg.)", fontfamily=font(14, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
	return plt
end

# Iterate through each cell cᵢ in a PWA function and find fixed points if they exist.
# The raw neural network has affine functions: ỹ = Cx̃ + d
# The networks inputs and outputs are normalized/unnormalized as: x̃ = Aᵢₙx + bᵢₙ, y = Aₒᵤₜỹ + bₒᵤₜ
# Thus, y = Aₒᵤₜ(C(Aᵢₙx + bᵢₙ) + d) + bₒᵤₜ = Aₒᵤₜ(CAᵢₙx + Cbᵢₙ + d) + bₒᵤₜ = AₒᵤₜCAᵢₙx + AₒᵤₜCbᵢₙ + Aₒᵤₜd + bₒᵤₜ
# ⟹ C̄ = AₒᵤₜCAᵢₙ, d̄ = AₒᵤₜCbᵢₙ + Aₒᵤₜd + bₒᵤₜ
# Fixed point p satisfies: C̄p + d̄ = p ⟹ (I - C̄)p = d̄.
# p must lie in the polytope for which the affine map is valid
# The raw neural network has polytopes Ax̄ ≤ b  ⟹  A(Aᵢₙx + bᵢₙ) ≤ b  ⟹  AAᵢₙx ≤ b - Abᵢₙ  ⟹  Ā = AAᵢₙ, b̄ = b - Abᵢₙ
function find_fixed_points(state2map, state2input, weights, net_dict)
	Aᵢₙ, bᵢₙ = net_dict["input_norm_map"]
	Aₒᵤₜ, bₒᵤₜ = net_dict["output_unnorm_map"]
	dim = net_dict["input_size"]
	fixed_points = Vector{Vector{Float64}}(undef, 0)
	fp_dict = Dict{ Vector{Float64}, Vector{Tuple{Matrix{Float64},Vector{Float64}}} }() # stores [(Ā, b̄), (C̄, d̄)]
	for ap in keys(state2map)
		C, d = state2map[ap]
		A, b = state2input[ap]
		C̄ = Aₒᵤₜ*C*Aᵢₙ
		d̄ = Aₒᵤₜ*C*bᵢₙ + Aₒᵤₜ*d + bₒᵤₜ
		Ā = A*Aᵢₙ
		b̄ = b - A*bᵢₙ

		if rank(I - C̄) == dim
			fp = inv(I - C̄) * d̄
			if in_polytope(fp, Ā, b̄) && (eval_net(fp, weights, net_dict, 1) ≈ fp) # second condition is sanity check
				push!(fixed_points, fp)
				fp_dict[fp] = [(Ā, b̄), (C̄, d̄)]
				println("Found fixed point.")
			end
		else
			error("Non-unique fixed point! Make more general")
		end
	end
	return fixed_points, fp_dict
end

# Find local quadratic Lyapunov function of the form: (x - c)'Q⁻¹(x - c)
function local_stability(fp, fp_dict)
	region = fp_dict[fp]
	C, d = region[2]
	dim = length(d)
	F = eigen(C)
	if all(norm.(F.values) .< 1) # then stable
		# Transform affine sytem to linear system:
		# xₜ₊₁ = Cxₜ + d,  x_ₜ = xₜ - p  ⟹ x_ₜ₊₁ = Cx_ₜ
		# local Lyapunov function: x_'*Q*x_ =  (x - p)'*Q*(x - p)
		Q = lyapd(C', Matrix{Float64}(I, dim, dim)) # 2nd arg is symmetric. Where does it come from?
	else
		error("Unstable local system.")
	end
	return Q
end


# Compute Q̄ such that (x₊ - p)'*Q̄*(x₊ - p) ≤ 1 = {x₊ | (x - p)'*Q*(x - p) ≤ 1, x₊ = C*x + d}
function forward_reach_ellipse(Q, fp, fp_dict)
	region = fp_dict[fp]
	C, d = region[2]
	C_inv = inv(C)
	return C_inv'*Q*C_inv
end

# Solve the optimization problem:
# maximize a'x
# subject to (x - fp)'P(x - fp) ≤ 1
function lin_quad_opt(a, P_inv, fp)
	x_opt = P_inv*a ./ sqrt(a⋅(P_inv*a)) + fp
	opt_val = a⋅x_opt
	return x_opt, opt_val
end

# Checks whether a given polyhedron is bounded (i.e. is a polyotpe)
# Can be made faster by solving Cheby LP
function is_polytope(A, b)
	return isbounded(HPolyhedron(constraints_list(A, b)))
end

# Checks whether a given polytope is a subset of a given ellipsoid
function check_P_in_E(A, b, Q, α, fp)
	is_polytope(A, b) ? nothing : (return false)
	P_inv = inv(Q/α)
	for i in 1:length(b)
		x_opt, opt_val = lin_quad_opt(A[i,:], P_inv, fp)
		opt_val < b[i] ? (return false) : nothing
	end
	return true
end

# find polytope P s.t. E_innner ⊂ P ⊂ E_outer
# E_inner = {x | (x - x_f)'(1\α)*Q̄*(x - x_f) ≤ 1}
# E_outer = {x | (x - x_f)'(1\α)*Q*(x - x_f) ≤ 1}
function intermediate_polytope(Q̄, α, fp, Aₓ, bₓ, C; max_constraints=100)
	dim = length(fp)
	# get constraints for linear system (x̄ frame)
	A = deepcopy(Aₓ)
	b = deepcopy(bₓ) - A*fp

	P_inv = inv(Q̄/α)
	for k in 1:max_constraints
		A = vcat(A, reshape([bound_r(-1,1) for _ in 1:dim], 1,2))
		x_opt, opt_val = lin_quad_opt(A[end,:], P_inv, zeros(dim)) # want constraints in the x̄ frame
		b = vcat(b, [opt_val])
	end
	
	return A, b + A*fp # return constraints in the x frame
end

# generate random polytope that contains the fixed point
# we have that x̄ = x - fp
function random_polytope(fp, num_constraints)
	F = [bound_r(-1, 1) for i in 1:num_constraints, j in 1:length(fp)]
	b = ones(num_constraints)
	
	return F, b + F*fp # return constraints in the x frame
end

# Plot convergence over time of some points to a fixed point
function convergence(fp, state2backward, weights, net_dict, traj_length)
	plt = plot(title="Distance to Fixed Point vs Time-Step", xlabel="Time-Step", ylabel="Distance")
	for state in keys(state2backward)
		Aᵢ, bᵢ = state2backward[state]
		Aᵢₙ, bᵢₙ = net_dict["input_norm_map"]
		Ā = Aᵢ*Aᵢₙ
		b̄ = bᵢ - Aᵢ*bᵢₙ
		xₒ, nothing, nothing = cheby_lp([], [], Ā, b̄, [])
		state_traj = compute_traj(xₒ, traj_length, weights, net_dict)
		distances = [norm(state_traj[:,k] - fp) for k in 1:traj_length+1]
		plot!(plt, 1:traj_length+1, distances, label=false)
	end
	return plt
end

# Solve polytope ROA SDP using JuMP
function solve_sdp_jump(Fₒ, C)
	n_cons, dim = size(Fₒ)
	model = Model(COSMO.Optimizer)
	set_optimizer_attribute(model, "max_iter", 4000)
	set_optimizer_attribute(model, "verbose", false)
	
	@variable(model, Q_diag[1:n_cons])
	Q = Matrix(Diagonal(Q_diag))
	@variable(model, R[1:n_cons, 1:n_cons])

	# @objective(model, Min, -sum(Q_diag) + sum(R))
	@objective(model, Min, 0)
	
	@constraint(model, Q*Fₒ*C - R'*Fₒ .== zeros(n_cons, dim))
	@SDconstraint(model, [Q R; R' Q] ≥ 0)
	@constraint(model, Q_diag .≥ 10.0*ones(n_cons))
	# @constraint(model, Q_diag .≤ 1e6*ones(n_cons))
	@constraint(model, R .≥ zeros(n_cons, n_cons))
	
	optimize!(model)
	if termination_status(model) == MOI.OPTIMAL
		return Diagonal(value.(Q_diag)), value.(R)
	else
		@show termination_status(model)
		error("Suboptimal SDP!")
	end
end

# Solve polytope ROA SDP using CVX
function solve_sdp_cvx(Fₒ, C)
	n_cons, dim = size(Fₒ)
	
	Q_diag = Variable(n_cons, Positive())
	R = Variable(n_cons, n_cons, Positive())

	# problem = minimize(-sum(Q_diag) + sum(R))
	problem = minimize(0)

	problem.constraints += diagm(Q_diag)*Fₒ*C - R'*Fₒ == zeros(n_cons, dim)
	problem.constraints += ([diagm(Q_diag) R; R' diagm(Q_diag)] in :SDP)
	problem.constraints += Q_diag ≥ ones(n_cons)
	
	solve!(problem, () -> SCS.Optimizer())
	if problem.status == MOI.OPTIMAL
		return Diagonal(evaluate(Q_diag)), evaluate(R)
	else
		@show problem.status
		error("Suboptimal SDP!")
	end
end

# Solve polytope ROA LP using JuMP
function solve_lp(Pₒ, n_cons, n_consₓ)
	model = Model(GLPK.Optimizer)
	@variable(model, λ[1:n_cons+n_consₓ])
	@objective(model, Max, sum(λ[1:n_cons]))
	@constraint(model, (Pₒ' - I)*λ .≤ zeros(n_cons+n_consₓ))
	@constraint(model, λ[n_cons+1:end] .≤ ones(n_consₓ))
	# @constraint(model, λ[1:n_cons] .≤ 1e2*ones(n_cons)) # extra
	@constraint(model, λ .≥ 1e-6*ones(n_cons+n_consₓ)) # extra
	optimize!(model)
	if termination_status(model) == MOI.OPTIMAL
		return value.(λ)
	else
		@show termination_status(model)
		error("Suboptimal LP!")
	end
end


# Compute polytopic ROA #
function invariant_polytope(Aₓ, bₓ, Aₒ, bₒ, C)
	n_consₓ, dim = size(Aₓ)
	n_cons = length(bₒ)

	Fₓ = vcat([reshape(Aₓ[i,:] ./ bₓ[i], (1, :)) for i in 1:length(bₓ)]...) 
	Fₒ = vcat([reshape(Aₒ[i,:] ./ bₒ[i], (1, :)) for i in 1:length(bₒ)]...) 
	Fₒ = vcat(Fₒ, Fₓ)

	# solve SDP
	Q_jmp, R_jmp = solve_sdp_jump(Fₒ, C)
	Pₒ_jmp = R_jmp*inv(Q_jmp)
	λ = solve_lp(Pₒ_jmp, n_cons, n_consₓ)
	F_res = inv(Diagonal(λ))*Fₒ

	# Verify that found polytope is invariant
	if is_invariant(F_res, C)
		println("Found invariant polytope!.")
		return F_res, ones(size(F_res,1))
	else
		println("Polytope not invariant.")
		return F_res, ones(size(F_res,1))
	end
end

# check if polytope Fx≤1 is invariant under dynamics xₜ₊₁ = Cxₜ
# The polytope is invariant iff ∃ P ∈ PSD s.t.
# FA = PF  &&  P1 ≤ 1
function is_invariant(F, C)
	n_cons, dim = size(F)
	model = Model(GLPK.Optimizer)

	@variable(model, P[1:n_cons, 1:n_cons])
	@objective(model, Min, 0) # feasibility

	@constraint(model, P.≥ zeros(n_cons, n_cons))
	@constraint(model, P*F .== F*C )
	@constraint(model, P*ones(n_cons).≤ ones(n_cons))
	
	optimize!(model)
	if termination_status(model) == MOI.OPTIMAL
		return true
	else
		@show termination_status(model)
		return false
	end
end


# Attempt to find a polytopic ROA via the SDP method
 function polytope_roa_sdp(Aₓ, bₓ, C, fp, num_constraints)
 	# Generate initial polytope
 	Aₛ, bₛ = random_polytope(fp, num_constraints)

 	# Put constraints in x̄ frame
	Aₓ_ = Aₓ
	bₓ_ = bₓ - Aₓ_*fp
	Aₛ_ = Aₛ
	bₛ_ = bₛ - Aₛ_*fp

	A_roa_, b_roa_ = invariant_polytope(Aₓ_, bₓ_, Aₛ_, bₛ_, C)
	return  A_roa_, b_roa_ + A_roa_*fp # return constraints in the x frame
 end

# Given fixed points and fp_dict find one that is a local attractor
function find_attractor(fixed_points, fp_dict)
	for (i, fp) in enumerate(fixed_points)
		region = fp_dict[fp]
		Aₓ, bₓ = region[1]
		C, d = region[2]
		if all(norm.(eigen(C).values) .< 1)
			println("Verified fixed point is a local attractor.")
			return fp, C, d, Aₓ, bₓ
		elseif i == length(fixed_points)
			error("Unstable local system")
		end
	end
	return fp, C, d, Aₓ, bₓ
end



# Given a NN model, number of seed polytope ROA constraints, and number of backward reachability steps, compute ROA
function find_roa(dynamics::String, num_constraints, num_steps)
	if dynamics == "pendulum"
		weights, net_dict = pendulum_net("models/Pendulum/NN_params_pendulum_0_1s_1e7data_a15_12_2_L1.mat", 1)
		Aᵢ, bᵢ = input_constraints_pendulum(weights, "pendulum", net_dict)
		Aₒ, bₒ = output_constraints_pendulum(weights, "origin", net_dict)
		println("Input set: pendulum")
	elseif dynamics == "vanderpol"
		weights, net_dict = vanderpol_net(1)
		Aᵢ, bᵢ = input_constraints_vanderpol(weights, "box", net_dict)
		Aₒ, bₒ = output_constraints_vanderpol(weights, "origin", net_dict)
		println("Input set: van der Pol box")
	end

	
	# Run algorithm on one-step dynamics
	@time begin
	state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], reach=false, back=false, verification=false)
	end
	println("Dynamics function has ", length(state2input), " affine regions.") 

	# Find fixed point(s) #
	fixed_points, fp_dict = find_fixed_points(state2map, state2input, weights, net_dict)
	fp, C, d, Aₓ, bₓ = find_attractor(fixed_points, fp_dict)


	# Find Polytopic ROA via SDP method # 
	A_roa, b_roa = polytope_roa_sdp(Aₓ, bₓ, C, fp, num_constraints)

	# Perform backward reachability on the seed invariant set
	if dynamics == "pendulum"
		weights_chain, net_dict_chain = pendulum_net(model, num_steps)
	elseif dynamics == "vanderpol"
		weights_chain, net_dict_chain = vanderpol_net(num_steps)
	end
	Aₒᵤₜ, bₒᵤₜ = net_dict["output_unnorm_map"]
	Aₒ_chain, bₒ_chain = A_roa*Aₒᵤₜ, b_roa - A_roa*bₒᵤₜ
	state2input_chain, state2output_chain, state2map_chain, state2backward_chain = compute_reach(weights_chain, Aᵢ, bᵢ, [Aₒ_chain], [bₒ_chain], reach=false, back=true, verification=false)
	
	if dynamics == "pendulum"
		plt_in2  = plot_hrep_pendulum(state2backward_chain[1], net_dict_chain, space="input")
		plot!(plt_in2, title=string(num_steps, "-Step BRS"), xlims=(-90, 90), ylims=(-90, 90))
	elseif dynamics == "vanderpol"
		plt_in2  = plot_hrep_vanderpol(state2backward_chain[1], net_dict_chain, space="input")
		plot!(plt_in2, title=string(num_steps, "-Step BRS"), xlims=(-3, 3), ylims=(-3, 3))
	end

	return A_roa, b_roa, state2backward_chain[1], net_dict_chain, plt_in2
end

















## OLD ##

# # Sample P matrix s.t. P .≥ 0 && P*1 ≤ 1
# function sample_P(λ_c, constraints)
# 	# Each row is rand() * a randomly sampled vector from the unit simplex
# 	P = Matrix{Float64}(undef, constraints, constraints)
# 	for row in 1:constraints
# 		P[row, :] = rand()*rand(Dirichlet(ones(constraints)))
# 	end

# 	# Sylvester equation has unique solution when P and C have no common eigenvalues
# 	vals, vecs = eigen(P)

# 	# Could be dangerous infinite loop
# 	if !isempty(intersect(vals, λ_c))
# 		println("Eigenvalues intersect, generating new P matrix.")
# 		return sample_P(λ_c)
# 	else
# 		return P
# 	end

# end

# # Attempt to find a polytopic ROA via the SDP method
#  function polytope_roa_sylvester(Aₓ, bₓ, C, fp; constraints=100)
#  	# Put domain constraints in x̄ frame
# 	Aₓ_ = Aₓ
# 	bₓ_ = bₓ - Aₓ_*fp
# 	Aₛ_ = Aₛ
# 	bₛ_ = bₛ - Aₛ_*fp

# 	# Sample P s.t. P .≥ 0 && P*1 ≤ 1
# 	λ_c, vecs_c = eigen(C)
# 	P = sample_P(λ_c, constraints)	

# 	# Solve Sylvester equation PF + F(-C) = 0
# 	F = sylvc(P, -C, zeros(constraints, size(C,2)))

# 	# Grow/shrink polytope roa to fit within domain polytope


# 	return  A_roa_, b_roa_ + A_roa_*fp # return constraints in the x frame
#  end




# # Find an i-step invariant set given a fixed point p
# function i_step_invariance(fixed_point, max_steps)
# 	for i in 0:max_steps
# 		println("\n", i+1)
# 		# load neural network
# 		copies = i # copies = 0 is original network
# 		model = "models/Pendulum/NN_params_pendulum_0_1s_1e7data_a15_12_2_L1.mat"
# 		weights, net_dict = pendulum_net(model, copies)
# 		state = get_state(fixed_point, weights)
# 		num_neurons = sum([length(state[layer]) for layer in 1:length(state)])
		
# 		# get Cx+d, Ax≤b
# 		C, d = local_map(state, weights)
# 		println("opnorm(C'C): ", opnorm(C'C))
# 		F = eigen(C)
# 		println("eigen(C): ", F.values)
# 		A, b, nothing, nothing, unique_nonzerow_indices = get_constraints(weights, state, num_neurons)
# 		A, b, nothing, nothing, nothing = remove_redundant(A, b, [], [], unique_nonzerow_indices, [])

# 		# get forward reachable set
# 		Af, bf = affine_map(A, b, C, d)

# 		# check if forward reachable set is a strict subset of Ax≤b
# 		unique_nonzerow_indices = 1:(length(b) + length(bf)) # all indices unique & nonzero
# 		A_, b_, nothing, nothing, nothing = remove_redundant(vcat(A, Af), vcat(b, bf), [], [], unique_nonzerow_indices, [])
# 		if A_ == Af && b_ == bf
# 			println(i+1, "-step invariant set found!")
# 			return i, state, A_, b_
# 		end
# 	end
# 	return nothing, nothing, nothing, nothing
# end


