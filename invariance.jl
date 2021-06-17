using LinearAlgebra, MatrixEquations, COSMO

# Return true if x ∈ {x | Ax ≤ b}, otherwise return false
function in_polytope(x, A, b)
	for i in 1:length(b)
		A[i,:]⋅x > b[i] ? (return false) : nothing
	end
	return true
end

# Return trajectory of discrete time system where each column is the state at step i
# Julia arrays are stored in column-major order so it's faster to do matrix[:,i] = data rather than matrix[i,:] = data
function compute_traj(init, steps::Int64, weights, net_dict; dt=0.1)
	t = collect(0:dt:dt*steps)
	state_traj = Matrix{Float64}(undef, net_dict["input_size"], length(t))
	state_traj[:,1] = init
	for i in 2:length(t)
		state_traj[:,i] = eval_net(state_traj[:,i-1], weights, net_dict, 1)
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
function find_fixed_points(state2map, state2input, net_dict)
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
			p = inv(I - C̄) * d̄
			if in_polytope(p, Ā, b̄)
				push!(fixed_points, p)
				fp_dict[p] = [(Ā, b̄), (C̄, d̄)]
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
function intermediate_polytope(Q, Q̄, α, fp; max_constraints=10)
	dim = length(fp)
	A = Matrix{Float64}(undef, 1, dim)
	A[1,:] = [bound_r(-1,1) for _ in 1:dim]

	P_inv = inv(Q̄/α)
	x_opt, opt_val = lin_quad_opt(A[1,:], P_inv, fp)
	b = [opt_val]
	for k in 2:max_constraints
		# if k > dim + 1 && check_P_in_E(A, b, Q, α, fp)
		# 	return A, b
		# end
		A = vcat(A, reshape([bound_r(-1,1) for _ in 1:dim], 1,2))
		x_opt, opt_val = lin_quad_opt(A[k,:], P_inv, fp)
		b = vcat(b, [opt_val])
	end
	println("Incorrect Polytope!")
	return A, b
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
	
	@variable(model, Q_diag[1:n_cons])
	Q = Matrix(Diagonal(Q_diag))
	@variable(model, R[1:n_cons, 1:n_cons], PSD)

	@objective(model, Min, -sum(Q_diag) + sum(R))

	@constraint(model, Q_diag .≥ zeros(n_cons))
	@constraint(model, Q*Fₒ*C - R'*Fₒ .== zeros(n_cons, dim))
	@SDconstraint(model, [Q R; R' Q] ≥ 0)
	
	optimize!(model)
	if termination_status(model) == MOI.OPTIMAL
		return Diagonal(value.(Q_diag)), value.(R)
	else
		@show termination_status(model)
		error("Suboptimal SDP!")
	end
end

using Convex, SCS
# Solve polytope ROA SDP using CVX
function solve_sdp_cvx(Fₒ, C)
	n_cons, dim = size(Fₒ)
	
	Q_diag = Variable(n_cons, Positive())
	R = Semidefinite(n_cons)

	problem = minimize(-tr(diagm(Q_diag)) + sum(R))

	problem.constraints += diagm(Q_diag)*Fₒ*C - R'*Fₒ == zeros(n_cons, dim)
	problem.constraints += ([diagm(Q_diag) R; R' diagm(Q_diag)] in :SDP)
	
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
	model = Model(COSMO.Optimizer)
	@variable(model, λ[1:n_cons+n_consₓ])
	@objective(model, Max, sum(λ[1:n_cons]))
	@constraint(model, (Pₒ' - I)*λ .≤ zeros(n_cons+n_consₓ))
	@constraint(model, λ[n_cons+1:end] .≤ ones(n_consₓ))
	@constraint(model, λ[1:n_cons] .≤ 1e2*ones(n_cons)) # extra
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

	@show Fₒ

	# Λ ∈ R^(n_cons, n_cons), Diagonal
	# P ∈ R^(n_cons, n_cons), PSD
	# Pₒ = inv(Λ)*P*Λ
	# Q ∈ R^(n_cons, n_cons), Positive, Diagonal
	# R = PₒQ, PSD
	# Q ∈ R^(n_cons, n_cons), Diagonal
	# H = ΛQΛ ∈ R^(n_cons, n_cons), Positive, Diagonal
	# Pₒ_opt = R_opt*inv(Q_opt)

	# solve SDP
	Q_jmp, R_jmp = solve_sdp_jump(Fₒ, C)
	Q_cvx, R_cvx = solve_sdp_cvx(Fₒ, C)
	println("\n\n")
	@show norm(Q_jmp - Q_cvx)
	@show norm(R_jmp - R_cvx)
	println("\n\n")

	# with CVX variables
	println("CVX Variables:")
	Q_cvx = Matrix(Q_cvx)
	Pₒ_cvx = R_cvx*inv(Q_cvx)
	@show eigen(Pₒ_cvx).values
	@show eigen([Q_cvx R_cvx; R_cvx' Q_cvx]).values

	# with JuMP variables
	println("JuMP Variables:")
	Pₒ_jmp = R_jmp*inv(Q_jmp)
	@show eigen(Pₒ_jmp).values
	@show eigen([Q_jmp R_jmp; R_jmp' Q_jmp]).values
	

	# λ = solve_lp(Pₒ_cvx, n_cons, n_consₓ)
	# @show (Pₒ_cvx' - I)*λ
	# return λ
end




###########################################################################################


function solve_sdp_cvx_sym(Fₒ, C)
	n_cons, dim = size(Fₒ)
	
	Q_diag = Variable(n_cons, Positive())
	R1 = Semidefinite(n_cons)
	R2 = Variable(n_cons, n_cons)

	problem = minimize(-tr(diagm(Q_diag)) + sum(R1))

	problem.constraints += diagm(Q_diag)*Fₒ*C - R2'*Fₒ == zeros(n_cons, dim)
	problem.constraints += (R1' - R2' in :SDP)
	problem.constraints += (R1' + R2' in :SDP)
	problem.constraints += ([diagm(Q_diag) R1; R1' diagm(Q_diag)] in :SDP)
	
	solve!(problem, () -> SCS.Optimizer())
	if problem.status == MOI.OPTIMAL
		return Diagonal(evaluate(Q_diag)), evaluate(R1), evaluate(R2)
	else
		@show problem.status
		error("Suboptimal SDP!")
	end
end

# Solve polytope ROA LP using JuMP
function solve_lp_sym(Pₒ, n_cons, n_consₓ)
	model = Model(GLPK.Optimizer)
	@variable(model, λ[1:n_cons+n_consₓ])
	@objective(model, Max, sum(λ[1:n_cons]))
	@constraint(model, (Pₒ' - I)*λ .≤ zeros(n_cons+n_consₓ))
	@constraint(model, λ[n_cons+1:end] .≤ ones(n_consₓ))
	optimize!(model)
	if termination_status(model) == MOI.OPTIMAL
		return value.(λ)
	else
		@show termination_status(model)
		error("Suboptimal LP!")
	end
end

# Compute polytopic ROA #
function invariant_polytope_sym(Aₓ, bₓ, Aₒ, bₒ, C)
	n_consₓ, dim = size(Aₓ)
	n_cons = length(bₒ)

	Fₓ = vcat([reshape(Aₓ[i,:] ./ bₓ[i], (1, :)) for i in 1:length(bₓ)]...) 
	Fₒ = vcat([reshape(Aₒ[i,:] ./ bₒ[i], (1, :)) for i in 1:length(bₒ)]...) 
	Fₒ = vcat(Fₒ, Fₓ)

	# solve SDP
	# Q_jmp, R_jmp = solve_sdp_jump_sym(Fₒ, C)
	Q_cvx, R1_cvx, R2_cvx = solve_sdp_cvx_sym(Fₒ, C)
	# println("\n\n")
	# @show norm(Q_jmp - Q_cvx)
	# @show norm(R_jmp - R_cvx)
	# println("\n\n")

	# with CVX variables
	println("CVX Variables:")
	Pₒ_cvx = R1_cvx*inv(Q_cvx)

	# with JuMP variables
	# println("JuMP Variables:")
	# Pₒ_jmp = R_jmp*inv(Q_jmp)
	# @show eigen(Pₒ_jmp).values
	# @show eigen([Q_jmp R_jmp; R_jmp' Q_jmp]).values
	
	@show n_cons
	@show n_consₓ
	λ = solve_lp_sym(Pₒ_cvx, n_cons, n_consₓ)
	return λ
end














































## OLD ##
# Find an i-step invariant set given a fixed point p
function i_step_invariance(fixed_point, max_steps)
	for i in 0:max_steps
		println("\n", i+1)
		# load neural network
		copies = i # copies = 0 is original network
		model = "models/Pendulum/NN_params_pendulum_0_1s_1e7data_a15_12_2_L1.mat"
		weights, net_dict = pendulum_net(model, copies)
		state = get_state(fixed_point, weights)
		num_neurons = sum([length(state[layer]) for layer in 1:length(state)])
		
		# get Cx+d, Ax≤b
		C, d = local_map(state, weights)
		println("opnorm(C'C): ", opnorm(C'C))
		F = eigen(C)
		println("eigen(C): ", F.values)
		A, b, nothing, nothing, unique_nonzerow_indices = get_constraints(weights, state, num_neurons)
		A, b, nothing, nothing, nothing = remove_redundant(A, b, [], [], unique_nonzerow_indices, [])

		# get forward reachable set
		Af, bf = affine_map(A, b, C, d)

		# check if forward reachable set is a strict subset of Ax≤b
		unique_nonzerow_indices = 1:(length(b) + length(bf)) # all indices unique & nonzero
		A_, b_, nothing, nothing, nothing = remove_redundant(vcat(A, Af), vcat(b, bf), [], [], unique_nonzerow_indices, [])
		if A_ == Af && b_ == bf
			println(i+1, "-step invariant set found!")
			return i, state, A_, b_
		end
	end
	return nothing, nothing, nothing, nothing
end


