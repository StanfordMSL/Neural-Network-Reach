using LinearAlgebra, MatrixEquations

# Return true if x ∈ {x | Ax ≤ b}, otherwise return false
function in_polytope(x, A, b)
	for i in 1:length(b)
		A[i,:]⋅x > b[i] ? (return false) : nothing
	end
	return true
end

# Return trajectory of discrete time system where each column is the state at step i
# Julia arrays are stored in column-major order so it's faster to do matrix[:,i] = data rather than matrix[i,:] = data
function compute_traj(init, steps::Int64, net_dict; dt=0.1)
	t = collect(0:dt:dt*steps)
	state_traj = Matrix{Float64}(undef, net_dict["input_size"], length(t))
	state_traj[:,1] = init
	for i in 2:length(t)
		state_traj[:,i] = eval_net(state_traj[:,i-1], weights, net_dict, 0)
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

# Find local quadratic Lyapunov function of the form: x'Qx + r'x + constant
function local_stability(p, fp_dict)
	region = fp_dict[p]
	A, b = region[1]
	C, d = region[2]
	dim = length(d)
	F = eigen(C)
	if all(norm.(F.values) .< 1) # then stable
		# Transform affine sytem to linear system:
		# xₜ₊₁ = Cxₜ + d,  x_ₜ = xₜ + inv(C)*d  ⟹ x_ₜ₊₁ = Cx_ₜ
		# local Lyapunov function: x_'*Q*x_ = x'*Q*x + 2*d'*inv(C)*Q + constant
		Q = lyapd(C, Matrix{Float64}(I, dim, dim)) # 2nd arg is symmetric. Where does it come from?
		r = 2*d'*inv(C)*Q
	else
		error("Unstable local system.")
	end
	return Q, r
end






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


