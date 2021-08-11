using NPZ, LinearAlgebra, Convex, MathOptInterface, SCS, ECOS

# MPC for unstable LTI system #
# dynamics from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.483.6896&rep=rep1&type=pdf


# One step lookahead
function update(S, N)
	A = [1.2 1.2; 0.0 1.2]
	B = [1.0, 0.4]
	Qₓ, Qᵤ = [1. 0.; 0. 1.], ones(1,1)

	# solve QP for u
	u = Variable(1, N)
	x = Variable(2, N+1)
	problem = minimize(sum(quadform(x[:,i+1], Qₓ'*Qₓ) + quadform(u[:,i], Qᵤ'*Qᵤ) for i in 1:N))
	# State and Control limits
	for i in 1:N
		problem.constraints += norm(x[:,i], Inf) ≤ 5
		problem.constraints += norm(u[:,i], Inf) ≤ 1
	end
	problem.constraints += norm(x[:,N+1], Inf) ≤ 5

	# Dynamic Feasibility
	problem.constraints += x[:,1] == S
	for i in 2:N+1
		problem.constraints += x[:,i] == A*x[:,i-1] + B*u[:,i-1]
	end

	solve!(problem, ECOS.Optimizer; silent_solver=true)

	if problem.status == MathOptInterface.OPTIMAL
		if N == 1
			return [evaluate(u)]
		else
			return evaluate(u[:,1])
		end
	else
		return [100.] # kind of a hack
	end

	
end


bound_r(a,b) = (b-a)*(rand()-1) + b # Generates a uniformly random number on [a,b]

# generates data where each X[i,:] is an input and each corresponding Y[i,:] is an output
# n is num_samples; N is lookahead horizon for MPC
function gen_data(n, N)
	X = Matrix{Float64}(undef, n, 2)
	Y = Matrix{Float64}(undef, n, 1)

	for i in 1:n
		while true
			x = [bound_r(-5.0, 5.0), bound_r(-5.0, 5.0)]
			y = update(x, N)
			if y != [100.] # hacky
				X[i,:] = x
				Y[i,:] = y
				break
			end
		end
	end

	npzwrite(string("models/mpc/X", N, ".npy"), X)
	npzwrite(string("models/mpc/Y", N, ".npy"), Y)
	return nothing
end

function gen_traj(xₒ, T)
	traj = Matrix{Float64}(undef, 2, T+1)
	traj[:,1] = xₒ
	for t in 1:T
		traj[:,t+1] = update(traj[:,t])
	end
	return traj
end


# pyplot()
# copies = 1 # copies = 1 is original network
# weights, net_dict = pytorch_net("mpc", copies)


# plt = plot(title="True", reuse=false)
# plt_nn = plot(title="Neural Network", reuse=false)
# for i in 1:20
# 	n = 200
# 	xₒ = [bound_r(-1., 1.), bound_r(-1., 1.)]
# 	traj = gen_traj(xₒ, n)
# 	traj_nn = compute_traj(xₒ, n, weights, net_dict, type="normal")
	
# 	scatter!(plt, [xₒ[1]], [xₒ[2]], label=false, reuse=false)
# 	scatter!(plt_nn, [xₒ[1]], [xₒ[2]], label=false, reuse=false)
# 	plot!(plt, traj[1,:], traj[2,:], label=false, reuse=false)
# 	plot!(plt_nn, traj_nn[1,:], traj_nn[2,:], label=false, reuse=false)
# end


