using NPZ

# van der Pol Oscillator #
# dynamics from http://control.asu.edu/Publications/2018/Colbert_CDC_2018.pdf


# RK_f(S; γ=1., ϵ=-1., ω=1.) = [S[2], (-γ - ϵ*S[1]^2)*S[2] + ω^2*S[1]] # forward time

RK_f(S) = [-S[2], S[1] + S[2]*(S[1]^2 - 1)] # reverse time. ROA is a nonconvex subset of a square of +- 3 around the origin


function RK_update(S, dt)
	k1 = RK_f(S)
	k2 = RK_f(S + dt*0.5*k1)
	k3 = RK_f(S + dt*0.5*k2)
	k4 = RK_f(S + dt*k3)
	return S + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
end

bound_r(a,b) = (b-a)*(rand()-1) + b # Generates a uniformly random number on [a,b]

# generates data where each X[i,:] is an input and each corresponding Y[i,:] is an output
function gen_data(n)
	dt = 0.1
	X = hcat([[bound_r(-2.5, 2.5), bound_r(-3., 3.)] for i in 1:n]...)'
	Y = hcat([RK_update(X[i,:], dt) for i in 1:n]...)'
	npzwrite("models/vanderpol/X.npy", X)
	npzwrite("models/vanderpol/Y.npy", Y)
	return nothing
end

function gen_traj(xₒ, T)
	traj = Matrix{Float64}(undef, 2, T+1)
	traj[:,1] = xₒ
	for t in 1:T
		traj[:,t+1] = RK_update(traj[:,t], 0.1)
	end
	return traj
end


# pyplot()
# copies = 1 # copies = 1 is original network
# weights, net_dict = vanderpol_net(copies)


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



# params = npzread("models/vanderpol/norm_params.npz")
# X_mean = vec(params["X_mean"])
# X_std = vec(params["X_std"])
# Y_mean = vec(params["Y_mean"])
# Y_std = vec(params["Y_std"])

# println("\n X_mean: ", X_mean)
# println("X_std: ", X_std)
# println("Y_mean: ", Y_mean)
# println("Y_std: ", Y_std)

# input = [0.5, 0.3]
# println("Validation Input: ", input)
# evl = eval_net(input, weights, net_dict, 1, type="normal")

# evlpy = (input - X_mean) ./ X_std
# println("Normalized Input: ", evlpy)

# # evlpy = weights[3]*max.(0, weights[2]*max.(0, weights[1]*vcat(evlpy, [1.])))
# println("Normalized Output: ", evlpy)
# evlpy = (Y_std .* evlpy) + Y_mean#  + input
# println("Unnormalized Output: ", evlpy)


# @show norm(evl - evlpy)
# @show evlpy - input

# Y_net = (Y - Y_mean) / Y_std

