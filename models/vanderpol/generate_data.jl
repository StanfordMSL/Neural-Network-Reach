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
	X = [bound_r(-5, 5) for i in 1:n, j in 1:2]
	Y = hcat([RK_update(X[i,:], dt) for i in 1:n]...)'
	npzwrite("models/vanderpol/X.npy", X)
	npzwrite("models/vanderpol/Y.npy", Y)
	return nothing
end

function gen_traj(T)
	xₒ = [bound_r(-1., 1.), bound_r(-1., 1.)]
	traj = Matrix{Float64}(undef, T+1, 2)
	traj[1,:] = xₒ
	for t in 1:T
		traj[t+1,:] = RK_update(traj[t,:], 0.1)
	end
	return traj
end

plt = plot()
for _ in 1:50
	traj = gen_traj(100)
	plot!(plt, traj[:,1], traj[:,2], label=false)
end
plt


plt_nn = plot()
for _ in 1:2
	xₒ = [bound_r(-0.1, 0.1), bound_r(-0.1, 0.1)]
	traj = compute_traj(xₒ, 100, weights, net_dict)
	plot!(plt_nn, traj[1, :], traj[2, :], label=false)
end
plt_nn