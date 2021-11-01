using NPZ

# torque-controlled damped pendulum #
# dynamics from http://control.asu.edu/Publications/2018/Colbert_CDC_2018.pdf


RK_f(x, u; m=1., l=1., b=1.) = [x[2], (u - b*x[2] - m*9.81*l*sin(x[1]))/(m*l^2)]


function RK_update(x, u, dt)
	k1 = RK_f(x, u)
	k2 = RK_f(x + dt*0.5*k1, u)
	k3 = RK_f(x + dt*0.5*k2, u)
	k4 = RK_f(x + dt*k3, u)
	return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
end

bound_r(a,b) = (b-a)*(rand()-1) + b # Generates a uniformly random number on [a,b]

# generates data where each X[i,:] is an input and each corresponding Y[i,:] is an output
function gen_data(n)
	dt = 0.1
	X = hcat([[bound_r(-2*π/3, 2*π/3), bound_r(-π, π), bound_r(-2,2)] for i in 1:n]...)'
	Y = hcat([RK_update(X[i,1:2], X[i,3], dt) for i in 1:n]...)'
	npzwrite("models/Pendulum/X_controlled.npy", X)
	npzwrite("models/Pendulum/Y_controlled.npy", Y)
	return nothing
end

function gen_traj(xₒ, T)
	traj = Matrix{Float64}(undef, 2, T+1)
	traj[:,1] = xₒ
	for t in 1:T
		traj[:,t+1] = RK_update(traj[:,t], 0.,  0.1)
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


