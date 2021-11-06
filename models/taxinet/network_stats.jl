# File to evaluate accuracy of full_mlp_supervised.nnet
# locations of nnet.jl and full_mlp_supervised.nnet are specific to my computer

using Plots
include("../../nnet.jl")
include("../../load_networks.jl")

```Generates a uniformly random number on "["a,b"]"```
bound_r(a,b) = (b-a)*(rand()-1) + b

# Load state estimation network
# nn = NNet("models/taxinet/full_mlp_supervised.nnet") # x -> x_est
# n = 1000
# errors = Matrix{Float64}(undef, 2, n)

# for i in 1:n
# 	x = [bound_r(-11,11), bound_r(-30,30)]
# 	zx = [bound_r(-0.9,0.9); bound_r(-0.9,0.9); x]
# 	# zx = [0; 0; x]
# 	errors[:,i] = evaluate_network(nn, zx) - x
# end

# # Plot closed-loop error.
# plt1 = plot(reuse=false, legend=false, title="State Estimation Error", xlabel="x₁: Crosstrack Position (m.)", ylabel="x₂: Heading (deg.)")
# scatter!(plt1, errors[1,:], errors[2,:])

# Update control 1Hz
# Update dynamics 20Hz

function dynamics(x, u; dt=0.05)
	v, L= 5, 5
	x′ = [x[1] + v*sind(x[2])*dt, x[2] + rad2deg((v/L)*tand(u))*dt, x[3] + v*dt*cosd(x[2])] # changed this.
	return x′
end

function dynamics_learned(x, u; dt=0.05)
	v, L= 5, 5
	x′ = [eval_net(x[1:2], W_dyn, 1); x[3] + v*dt*cosd(x[2])] # changed this.
	return x′
end

W_gen_est = nnet_load("models/taxinet/full_mlp_supervised_2input.nnet") # x -> x_est
W_gen_4in = nnet_load("models/taxinet/full_mlp_supervised.nnet") # [z; x] -> x_est
W_dyn =  pytorch_net("models/taxinet/weights_dynamics.npz", "models/taxinet/norm_params_dynamics.npz", 1)

# latent variable that gives best tracking to centerline
z = [-1.8940158446577924, 0.9738946920139069]

function step(x; learned=false)
	x_est = eval_net(x[1:2], W_gen_est, 1)
	u = [-0.74, -0.44]⋅x_est
	learned ? (dynamics_learned(x, u)) : (return dynamics([x_est; x[3]], u))
	
end

weights = taxinet_cl()
m = 500

# Plot trajectory
plt2 = plot(reuse=false, legend=false, xlabel="Downtrack Position (m.)", ylabel="x₁: Crosstrack Position (m.)")
for ii = -9:9
	xₒ = [ii, 0.0, 0.0]
	traj = Matrix{Float64}(undef, 3, m) # [crosstrack position, heading error, downtrack position]
	traj[:,1] = xₒ
	for i in 2:m
		traj[:,i] = step(traj[:,i-1], learned=false)
	end
	plot!(plt2, traj[3,:], traj[1,:], xlims=[0, 120], ylims=[-10, 10])
end

plt2