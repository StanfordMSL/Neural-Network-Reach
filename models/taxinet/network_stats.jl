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
	x′ = [x[1] + v*sind(x[2])*dt, x[2] + rad2deg((v/L)*tand(u))*dt, x[3] + v*dt*cosd(x[2])]
	return x′
end

function dynamics_learned(x, u; dt=0.05)
	v, L= 5, 5
	x′ = [eval_net([u; x[1:2]], W_dyn, 1); x[3] + v*dt*cosd(x[2])] # changed this.
	return x′
end

W_gen_est = nnet_load("models/taxinet/full_mlp_supervised_2input_0.nnet") # x -> x_est
W_gen_4in = nnet_load("models/taxinet/full_mlp_supervised.nnet") # [z; x] -> x_est
W_dyn =  pytorch_net("models/taxinet/weights_dynamics.npz", "models/taxinet/norm_params_dynamics.npz", 1) # [u; x] -> x′
W_ux = taxinet_2input_resid() # x -> [u; x]
W_cl = taxinet_cl(1)

# latent variable that gives best tracking to centerline
# z = [0, 0]

function step(x; learned=false)
	x_est = eval_net(x[1:2], W_gen_est, 1)
	# x_est = eval_net([z; x[1:2]], W_gen_4in, 1)
	u = [-0.74, -0.44]⋅x_est
	learned ? (return dynamics_learned(x, u)) : (return dynamics(x, u))
end

m = 1500

# Plot true trajectories
# plt2 = plot(reuse=false, legend=false, xlabel="Downtrack Position (m.)", ylabel="x₁: Crosstrack Position (m.)")
# for ii = -9:9
# 	xₒ = [ii, 0.0, 0.0]
# 	traj = Matrix{Float64}(undef, 3, m) # [crosstrack position, heading error, downtrack position]
# 	traj[:,1] = xₒ
# 	for i in 2:m
# 		traj[:,i] = step(traj[:,i-1], learned=true)
# 	end
# 	plot!(plt2, traj[3,:], traj[1,:], xlims=[0, 120], ylims=[-10, 10])
# end


# Plot closed-loop NN trajectories
# plt3 = plot(reuse=false, legend=false, xlabel="Downtrack Position (m.)", ylabel="x₁: Crosstrack Position (m.)")
plt3 = plot(reuse=false, legend=false, size=(692,195), xaxis=false)
plot!(plt3, Shape([(130,10),(130,-10),(0,-10),(0,10)]), color="gray68")
plot!(plt3, Shape([(130,20),(130,10),(0,10),(0,20)]), color="darkseagreen2")
plot!(plt3, Shape([(130,-10),(130,-20),(0,-20),(0,-10)]), color="darkseagreen2")
hline!(plt3, [0; 120], [0.0], color="white", line=:dash)
hline!(plt3, [0; 120], [10.0, -10.0], color="black", linewidth=2, xlims=[0, 120])
for ii = -8:2:8
	xₒ = [ii, 0.0, 0.0]
	traj_nn = Matrix{Float64}(undef, 3, m) # [crosstrack position, heading error, downtrack position]
	traj_nn[:,1] = xₒ
	for i in 2:m
		x′ = eval_net(traj_nn[1:2, i-1], W_cl, 1)
		d′ = traj_nn[3, i-1] + 5*0.05*cosd(traj_nn[2, i-1])
		traj_nn[:,i] = [x′; d′]
	end
	# @show traj_nn[:,end]
	plot!(plt3, traj_nn[3,:], traj_nn[1,:], xlims=[0, 120], ylims=[-11, 11], color="blue")
end



plt3

