# File to evaluate accuracy of full_mlp_supervised.nnet
# locations of nnet.jl and full_mlp_supervised.nnet are specific to my computer

using Plots
include("../../nnet.jl")

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


weights = taxinet_cl()
m = 200
xₒ = [0.5, -1]
traj = Matrix{Float64}(undef, 2, m)
for i in 1:m
	traj[:,i] = eval_net(xₒ, weights, i)
end

# Plot trajectory
plt2 = plot(reuse=false, legend=false, title="Trajectory", xlabel="Step", ylabel="x₁: Crosstrack Position (m.)")
scatter!(plt2, 1:m, traj[1,:])