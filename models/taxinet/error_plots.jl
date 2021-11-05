using NPZ, Plots
include("../../load_networks.jl")

```
from https://arxiv.org/pdf/2105.07091.pdf Eq. 8
x = "["p, θ"]" where p in meters and θ in degrees
```
function dynamics(x; dt=0.05)
	v, L= 5, 5
	u = [-0.74, -0.44]⋅x
	x′ = [x[1] + v*sind(x[2]*dt), x[2] + rad2deg((v/L)*tand(u))*dt]
	return x′
end


# Load closed-loop and state estimation networks
W_cl = taxinet_cl() # x -> x′
W_est = nnet_load("models/taxinet/TinyTaxiNet.nnet") # image -> x_est
W_dyn =  pytorch_net("models/taxinet/weights_dynamics.npz", "models/taxinet/norm_params_dynamics.npz", 1) # x_est -> x′
W_gen_est = nnet_load("models/taxinet/full_mlp_supervised_2input.nnet") # x -> x_est
W_gen_4in = nnet_load("models/taxinet/full_mlp_supervised.nnet") # [x; z] -> x_est

# load (image,state) data set
images = npzread("models/taxinet/X_image.npy")
states = npzread("models/taxinet/Y_image.npy")
n = size(states,1)

println("Computing approximation errors for ", n, " data points.")
errors = Matrix{Float64}(undef, 2, n) # closed-loop errors
errors_dyn = Matrix{Float64}(undef, 2, n) # dynamics errors
errors_gen_est = Matrix{Float64}(undef, 2, n) # generator + state estimation errors
errors_connect = Matrix{Float64}(undef, 2, n) # concactenated network errors
errors_2in = Matrix{Float64}(undef, 2, n) # deleted inputs errors
errors_true_dyn = Matrix{Float64}(undef, 2, n) # using true dynamics

for i in 1:n
	# Compute true next state
	image, state = images[i,:], states[i,:]
	x_est = eval_net(image, W_est, 1)
	x′_true = dynamics(x_est)

	# Compute approximate next state
	x′_approx = eval_net(state, W_cl, 1)

	# Store error
	errors[:,i] = x′_approx - x′_true
	errors_dyn[:,i] = eval_net(state, W_dyn, 1) - dynamics(state)
	errors_gen_est[:,i] = eval_net(state, W_gen_est, 1) - state
	errors_connect[:,i] = eval_net(eval_net(state, W_gen_est, 1), W_dyn, 1)  - x′_approx
	errors_2in[:,i] = eval_net(state, W_gen_est, 1) - eval_net([0; 0; state], W_gen_4in, 1)

	# true dynamics
	errors_true_dyn[:,i] = dynamics(eval_net(state, W_gen_est, 1)) - x′_true
end

# Plot closed-loop error.
plt1 = plot(reuse=false, legend=false, title="CL Error", xlabel="x₁: Crosstrack Position (m.)", ylabel="x₂: Heading (deg.)")
scatter!(plt1, errors[1,:], errors[2,:])

# Plot dynamics error. Negligible error.
plt2 = plot(reuse=false, legend=false, title="Dynamics Error", xlabel="x₁: Crosstrack Position (m.)", ylabel="x₂: Heading (deg.)")
scatter!(plt2, errors_dyn[1,:], errors_dyn[2,:])

# Plot generator + estimation error. Considerable error.
plt3 = plot(reuse=false, legend=false, title="Generator + Estimator Error", xlabel="x₁: Crosstrack Position (m.)", ylabel="x₂: Heading (deg.)")
scatter!(plt3, errors_gen_est[1,:], errors_gen_est[2,:])

# Plot network concactenation error. Negligible error.
plt4 = plot(reuse=false, legend=false, title="Network Concactenation Error", xlabel="x₁: Crosstrack Position (m.)", ylabel="x₂: Heading (deg.)")
scatter!(plt4, errors_connect[1,:], errors_connect[2,:])

# Plot deleted inputs error. No error.
plt5 = plot(reuse=false, legend=false, title="Deleted Inputs Error", xlabel="x₁: Crosstrack Position (m.)", ylabel="x₂: Heading (deg.)")
scatter!(plt5, errors_2in[1,:], errors_2in[2,:])

# Plot deleted inputs error. No error.
plt6 = plot(reuse=false, legend=false, title="True Dynamics Error", xlabel="x₁: Crosstrack Position (m.)", ylabel="x₂: Heading (deg.)")
scatter!(plt6, errors_true_dyn[1,:], errors_true_dyn[2,:])



# Takeaway: the full_mlp file is not a good state estimator.