using MAT, NPZ, LinearAlgebra
include("nnet.jl")


# Evaluate network. Takes care of normalizing inputs and un-normalizing outputs
function eval_net(input, weights, copies::Int64; type="normal")
	copies == 0 ? (return input) : nothing
	NN_out = vcat(input, [1.])
    for layer = 1:length(weights)-1
        NN_out = max.(0, weights[layer]*NN_out)
    end
    output = weights[end]*NN_out
    type == "residual" ? output += input : nothing
    return eval_net(output, weights, copies-1)
end


bound_r(a,b) = (b-a)*(rand()-1) + b # Generates a uniformly random number on [a,b]

# NN with input in, output out, hidden dim hdim, and hidden layers layers.
function random_net(in_d, out_d, hdim, layers)
	Weights = Vector{Array{Float64,2}}(undef,layers)
	Weights[1] = sqrt(2/515)*(2*rand(hdim, in_d) - rand(hdim, in_d))
	for i in 2:layers-1
		Weights[i] = sqrt(2/515)*(2*rand(hdim,hdim) - rand(hdim,hdim)) # Kaiming Initialization
	end
	Weights[end] = sqrt(2/515)*(2*rand(out_d, hdim) - rand(out_d, hdim))
	return Weights
end


# Load nnet network #
# filename = "ACASXU_experimental_v2a_1_1.nnet"
function nnet_load(filename)
	nnet = NNet(filename)
	weights = Vector{Array{Float64,2}}(undef, nnet.numLayers)
	for i in 1:(nnet.numLayers-1)
		weights[i] = vcat(hcat(nnet.weights[i], nnet.biases[i]), reshape(zeros(1+nnet.layerSizes[i]),1,:))
		weights[i][end,end] = 1
	end
	# last layer weight shouldn't carry forward the bias term. i.e. augmented but with last row removed
	weights[end] = hcat(nnet.weights[end], nnet.biases[end])

	# make net_dict
	σᵢ = Diagonal(nnet.ranges[1:end-1])
	μᵢ = nnet.means[1:end-1]
	σₒ = nnet.ranges[end]*Matrix{Float64}(I, nnet.inputSize, nnet.inputSize)
	μₒ = nnet.means[end]*ones(nnet.outputSize)

	net_dict = Dict()
	net_dict["num_layers"] = nnet.numLayers
	net_dict["layer_sizes"] = nnet.layerSizes
	net_dict["input_size"] = nnet.inputSize 
	net_dict["output_size"] = nnet.outputSize
	net_dict["input_norm_map"] = (inv(σᵢ), -inv(σᵢ)*μᵢ)
	net_dict["output_unnorm_map"] = (σₒ, μₒ) 

	return weights, nnet, net_dict
end


# Load ACAS Networks #
function acas_net_nnet(a::Int64, b::Int64)
	filename = string("models/ACAS_nnet/ACASXU_experimental_v2a_", a, "_", b, ".nnet")
	return nnet_load(filename)
end

# chain together multiple networks
function chain_net(w, copies, num_layers, layer_sizes)
	weights = Vector{Array{Float64,2}}(undef, copies*num_layers - (copies-1))
	merged_layers = [c*num_layers - (c-1) for c in 1:copies]
	w_idx = 1
	for k in 1:length(weights)
		if k == 1
			weights[k] = w[1]
			w_idx += 1
		elseif k == length(weights)
			weights[k] = w[end]
		elseif k in merged_layers
			w̄ₒ = vcat(w[end], reshape(zeros(1+layer_sizes[end-1]),1,:))
			w̄ₒ[end,end] = 1
			weights[k] = w[1]*w̄ₒ
			w_idx = 2
		else
			weights[k] = w[w_idx]
			w_idx += 1
		end
	end

	return weights
end


# Load Pendulum Networks #
function pendulum_net(filename::String, copies::Int64)
	model = matread(filename)
	num_layers = length(model["weights"])
	layer_sizes = vcat(size(model["weights"][1], 2), [length(vec(model["biases"][i])) for i in 1:num_layers])

	σᵢ = Float64.(Diagonal(vec(model["X_std"])))
	μᵢ = Float64.(vec(model["X_mean"]))
	σₒ = Float64.(Diagonal(vec(model["Y_std"])))
	μₒ = Float64.(vec(model["Y_mean"]))
	Aᵢₙ, bᵢₙ = inv(σᵢ), -inv(σᵢ)*μᵢ
	Aₒᵤₜ, bₒᵤₜ = σₒ, μₒ

	w = Vector{Array{Float64,2}}(undef, num_layers)
	weight = model["weights"][1]*Aᵢₙ
	bias   = vec(model["biases"][1]) + model["weights"][1]*bᵢₙ
	w[1]   = vcat(hcat(weight, vec(bias)), reshape(zeros(1+layer_sizes[1]),1,:))
	w[1][end,end] = 1
	for i in 2:(num_layers-1)
		weight = model["weights"][i]
		bias   = vec(model["biases"][i])
		w[i]   = vcat(hcat(weight, vec(bias)), reshape(zeros(1+layer_sizes[i]),1,:))
		w[i][end,end] = 1
	end
	weight = Aₒᵤₜ*model["weights"][end]
	bias   = Aₒᵤₜ*vec(model["biases"][end]) + bₒᵤₜ
	w[end] = hcat(weight, vec(bias))
	
	weights = chain_net(w, copies, num_layers, layer_sizes)

	return weights
end



# Load pytorch networks saved as numpy variables
function pytorch_net(model, copies::Int64)
	W = npzread(string("models/", model, "/weights.npz"))
	params = npzread(string("models/", model, "/norm_params.npz"))

	num_layers = Int(length(W)/2)
	layer_sizes = params["layer_sizes"]

	σᵢ = Float64.(Diagonal(vec(params["X_std"])))
	μᵢ = Float64.(vec(params["X_mean"]))
	σₒ = Float64.(Diagonal(vec(params["Y_std"])))
	μₒ = Float64.(vec(params["Y_mean"]))
	Aᵢₙ, bᵢₙ = inv(σᵢ), -inv(σᵢ)*μᵢ
	Aₒᵤₜ, bₒᵤₜ = σₒ, μₒ

	w = Vector{Array{Float64,2}}(undef, num_layers)
	weight = W[string("arr_", 0)]*Aᵢₙ
	bias   = W[string("arr_", 1)] + W[string("arr_", 0)]*bᵢₙ
	w[1] = vcat(hcat(weight, vec(bias)), reshape(zeros(1+layer_sizes[1]),1,:))
	w[1][end,end] = 1
	for i in 2:(num_layers-1)
		weight = W[string("arr_", 2*(i-1))]
		bias   = W[string("arr_", 2*(i-1)+1)]
		w[i] = vcat(hcat(weight, vec(bias)), reshape(zeros(1+layer_sizes[i]),1,:))
		w[i][end,end] = 1
	end

	weight = Aₒᵤₜ*W[string("arr_", 2*(num_layers-1))]
	bias   = Aₒᵤₜ*W[string("arr_", 2*(num_layers-1)+1)] + bₒᵤₜ
	w[end] = hcat(weight, vec(bias))

	weights = chain_net(w, copies, num_layers, layer_sizes)

	return weights
end



# Load pytorch networks that are controllers for linear MPC models
# I apply the dynamics A matrix in the first layer to try to avoid hyperplanes through the origin
function pytorch_mpc_net(model, copies::Int64)
	W = npzread(string("models/", model, "/weights.npz"))
	params = npzread(string("models/", model, "/norm_params.npz"))

	# x_+ = Ax + Bu
	A = [1.2 1.2; 0.0 1.2]
	B = reshape([1.0, 0.4], (2,1))

	num_layers = Int(length(W)/2)
	layer_sizes = params["layer_sizes"]

	σᵢ = Float64.(Diagonal(vec(params["X_std"])))
	μᵢ = Float64.(vec(params["X_mean"]))
	σₒ = Float64.(Diagonal(vec(params["Y_std"])))
	μₒ = Float64.(vec(params["Y_mean"]))
	Aᵢₙ, bᵢₙ = inv(σᵢ), -inv(σᵢ)*μᵢ
	Aₒᵤₜ, bₒᵤₜ = σₒ, μₒ

	# make identity weights
	sze = layer_sizes[1]
	II = Matrix{Float64}(I, sze, sze)
	# w_I1 = [A; -A]
	w_I1 = [II; -II]
	w_Im = [II -II; -II II]
	b_I = zeros(2*sze)

	# make single network augmented weights
	w = Vector{Array{Float64,2}}(undef, num_layers)
	weight = vcat(W[string("arr_", 0)]*Aᵢₙ, w_I1)
	bias   = vcat(W[string("arr_", 1)] + W[string("arr_", 0)]*bᵢₙ, b_I)
	w[1] = vcat(hcat(weight, vec(bias)), reshape(zeros(1+layer_sizes[1]),1,:))
	w[1][end,end] = 1
	for i in 2:(num_layers-1)
		weight = vcat(W[string("arr_", 2*(i-1))], zeros(2*sze,layer_sizes[i]))
		weight = hcat(weight, vcat(zeros(layer_sizes[i+1],2*sze), w_Im))
		bias   = vcat(W[string("arr_", 2*(i-1)+1)], b_I)
		w[i]   = vcat(hcat(weight, vec(bias)), reshape(zeros(1+layer_sizes[i]+2*sze),1,:))
		w[i][end,end] = 1
	end
	
	weight = B*Aₒᵤₜ*W[string("arr_", 2*(num_layers-1))]
	# weight = hcat(weight, [II -II])
	weight = hcat(weight, [A -A])
	bias   = B*(Aₒᵤₜ*W[string("arr_", 2*(num_layers-1)+1)] + bₒᵤₜ)
	w[end] = hcat(weight, vec(bias))

	# change layer sizes to be correct for this new network
	for i in 2:length(layer_sizes)-1
		layer_sizes[i] += 2*sze 
	end
	layer_sizes[end] = layer_sizes[1]

	weights = chain_net(w, copies, num_layers, layer_sizes)

	return weights
end
