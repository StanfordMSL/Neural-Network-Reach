using Flux, Tracker, JLD2, FileIO, MAT, NPZ, LinearAlgebra
include("nnet.jl")


# Evaluate network. Takes care of normalizing inputs and un-normalizing outputs
function eval_net(input, net_dict, copies::Int64)
	copies == -1 ? (return input) : nothing
	NN_out = (input - net_dict["X_mean"]') ./ net_dict["X_std"]'
    for layer = 1:length(net_dict["weights"])-1
        NN_out = relu.(net_dict["weights"][1,layer]*NN_out + net_dict["biases"][1,layer]')
    end
    output = vec( (net_dict["weights"][1,end]*NN_out + net_dict["biases"][1,end]') .* net_dict["Y_std"]' + net_dict["Y_mean"]' )
    return eval_net(output, net_dict, copies-1)
end

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

### LOAD NETWORKS ###
# Want to load each network in such that I get:
# ⋅ augmented weights
# ⋅	numLayers (int): Number of weight matrices or bias vectors in neural network
# ⋅ layerSizes (list of ints): Size of input layer, hidden layers, and output layer
# ⋅ inputSize (int): Size of input
# ⋅ outputSize (int): Size of output
# ⋅ input normalization map
# ⋅ output unnormalization map


# Converts network weights and biases to just weights. i.e. only linear maps, not affine maps.
# Works for Dense Flux networks
function flux2augmented(NN)
	Weights = Vector{Array{Float64,2}}(undef,length(NN))
	for layer in 1:(length(NN)-1)
		Weights[layer] = vcat(hcat(Tracker.data(NN[layer].W), Tracker.data(NN[layer].b)), reshape(zeros(1+size(NN[layer].W,2)),1,:))
		Weights[layer][end,end] = 1
	end
	# last layer weight shouldn't carry forward the bias term. i.e. augmented but with last row removed
	Weights[length(NN)] = hcat(Tracker.data(NN[length(NN)].W), Tracker.data(NN[length(NN)].b))
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

# Load Pendulum Networks #
#= 
net = matread("models/Pendulum/NN_params_pendulum_0_1s_1e7data_a15_12_2_L1.mat")
"weights"
"biases"
"X_mean"
"X_std"
"Y_mean"
"Y_std"
input  = [θ, θ_dot]_t
output = [θ, θ_dot]_t+1
=#

function pendulum_net2(filename::String, copies::Int64)
	model = matread(filename)
	num_layers = length(model["weights"])
	layer_sizes = vcat(size(model["weights"][1], 2), [length(model["biases"][i]) for i in 1:num_layers])

	# make net_dict
	σᵢ = Float64.(Diagonal(vec(model["X_std"])))
	μᵢ = Float64.(vec(model["X_mean"]))
	σₒ = Float64.(Diagonal(vec(model["Y_std"])))
	μₒ = Float64.(vec(model["Y_mean"]))
	Aᵢₙ, bᵢₙ = inv(σᵢ), -inv(σᵢ)*μᵢ
	Aₒᵤₜ, bₒᵤₜ = σₒ, μₒ

	net_dict = Dict()
	net_dict["num_layers"] = num_layers
	net_dict["layer_sizes"] = layer_sizes
	net_dict["input_size"] = layer_sizes[1]
	net_dict["output_size"] = layer_sizes[end]
	net_dict["input_norm_map"] = (Aᵢₙ, bᵢₙ)
	net_dict["output_unnorm_map"] = (Aₒᵤₜ, bₒᵤₜ) 


	w = Vector{Array{Float64,2}}(undef, num_layers)
	for i in 1:(num_layers-1)
		w[i] = vcat(hcat(model["weights"][i], vec(model["biases"][i])), reshape(zeros(1+layer_sizes[i]),1,:))
		w[i][end,end] = 1
	end
	w[end] = hcat(model["weights"][end], vec(model["biases"][end]))
	
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
			Āₒ = vcat(hcat(Aₒᵤₜ, bₒᵤₜ), reshape(zeros(1+layer_sizes[end]),1,:))
			Āₒ[end,end] = 1
			Āᵢ = vcat(hcat(Aᵢₙ, bᵢₙ), reshape(zeros(1+layer_sizes[1]),1,:))
			Āᵢ[end,end] = 1
			
			weights[k] = w[1]*Āᵢ*Āₒ*w̄ₒ
			w_idx = 2
		else
			weights[k] = w[w_idx]
			w_idx += 1
		end
	end

	return weights, net_dict
end


function pendulum_net(model::String, copies::Int64)
	net_dict = matread(model)
	σ_x = Diagonal(vec(net_dict["X_std"]))
	μ_x = vec(net_dict["X_mean"])
	σ_y = Diagonal(vec(net_dict["Y_std"]))
	μ_y = vec(net_dict["Y_mean"])

	layers = [Dense(net_dict["weights"][1], net_dict["biases"][1]', relu)]
	for net_copy in 1:copies # copies = 0 means just original network
		layers = vcat(layers, [Dense(net_dict["weights"][i], net_dict["biases"][i]', relu) for i in 2:length(net_dict["weights"])-1])
		W = net_dict["weights"][1]*inv(σ_x)*σ_y*net_dict["weights"][end]
		b = net_dict["weights"][1]*inv(σ_x)*(σ_y*net_dict["biases"][end]' + μ_y - μ_x) + net_dict["biases"][1]'
		layers = vcat(layers, [Dense(W, b, relu)])
	end
	layers = vcat(layers, [Dense(net_dict["weights"][i], net_dict["biases"][i]', relu) for i in 2:length(net_dict["weights"])-1])
	layers = vcat(layers, [Dense(net_dict["weights"][end], net_dict["biases"][end]', identity)])
	flux_net = Chain(layers...)
	str = string("pend_net_", copies, ".mat")
	matwrite(str,
	Dict(
	"W" => [Float64.(flux_net[i].W) for i in 1:length(flux_net)],
	"b" => [Float64.(flux_net[i].b) for i in 1:length(flux_net)],
	"range_for_scaling" => [1.0, 1.0, 1.0],
	"means_for_scaling" => [0.0, 0.0, 0.0]))
	return flux2augmented(flux_net), net_dict
end


# Load ACAS Networks #
# NNET #
function acas_net_nnet(a::Int64, b::Int64)
	filename = string("models/ACAS_nnet/ACASXU_experimental_v2a_", a, "_", b, ".nnet")
	return nnet_load(filename)
end



