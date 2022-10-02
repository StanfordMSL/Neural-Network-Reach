using MAT, NPZ, LinearAlgebra
include("nnet.jl")


```evaluate the network given input, weights, and how many copies chained together```
function eval_net(input, weights, copies::Int64)
	copies == 0 ? (return input) : nothing
	NN_out = vcat(input, [1.])
    for layer = 1:length(weights)-1
        NN_out = max.(0, weights[layer]*NN_out)
    end
    output = weights[end]*NN_out
    return eval_net(output, weights, copies-1)
end


```Generates a uniformly random number on "["a,b"]"```
bound_r(a,b) = (b-a)*(rand()-1) + b


```Generate random neural network with Kaiming initialization```
function random_net(in_d, out_d, hdim, layers)
	α = sqrt(2/515)
	Weights = Vector{Array{Float64,2}}(undef,layers)
	r_weight = α*(2*rand(hdim, in_d) - rand(hdim, in_d))
	r_bias   = α*(2*rand(hdim, 1) - rand(hdim, 1))
	Weights[1] = vcat(hcat(r_weight, r_bias), reshape(zeros(1+in_d),1,:))
	Weights[1][end,end] = 1
	for i in 2:layers-1
		r_weight = α*(2*rand(hdim, hdim) - rand(hdim, hdim))
		r_bias   = α*(2*rand(hdim, 1) - rand(hdim, 1))
		Weights[i] = vcat(hcat(r_weight, r_bias), reshape(zeros(1+hdim),1,:))
		Weights[i][end,end] = 1
	end
	r_weight = α*(2*rand(out_d, hdim) - rand(out_d, hdim))
	r_bias   = α*(2*rand(out_d, 1) - rand(out_d, 1))
	Weights[end] = hcat(r_weight, r_bias)
	return Weights
end


``` 
Load nnet network 
ex: filename = "models/ACAS_nnet/ACASXU_experimental_v2a_1_1.nnet" 
```
function nnet_load(filename)
	nnet = NNet(filename)

	σᵢ = Diagonal(nnet.ranges[1:end-1])
	μᵢ = nnet.means[1:end-1]
	σₒ = nnet.ranges[end]*Matrix{Float64}(I, nnet.outputSize, nnet.outputSize)
	μₒ = nnet.means[end]*ones(nnet.outputSize)
	Aᵢₙ, bᵢₙ = inv(σᵢ), -inv(σᵢ)*μᵢ
	Aₒᵤₜ, bₒᵤₜ = σₒ, μₒ

	weights = Vector{Array{Float64,2}}(undef, nnet.numLayers)
	weight = nnet.weights[1]*Aᵢₙ
	bias   = vec(nnet.biases[1]) + nnet.weights[1]*bᵢₙ
	weights[1]   = vcat(hcat(weight, vec(bias)), reshape(zeros(1+nnet.layerSizes[1]),1,:))
	weights[1][end,end] = 1
	for i in 2:(nnet.numLayers-1)
		weight = nnet.weights[i]
		bias   = vec(nnet.biases[i])
		weights[i]   = vcat(hcat(weight, vec(bias)), reshape(zeros(1+nnet.layerSizes[i]),1,:))
		weights[i][end,end] = 1
	end
	# last layer weight shouldn't carry forward the bias term. i.e. augmented but with last row removed
	weight = Aₒᵤₜ*nnet.weights[end]
	bias   = Aₒᵤₜ*vec(nnet.biases[end]) + bₒᵤₜ
	weights[end] = hcat(weight, vec(bias))
	return weights
end


``` Load ACAS Networks ```
function acas_net_nnet(a::Int64, b::Int64)
	filename = string("models/ACAS_nnet/ACASXU_experimental_v2a_", a, "_", b, ".nnet")
	return nnet_load(filename)
end


``` chain together multiple networks ```
function chain_net(w, copies, num_layers)
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
			w̄ₒ = vcat(w[end], reshape(zeros(size(w[end],2)),1,:))
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


``` load pendulum network with normalization ```
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
	
	weights = chain_net(w, copies, num_layers)

	return weights
end


#=
Load pytorch networks saved as numpy variables.
This function assumes the weights and normalization parameters are saved as follows:

weights = []
for name, param in model.named_parameters():
    weights.append(param.detach().numpy())

numpy.savez("weights.npz", *weights)
numpy.savez("params.npz", X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std, layer_sizes=layer_sizes)

where:

X_mean, X_std = numpy.mean(X, axis=0), numpy.std(X, axis=0)
Y_mean, Y_std = numpy.mean(Y, axis=0), numpy.std(Y, axis=0)
=#
function pytorch_net(nn_weights, nn_params, copies::Int64)
	W = npzread(nn_weights)
	params = npzread(nn_params)

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

	weights = chain_net(w, copies, num_layers)

	return weights
end