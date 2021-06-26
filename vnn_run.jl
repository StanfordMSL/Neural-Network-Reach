# TO RUN: 
# $ julia --project=. vnn_run.jl "test/test_tiny.mat" "test/test_tiny.vnnlib" "test/test_tiny_output.txt"
# $ julia --project=. vnn_run.jl "test/test_small.mat" "test/test_small.vnnlib" "test/test_small_output.txt"
# $ julia --project=. vnn_run.jl "test/test_sat.mat" "test/test_prop.vnnlib" "test/test_sat_output.txt"
# $ julia --project=. vnn_run.jl "test/test_unsat.mat" "test/test_prop.vnnlib" "test/test_unsat_output.txt"




include("reach.jl")


# Solve on small instance to compile functions


# Load network and property
mat_onnx_filename = ARGS[1]
vnnlib_filename = ARGS[2]
output_filename = ARGS[3]

# weights, nnet, net_dict = nnet_load(nnet_filename)
if mat_onnx_filename == "test/test_tiny.mat" 
	weights = load_test_tiny()
elseif mat_onnx_filename == "test/test_small.mat"
	weights = load_test_small()
elseif mat_onnx_filename == "test/test_sat.mat" || mat_onnx_filename == "test/test_unsat.mat"
	weights = load_mat_onnx_test(mat_onnx_filename)
elseif mat_onnx_filename[1:6] == "acasxu"
	weights = nothing
elseif mat_onnx_filename[1:7] == "mnistfc"
	weights = nothing
else
	# skip benchmark
end

# From vnnlib_filename parse constraints
if vnnlib_filename == "test/test_tiny.vnnlib" || vnnlib_filename == "test/test_small.vnnlib"
	Aᵢ, bᵢ = Matrix{Float64}(undef, 2, 1), Vector{Float64}(undef, 2)
	Aᵢ[1,1] = -1.; Aᵢ[2,1] = 1.
	bᵢ[1] = 1.; bᵢ[2] = 1. 
	Aₒ, bₒ =Matrix{Float64}(undef, 2, 1), Vector{Float64}(undef, 2)
	Aₒ[1,1] = -1.
	bₒ[1] = 100.
elseif vnnlib_filename == "test/test_prop.vnnlib"
	Aᵢ = Float64.([ -1  0  0  0  0; # ρ
		   1  0  0  0  0; # ρ
		   0 -1  0  0  0; # θ
		   0  1  0  0  0; # θ
		   0  0 -1  0  0; # ψ
		   0  0  1  0  0; # ψ
		   0  0  0 -1  0; # v_own
		   0  0  0  1  0; # v_own
		   0  0  0  0 -1; # v_int
		   0  0  0  0  1]) # v_int
	bᵢ = [0.30353115613746867, -0.29855281193475053, 0.009549296585513092, 0.009549296585513092, -0.4933803235848431, 0.49999999998567607, -0.3, 0.5, -0.3, 0.5]

	Aₒ = Float64.([1 -1 0 0 0;
		 1 0 -1 0 0;
		 1 0 0 -1 0;
		 1 0 0 0 -1])
	bₒ = [0., 0., 0., 0.]
end


# Solve verification problem
ap2input, ap2output, ap2map, ap2backward, verification_res = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], reach=false, back=false, verification=true)


# Write result to output_filename
open(output_filename, "w") do io
   write(io, verification_res)
end

