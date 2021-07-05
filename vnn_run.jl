# TO RUN from home directory
# $ julia --project="Neural-Network-Reach/" Neural-Network-Reach/vnn_run.jl "./benchmarks/test/test_nano.onnx" "./benchmarks/test/test_nano.vnnlib" "Neural-Network-Reach/test/test_nano_output.txt" 200
# $ julia --project="Neural-Network-Reach/" Neural-Network-Reach/vnn_run.jl "./benchmarks/test/test_tiny.onnx" "./benchmarks/test/test_tiny.vnnlib" "Neural-Network-Reach/test/test_tiny_output.txt" 200
# $ julia --project="Neural-Network-Reach/" Neural-Network-Reach/vnn_run.jl "./benchmarks/test/test_small.onnx" "./benchmarks/test/test_small.vnnlib" "Neural-Network-Reach/test/test_small_output.txt" 200
# $ julia --project="Neural-Network-Reach/" Neural-Network-Reach/vnn_run.jl "./benchmarks/test/test_sat.onnx" "./benchmarks/test/test_prop.vnnlib" "Neural-Network-Reach/test/test_sat_output.txt" 200
# $ julia --project="Neural-Network-Reach/" Neural-Network-Reach/vnn_run.jl "./benchmarks/test/test_unsat.onnx" "./benchmarks/test/test_prop.vnnlib" "Neural-Network-Reach/test/test_unsat_output.txt" 10
# $ julia --project="Neural-Network-Reach/" Neural-Network-Reach/vnn_run.jl "./benchmarks/acasxu/ACASXU_run2a_5_7_batch_2000.onnx" "./benchmarks/acasxu/prop_3.vnnlib" "Neural-Network-Reach/acasxu/prop_3_output.txt" 200
# $ julia --project="Neural-Network-Reach/" Neural-Network-Reach/vnn_run.jl "./benchmarks/mnistfc/mnist-net_256x2.onnx" "./benchmarks/mnistfc/prop_0_0.03.vnnlib" "Neural-Network-Reach/mnistfc/prop_0_0.03.txt" 200
# $ julia --project="Neural-Network-Reach/" Neural-Network-Reach/vnn_run.jl "./benchmarks/mnistfc/mnist-net_256x6.onnx" "./benchmarks/mnistfc/prop_0_0.03.vnnlib" "Neural-Network-Reach/mnistfc/prop_0_0.03.txt" 20
using MAT

include("reach.jl")
include("load_vnn.jl")


function solve_problem(weights, A_in, b_in, A_out, b_out, output_filename)
	println("Functions compiled, running problem...")
	try
		verification_res = "unknown"
		for i in 1:length(b_in)
			ap2input, ap2output, ap2map, ap2backward, verification_res = compute_reach(weights, A_in[i], b_in[i], A_out, b_out, verification=true)
			if verification_res == "violated"
				open(output_filename, "w") do io
			    write(io, verification_res)
				end
			   return nothing
			end
		end
		open(output_filename, "w") do io
	    write(io, verification_res)
		end
		return nothing

	catch y
		if isa(y, InterruptException)
			println("timeout")
			open(output_filename, "w") do io
		    write(io, "timeout")
			end
		else
			@show y
			open(output_filename, "w") do io
			println("unknown")
		    write(io, "unknown")
			end
		end
	end
   return nothing
end

macro timeout(expr, seconds=-1, cb=(tsk) -> Base.throwto(tsk, InterruptException()))
    quote
        tsk = @task $expr
        schedule(tsk)
        if $seconds > -1
            Timer((timer) -> $cb(tsk), $seconds)
        end
        return fetch(tsk)
    end
end


# Solve on small problem to compile functions
function small_compile()
	weights = matread("/home/ubuntu/work/Neural-Network-Reach/small_weights.mat")["small_weights"] # for vnn_comp evaluation
	# weights = Vector{Matrix{Float64}}(matread("/Neural-Network-Reach/small_weights.mat")["small_weights"]) # for testing locally
	Aᵢ = [1. 0.; -1. 0.; 0. 1.; 0. -1.; 1. 1.; -1. 1.; 1. -1.; -1. -1.]
	bᵢ = [5., 5., 5., 5., 8., 8., 8., 8.]
	Aₒ = [1. 0.; -1. 0.; 0. 1.; 0. -1.]
	bₒ = [101., -100., 101., -100.]
	ap2input, ap2output, ap2map, ap2backward, verification_res = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], verification=true, verbose=false)
	return nothing
end




# Solve on small problem to compile functions
small_compile()

# Load in arguments
onnx_filename = ARGS[1]
mat_filename = string(onnx_filename[1:end-4], "mat")
vnnlib_filename = ARGS[2]
output_filename = ARGS[3]
time_limit = parse(Float64, ARGS[4])


# Get network weights
if mat_filename[end-17:end] == "test/test_nano.mat"
	weights = load_test_nano()
	vnnlib_filename = vnnlib_filename[end-20:end]

elseif mat_filename[end-17:end] == "test/test_tiny.mat" 
	weights = load_test_tiny()
	vnnlib_filename = vnnlib_filename[end-20:end]

elseif mat_filename[end-18:end] == "test/test_small.mat"
	weights = load_test_small()
	vnnlib_filename = vnnlib_filename[end-21:end]

elseif mat_filename[end-16:end] == "test/test_sat.mat" 
	weights = load_mat_onnx_test_acas("test/test_sat.mat")
	vnnlib_filename = vnnlib_filename[end-20:end]

elseif mat_filename[end-18:end] == "test/test_unsat.mat"
	weights = load_mat_onnx_test_acas("test/test_unsat.mat")
	vnnlib_filename = vnnlib_filename[end-20:end]

elseif mat_filename[end-37:end-32] == "acasxu"
	weights = load_mat_onnx_acas(mat_filename[end-37:end])
	if vnnlib_filename[end-19:end-14] == "acasxu"
		vnnlib_filename = vnnlib_filename[end-19:end]
	elseif vnnlib_filename[end-20:end-15] == "acasxu"
		vnnlib_filename = vnnlib_filename[end-20:end]
	end

elseif mat_filename[end-26:end-20] == "mnistfc"
	weights = load_mat_onnx_mnist(mat_filename[end-26:end])
	if vnnlib_filename[end-25:end-19] == "mnistfc"
		vnnlib_filename = vnnlib_filename[end-25:end]
	elseif vnnlib_filename[end-26:end-20] == "mnistfc"
		vnnlib_filename = vnnlib_filename[end-26:end]
	end

else
	# skip benchmark
	println("Got unexpected ONNX filename!")
	@show mat_filename
	weights = nothing
end

A_in, b_in, A_out, b_out = get_constraints(vnnlib_filename)
@timeout solve_problem(weights, A_in, b_in, A_out, b_out, output_filename) time_limit