# TO RUN: 
# $ julia --project=. vnn_run.jl "test/test_tiny.mat" "test/test_tiny.vnnlib" "test/test_tiny_output.txt"
# $ julia --project=. vnn_run.jl "test/test_small.mat" "test/test_small.vnnlib" "test/test_small_output.txt"
# $ julia --project=. vnn_run.jl "test/test_sat.mat" "test/test_prop.vnnlib" "test/test_sat_output.txt"
# $ julia --project=. vnn_run.jl "test/test_unsat.mat" "test/test_prop.vnnlib" "test/test_unsat_output.txt"
# $ julia --project=. vnn_run.jl "acasxu/ACASXU_run2a_5_7_batch_2000.mat" "acasxu/prop_3.vnnlib" "acasxu/prop_3_output.txt" 20
# $ julia --project=. vnn_run.jl "mnistfc/mnist-net_256x2.mat" "mnistfc/prop_0_0.03.vnnlib" "mnistfc/prop_0_0.03.txt" 200
# $ julia --project=. vnn_run.jl "mnistfc/mnist-net_256x6.mat" "mnistfc/prop_0_0.03.vnnlib" "mnistfc/prop_0_0.03.txt" 200

# $ julia --project=. "${project_path}/vnn_run.jl" "$ONNX_FILE" "$VNNLIB_FILE" "$RESULTS_FILE" "$TIMEOUT"



include("reach.jl")
include("load_vnn.jl")


function solve_problem(weights, A_in, b_in, A_out, b_out, output_filename)
	try
		for i in 1:length(b_in)
			ap2input, ap2output, ap2map, ap2backward, verification_res = compute_reach(weights, A_in[i], b_in[i], A_out, b_out, reach=false, back=false, verification=true)
			if verification_res == "violated"
				# Write result to output_filename
				open(output_filename, "w") do io
			    write(io, verification_res)
				end
			   return nothing
			end
		end

		# Write result to output_filename
		open(output_filename, "w") do io
	   write(io, verification_res)
		end

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
	weights = random_net(3, 2, 10, 5) # (in_d, out_d, hdim, layers)
	Aᵢ = [1. 0.; -1. 0.; 0. 1.; 0. -1.; 1. 1.; -1. 1.; 1. -1.; -1. -1.]
	bᵢ = [5., 5., 5., 5., 8., 8., 8., 8.]
	Aₒ = [1. 0.; -1. 0.; 0. 1.; 0. -1.]
	bₒ = [101., -100., 101., -100.]
	ap2input, ap2output, ap2map, ap2backward, verification_res = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], reach=false, back=false, verification=true)
	return nothing
end

# Solve on small problem to compile functions
small_compile()


# Load network and property
onnx_filename = ARGS[1]
mat_onnx_filename = string(onnx_filename[1:end-4], ".mat")
vnnlib_filename = ARGS[2]
output_filename = ARGS[3]
time_limit = parse(Float64, ARGS[4])

# weights, nnet, net_dict = nnet_load(nnet_filename)
if mat_onnx_filename == "test_tiny.mat" 
	weights = load_test_tiny()
elseif mat_onnx_filename == "test_small.mat"
	weights = load_test_small()
elseif mat_onnx_filename == "test_sat.mat" || mat_onnx_filename == "test_unsat.mat"
	weights = load_mat_onnx_test_acas(mat_onnx_filename)
elseif mat_onnx_filename[1:6] == "ACASXU"
	weights = load_mat_onnx_acas(mat_onnx_filename)
elseif mat_onnx_filename[1:9] == "mnist-net"
	weights = load_mat_onnx_mnist(mat_onnx_filename)
else
	# skip benchmark
	error("Got unexpected ONNX filename!")
end

A_in, b_in, A_out, b_out = get_constraints(vnnlib_filename)
@timeout solve_problem(weights, A_in, b_in, A_out, b_out, output_filename) time_limit