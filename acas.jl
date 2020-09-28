using BenchmarkTools
include("reach.jl")
pyplot()

bound_r(a,b) = (b-a)*(rand()-1) + b # Generates a uniformly random number on [a,b]

### INPUT CONSTRAINT FUNCTIONS ###
# acas properties defined in original reluplex paper appendix
# For neural networks we often normalize the data before input to the network.
# This normalization is an affine map: x_net = (x - x_mean) ./ x_std --> x = Cx_net + d where C = Diagonal(x_std), d = x_mean
# We can then take our original input constraint, Ax≤b and substitute the above identity so it is properly defined in light of normalization: A(Cx_net + d)≤b
function input_constraints_acas(weights, type::String; net_dict=[])
	# ACAS input  = [ρ, θ, ψ, v_own, v_int]
	if type == "acas property 3"
		A = [-1  0  0  0  0; # ρ
			   1  0  0  0  0; # ρ
			   0 -1  0  0  0; # θ
			   0  1  0  0  0; # θ
			   0  0 -1  0  0; # ψ
			   0  0  1  0  0; # ψ
			   0  0  0 -1  0; # v_own
			   0  0  0  1  0; # v_own
			   0  0  0  0 -1; # v_int
 			   0  0  0  0  1] # v_int
 		b = [-1500, 1800, 0.06, 0.06, -3.1, 3.14, -980, 1200, -960, 1200]
		σ = Diagonal(vec(net_dict["range_for_scaling"][1:end-1]))
		μ = vec(net_dict["means_for_scaling"][1:end-1])
		Aᵢ = A*σ
		bᵢ = b - A*μ

	elseif type == "acas property 4"
		A = [-1  0  0  0  0; # ρ
			   1  0  0  0  0; # ρ
			   0 -1  0  0  0; # θ
			   0  1  0  0  0; # θ
			   0  0 -1  0  0; # ψ
			   0  0  1  0  0; # ψ
			   0  0  0 -1  0; # v_own
			   0  0  0  1  0; # v_own
			   0  0  0  0 -1; # v_int
 			   0  0  0  0  1] # v_int
 		b = [-1500, 1800, 0.06, 0.06, 0.0, 0.0, -1000, 1200, -700, 800]
		σ = Diagonal(vec(net_dict["range_for_scaling"][1:end-1]))
		μ = vec(net_dict["means_for_scaling"][1:end-1])
		Aᵢ = A*σ
		bᵢ = b - A*μ
	
	else
		error("Invalid input constraint specification.")
	end
	
	return Aᵢ, bᵢ
end

function output_constraints_acas(weights, type::String; net_dict=[])
	if type == "acas property 3" || type == "acas property 4" || type == "COC"
		A = [1 -1 0 0 0;
			 1 0 -1 0 0;
			 1 0 0 -1 0;
			 1 0 0 0 -1]
 		b = [0, 0, 0, 0]
	elseif type == "weak right"
		A = [-1 1 0 0 0;
			 0 1 -1 0 0;
			 0 1 0 -1 0;
			 0 1 0 0 -1]
 		b = [0, 0, 0, 0]
	elseif type == "strong right"
		A = [-1 0 1 0 0;
			 0 -1 1 0 0;
			 0 0 1 -1 0;
			 0 0 1 0 -1]
 		b = [0, 0, 0, 0]
	elseif type == "weak left"
		A = [-1 0 0 1 0;
			 0 -1 0 1 0;
			 0 0 -1 1 0;
			 0 0 0 1 -1]
 		b = [0, 0, 0, 0]
	elseif type == "strong left"
		A = [-1 0 0 0 1;
			 0 -1 0 0 1;
			 0 0 -1 0 1;
			 0 0 0 -1 1]
 		b = [0, 0, 0, 0]
 	else 
 		error("Invalid input constraint specification.")
 	end
 	σ = net_dict["range_for_scaling"][end]
	μ = net_dict["means_for_scaling"][end]
	Aₒ = σ*A
	bₒ = b - A*μ*ones(size(A,2))
 	return Aₒ, bₒ
end


######################################################################
## ACAS Examples ##
weights, net_dict = acas_net(5,7)
property = "acas property 3"
Aᵢ, bᵢ = input_constraints_acas(weights, property, net_dict=net_dict)
Aₒ, bₒ = output_constraints_acas(weights, property, net_dict=net_dict)

function profile(weights, Aᵢ, bᵢ, Aₒ, bₒ)
	state2input, state2output, state2map, state2backward = forward_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], reach=false, back=true, verification=false)
	@show length(state2input)
	return nothing
end

bm = @benchmark state2input, state2output, state2map, state2backward = forward_reach($weights, $Aᵢ, $bᵢ, $[Aₒ], $[bₒ], reach=false, back=false, verification=true)


## Solve for explicit policy ##
# This doesn't really incur extra solve time
# Aₒ₁, bₒ₁ = output_constraints_acas(weights, "COC", net_dict=net_dict)
# Aₒ₂, bₒ₂ = output_constraints_acas(weights, "weak right", net_dict=net_dict)
# Aₒ₃, bₒ₃ = output_constraints_acas(weights, "strong right", net_dict=net_dict)
# Aₒ₄, bₒ₄ = output_constraints_acas(weights, "weak left", net_dict=net_dict)
# Aₒ₅, bₒ₅ = output_constraints_acas(weights, "strong left", net_dict=net_dict)

# bm = @benchmark state2input, state2output, state2map, state2backward = forward_reach($weights, $Aᵢ, $bᵢ, $[Aₒ₁, Aₒ₂, Aₒ₃, Aₒ₄, Aₒ₅], $[bₒ₁, bₒ₂, bₒ₃, bₒ₄, bₒ₅], reach=false, back=true, verification=false)
# for i in 1:length(state2backward)
# 	@show length(state2backward[i])
# end




