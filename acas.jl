include("reach.jl")

### INPUT CONSTRAINT FUNCTIONS ###
# Returns H-rep of various input sets
# acas properties defined in original reluplex paper appendix
function input_constraints_acas(weights, type::String; net_dict=[])
	# Each input specification is in the form Ax≤b
	# ACAS input  = [ρ, θ, ψ, v_own, v_int]
	if type == "acas property 3"
		A = [ -1  0  0  0  0; # ρ
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
	else
		error("Invalid input constraint specification.")
	end
	return Matrix{Float64}(A), Vector{Float64}(b)
end

# Returns H-rep of various output sets
function output_constraints_acas(weights, type::String; net_dict=[])
	# Each output specification is in the form Ayₒᵤₜ≤b
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
 		error("Invalid output constraint specification.")
 	end
 	return Matrix{Float64}(A), Vector{Float64}(b)
end


###########################
######## SCRIPTING ########
###########################
# weights, nnet, net_dict = acas_net_nnet(5,6)
weights = acas_net_nnet(5,7)

property = "acas property 3"
Aᵢ, bᵢ = input_constraints_acas(weights, property)
Aₒ, bₒ = output_constraints_acas(weights, property)

# Multiple backward reachability queries can be solved by specifying multiple output sets i.e. [Aₒ₁, Aₒ₂], [bₒ₁, bₒ₂]
ap2input, ap2output, ap2map, ap2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], reach=false, back=false, verification=true)
@show length(ap2input)


## Solve for explicit policy ##
# We can identify exact input sets that lead to each possible output advisory
# Aₒ₁, bₒ₁ = output_constraints_acas(weights, "COC", net_dict=net_dict)
# Aₒ₂, bₒ₂ = output_constraints_acas(weights, "weak right", net_dict=net_dict)
# Aₒ₃, bₒ₃ = output_constraints_acas(weights, "strong right", net_dict=net_dict)
# Aₒ₄, bₒ₄ = output_constraints_acas(weights, "weak left", net_dict=net_dict)
# Aₒ₅, bₒ₅ = output_constraints_acas(weights, "strong left", net_dict=net_dict)

# ap2input, ap2output, ap2map, ap2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₒ₁, Aₒ₂, Aₒ₃, Aₒ₄, Aₒ₅], [bₒ₁, bₒ₂, bₒ₃, bₒ₄, bₒ₅], reach=false, back=true, verification=false)
# for i in 1:length(ap2backward)
# 	@show length(ap2backward[i])
# end
