include("reach.jl")

### INPUT CONSTRAINT FUNCTIONS ###
# Returns H-rep of various input sets
# acas properties defined in original reluplex paper appendix
# For neural networks we often normalize the data before input to the network.
# This normalization is an affine map: x_net = (x - x_mean) ./ x_std --> x = Cx_net + d where C = Diagonal(x_std), d = x_mean
# We can then take our original input constraint, Ax≤b and substitute the above identity so it is properly defined in light of normalization: A(Cx_net + d)≤b
function input_constraints_acas(weights, type::String; net_dict=[])
	# Each input specification is in the form Ax≤b
	# The network takes normalized inputs: xₙ = Aᵢₙx + bᵢₙ
	# Thus the input constraints for raw network inputs is: A*inv(Aᵢₙ)x ≤ b + A*inv(Aᵢₙ)*bᵢₙ
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
	Aᵢₙ, bᵢₙ = net_dict["input_norm_map"]
	return A*inv(Aᵢₙ), b + A*inv(Aᵢₙ)*bᵢₙ
end

# Returns H-rep of various output sets
function output_constraints_acas(weights, type::String; net_dict=[])
	# Each output specification is in the form Ay≤b
	# The network takes normalized inputs: yₒᵤₜ = Aₒᵤₜy + bₒᵤₜ
	# Thus the output constraints for raw network outputs is: A*inv(Aₒᵤₜ)y ≤ b + A*inv(Aₒᵤₜ)*bₒᵤₜ
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
 	Aₒᵤₜ, bₒᵤₜ = net_dict["output_unnorm_map"]
 	return A*inv(Aₒᵤₜ), b + A*inv(Aₒᵤₜ)*bₒᵤₜ
end


###########################
######## SCRIPTING ########
###########################
weights, nnet, net_dict = acas_net_nnet(5,7)
property = "acas property 3"
Aᵢ, bᵢ = input_constraints_acas(weights, property, net_dict=net_dict)
Aₒ, bₒ = output_constraints_acas(weights, property, net_dict=net_dict)

# Multiple backward reachability queries can be solved by specifying multiple output sets i.e. [Aₒ₁, Aₒ₂], [bₒ₁, bₒ₂]
state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], reach=false, back=false, verification=true)
@show length(state2input)


## Solve for explicit policy ##
# We can identify exact input sets that lead to each possible output advisory
# Aₒ₁, bₒ₁ = output_constraints_acas(weights, "COC", net_dict=net_dict)
# Aₒ₂, bₒ₂ = output_constraints_acas(weights, "weak right", net_dict=net_dict)
# Aₒ₃, bₒ₃ = output_constraints_acas(weights, "strong right", net_dict=net_dict)
# Aₒ₄, bₒ₄ = output_constraints_acas(weights, "weak left", net_dict=net_dict)
# Aₒ₅, bₒ₅ = output_constraints_acas(weights, "strong left", net_dict=net_dict)

# state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₒ₁, Aₒ₂, Aₒ₃, Aₒ₄, Aₒ₅], [bₒ₁, bₒ₂, bₒ₃, bₒ₄, bₒ₅], reach=false, back=true, verification=false)
# for i in 1:length(state2backward)
# 	@show length(state2backward[i])
# end




