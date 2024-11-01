include("reach.jl")

### INPUT CONSTRAINT FUNCTIONS ###
# Returns H-rep of various input sets
# acas properties defined in original reluplex paper appendix
function input_constraints_acas(type::String; net_dict=[])
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
function output_constraints_acas(type::String)
	# Each output specification is in the form Ayₒᵤₜ≤b
	# lowest score of outputs corresponds to best action
	# ACAS output = [COC, weak right, strong right, weak left, strong left]
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



function verify_all_networks(property)
	# initialize data structures
	unsafe_aps = Vector{Dict{Int32, Vector{BitVector}}}()
	unsafe_vecs = Vector{BitVector}()

	# set input set and unsafe output set
	Aᵢ, bᵢ = input_constraints_acas(property)
	Aₒ, bₒ = output_constraints_acas(property)

	# iterate over each network
	for i in 1:5
		for j in 1:9
			weights = acas_net_nnet(i,j)
			ap_unsafe, vec_unsafe = verify_safety(weights, Aᵢ, bᵢ, [Aₒ], [bₒ])
			push!(unsafe_aps, ap_unsafe)
			push!(unsafe_vecs, vec_unsafe)
		end
	end
	return unsafe_aps, unsafe_vecs
end


###########################
######## SCRIPTING ########
###########################
property = "acas property 3"
Aᵢ, bᵢ = input_constraints_acas(property)
Aₒ, bₒ = output_constraints_acas(property)
weights = acas_net_nnet(5,9)

@time ap_unsafe, vec_unsafe = verify_safety(weights, Aᵢ, bᵢ, [Aₒ], [bₒ])








## Solve for explicit policy ##
explicit_policy = false
if explicit_policy == true
	# We can identify exact input sets that lead to each possible output advisory
	Aₒ₁, bₒ₁ = output_constraints_acas("COC")
	Aₒ₂, bₒ₂ = output_constraints_acas("weak right")
	Aₒ₃, bₒ₃ = output_constraints_acas("strong right")
	Aₒ₄, bₒ₄ = output_constraints_acas("weak left")
	Aₒ₅, bₒ₅ = output_constraints_acas("strong left")

	ap2input, ap2output, ap2map, ap2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₒ₁, Aₒ₂, Aₒ₃, Aₒ₄, Aₒ₅], [bₒ₁, bₒ₂, bₒ₃, bₒ₄, bₒ₅], back=true)

	using QHull
	volumes = zeros(5)
	for i in 1:5
		println("\nAction ", i)
		println("Number of regions = ", length(ap2backward[i]))

		for key in keys(ap2backward[i])
			# get Vrep
			A, b = ap2backward[i][key]
			vrep = tovrep(HPolytope(A, b)).vertices
			try
				volumes[i] += chull(reduce(vcat, transpose.(vrep))).volume
			catch
				nothing
			end
			
		end
		println("Volume = ", round(volumes[i], digits=0))
	end
end