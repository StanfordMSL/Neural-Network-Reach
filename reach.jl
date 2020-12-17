using Plots, LinearAlgebra, JuMP, GLPK, LazySets, Polyhedra, CDDLib
include("load_networks.jl")
include("unique_custom.jl")

ϵ = 1e-10 # used for numerical tolerances throughout

### GENERAL PURPOSE FUNCTIONS ###
function normalize_row(row::Vector{Float64}; return_zero=false)
	for i in 1:length(row)
		if abs(row[i]) > 1e-12 # ith element nonzero
			return row ./ abs(row[i]) 
		elseif i == length(row) && return_zero
			return zeros(length(row))
		elseif i == length(row)
			@show row
			error("Rethink how you normalize rows!")
		end
	end
end

# Returns the algorithm initialization point given task and input space constraints
function get_input(Aᵢ, bᵢ)
	input, nothing, nothing = cheby_lp([], [], Aᵢ, bᵢ, [])	
	return input
end

# Given input point, perform forward pass to get state.
function get_state(input, weights)
	L = length(weights)
	state = Vector{BitVector}(undef, L-1)
	layer_vals = vcat(input, [1])
	for layer in 1:L-1 # No ReLU on last layer
		layer_vals = max.(0, weights[layer]*layer_vals)
		state[layer] = layer_vals .> ϵ
	end
	return state
end

# Given constraint index, return associated layer and neuron
function get_layer_neuron(index, state)
	layer, neuron, prev = (1, 1, 0)
	for l in 1:length(state)
		if index > length(state[l]) + prev
			prev += length(state[l])
		else
			return l, (index - prev)
		end
	end
end

# Get hyperplane equation associated with a given neuron and input
function neuron_map(layer, neuron, state, weights; normalize=true)
	matrix = I 
	for l in 1:layer-1
		matrix = diagm(0 => state[l])*weights[l]*matrix
	end
	matrix = weights[layer]*matrix
	if normalize
		return normalize_row(matrix[neuron,:], return_zero=true)
	else
		return matrix[neuron,:]
	end
end

# Get affine map associated with a state
function local_map(state::Vector{BitVector}, weights::Vector{Matrix{Float64}})
	matrix = I 
	for l in 1:length(weights)-1
		matrix = diagm(0 => state[l])*weights[l]*matrix
	end
	matrix = weights[end]*matrix
	
	return matrix[:,1:end-1], matrix[:,end] # C,d of y = Cx+d
end















### FUNCTIONS FOR GETTING CONSTRAINTS ###
# Get redundant A≤b constraints
function get_constraints(weights::Vector{Matrix{Float64}}, state::Vector{BitVector}, num_neurons)
	L = length(weights)

	# Initialize necessary data structures #
	idx2repeat = Dict{Int64,Vector{Int64}}() # Dict from indices of A to indices of A that define the same constraint (including itself)
	zerows = Vector{Int64}() # indices that correspond to degenerate constraints
	A = Matrix{Float64}(undef, num_neurons, size(weights[1],2)) # constraint matrix. A[:,1:end-1]x ≤ -A[:,end]
	lin_map = I

	# build constraint matrix #
	i = 1
	for layer in 1:L-1
		output = weights[layer]*lin_map
		for neuron in 1:length(state[layer])
			A[i,:] = (1-2*state[layer][neuron])*output[neuron,:]
			if !isapprox(A[i,1:end-1], zeros(size(A,2)-1), atol=ϵ) # check nonzero.
				A[i,:] = normalize_row(A[i,:])
			else
				push!(zerows, i)
			end
			i += 1
		end
		lin_map = diagm(0 => state[layer])*weights[layer]*lin_map
	end

	unique_rows, unique_row = unique_custom(A, dims=1)
	for i in 1:length(unique_row)
		idx2repeat[i] = findall(x-> x == i, unique_row)
	end
	unique_nonzerow_indices = setdiff(unique_rows, zerows)

	return A[:,1:end-1], -A[:,end], idx2repeat, zerows, unique_nonzerow_indices
end













### FUNCTIONS FOR REMOVING REDUNDANT CONSTRAINTS ###
# Remove redundant constraints (rows of A,b).
# If we relax a constraint, can we push past where its initial 'b' value was? It's essential iff yes.
function remove_redundant(A, b, Aᵢ, bᵢ, unique_nonzerow_indices, essential)
	redundant, redundantᵢ = remove_redundant_bounds(A, b, Aᵢ, bᵢ, unique_nonzerow_indices)
	non_redundant  = setdiff(unique_nonzerow_indices, redundant) # working non-redundant set
	non_redundantᵢ = setdiff(collect(1:length(bᵢ)), redundantᵢ)  # working non-redundant set
	essentialᵢ = Vector{Int64}() #; essential  = Vector{Int64}()
	unknown_set    = setdiff(non_redundant, essential)           # working non-redundant and non-essential set
	unknown_setᵢ   = setdiff(non_redundantᵢ, essentialᵢ)         # working non-redundant and non-essential set

	essential, essentialᵢ = exact_lp_remove(A, b, Aᵢ, bᵢ, essential, essentialᵢ, non_redundant, non_redundantᵢ, unknown_set, unknown_setᵢ)
	essential == [] && essentialᵢ == [] ? error("No essential constraints!") : nothing
	saved_lps = length(essential)+length(essentialᵢ) + length(redundant)+length(redundantᵢ)-2*size(A,2)
	solved_lps = 2*size(A,2) + length(unknown_set) + length(unknown_setᵢ)

	return vcat(A[essential,:], Aᵢ[essentialᵢ,:]), vcat(b[essential], bᵢ[essentialᵢ]), sort(essential), saved_lps, solved_lps
end

# Heuristic for finding redundant constraints: finds upper and lower bounds for each component of x given Ax≤b
function remove_redundant_bounds(A, b, Aᵢ, bᵢ, unique_nonzerow_indices; presolve=false)
	redundant = Vector{Int64}()
	redundantᵢ = Vector{Int64}()
	bounds = Matrix{Float64}(undef,size(A,2),2) # [min₁ max₁; ... ;minₙ maxₙ]
	model = Model(GLPK.Optimizer)
	presolve ? set_optimizer_attribute(model, "presolve", 1) : nothing
	@variable(model, x[1:size(A,2)])
	@constraint(model, A[unique_nonzerow_indices,:]*x .<= b[unique_nonzerow_indices])
	@constraint(model, Aᵢ*x .<= bᵢ)
	@constraint(model, x[1:size(A,2)] .<= 1e8*ones(size(A,2)))  # to keep bounded
	@constraint(model, x[1:size(A,2)] .>= -1e8*ones(size(A,2))) # to keep bounded
	for i in 1:size(A,2)
		for j in [1,2]
			j == 1 ? @objective(model, Min, x[i]) : @objective(model, Max, x[i])
			optimize!(model)
			if termination_status(model) == MOI.OPTIMAL
				bounds[i,j] = objective_value(model)
			elseif termination_status(model) == MOI.NUMERICAL_ERROR && !presolve
				return remove_redundant_bounds(A, b, Aᵢ, bᵢ, unique_nonzerow_indices, presolve=true)
			else
				@show termination_status(model)
				error("Suboptimal LP bounding box!")
			end
		end
	end

	# Find redundant constraints
	for i in unique_nonzerow_indices
		val = sum([A[i,j] > 0 ? A[i,j]*bounds[j,2] : A[i,j]*bounds[j,1] for j in 1:size(A,2)])
		val + ϵ < b[i]  ? push!(redundant,i) : nothing
	end
	for i in length(bᵢ)
		val = sum([Aᵢ[i,j] > 0 ? Aᵢ[i,j]*bounds[j,2] : Aᵢ[i,j]*bounds[j,1] for j in 1:size(Aᵢ,2)])
		val + ϵ < bᵢ[i]  ? push!(redundantᵢ,i) : nothing
	end
	return redundant, redundantᵢ
end

# Brute force removal of LP constraints given we've performed heurstics to find essential and redundant constraints already
# Dynamic memory allocation might be slowing this down (e.g. push!)
function exact_lp_remove(A, b, Aᵢ, bᵢ, essential, essentialᵢ, non_redundant, non_redundantᵢ, unknown_set, unknown_setᵢ; presolve=false)
	model = Model(GLPK.Optimizer)
	presolve ? set_optimizer_attribute(model, "presolve", 1) : nothing
	@variable(model, x[1:size(A,2)])
	@constraint(model, con[j in non_redundant], dot(A[j,:],x) <= b[j])
	@constraint(model, conₛ[j in non_redundantᵢ], dot(Aᵢ[j,:],x) <= bᵢ[j]) # tack on non-redundant superset constraints

	for (k,i) in enumerate(unknown_set)
		@objective(model, Max, dot(A[i,:],x))
		set_normalized_rhs(con[i], b[i]+100) # relax ith constraint
		k > 1 ? set_normalized_rhs(con[unknown_set[k-1]], b[unknown_set[k-1]]) : nothing # un-relax i-1 constraint
		optimize!(model)
		if termination_status(model) == MOI.OPTIMAL

			if objective_value(model) > b[i] + ϵ # 1e-15 is too small.
				push!(essential, i)
			end
		elseif termination_status(model) == MOI.NUMERICAL_ERROR && !presolve
			return exact_lp_remove(A, b, Aᵢ, bᵢ, essential, essentialᵢ, non_redundant, non_redundantᵢ, unknown_set, unknown_setᵢ, true)
		else
			@show termination_status(model)
			println("Dual infeasible implies that primal is unbounded.")
			error("Suboptimal LP constraint check!")
		end
	end

	# Brute force LP method on superset constraints#
	for (k,i) in enumerate(unknown_setᵢ)
		@objective(model, Max, dot(Aᵢ[i,:],x))
		set_normalized_rhs(conₛ[i], bᵢ[i]+100) # relax ith constraint
		k > 1 ? set_normalized_rhs(conₛ[unknown_setᵢ[k-1]], bᵢ[unknown_setᵢ[k-1]]) : nothing # un-relax i-1 constraint
		optimize!(model)
		if termination_status(model) == MOI.OPTIMAL
			if objective_value(model) > bᵢ[i] + ϵ
				push!(essentialᵢ, i)
			end
		elseif termination_status(model) == MOI.NUMERICAL_ERROR && !presolve
			return exact_lp_remove(A, b, Aᵢ, bᵢ, essential, essentialᵢ, non_redundant, non_redundantᵢ, unknown_set, unknown_setᵢ, presolve=true)
		else
			@show termination_status(model)
			println("Dual infeasible implies that primal is unbounded.")
			error("Suboptimal LP constraint check!")
		end
	end
	return essential, essentialᵢ
end
















### FUNCTIONS FOR FINDING NEIGHBORS ###
function find_neighbors(state::Vector{BitVector}, neighbor_indices::Vector{Int64}, idx2repeat::Dict{Int64,Vector{Int64}}, zerows::Vector{Int64}, weights::Vector{Matrix{Float64}}, visited)
	neighbors = Set{Vector{BitVector}}()
	for idx in neighbor_indices
		neighbor_state = deepcopy(state)
		neighbor_constraint = -normalize_row(neuron_map(get_layer_neuron(idx, neighbor_state)..., neighbor_state, weights))

		neighbor_state = type1!(idx2repeat[idx], neighbor_state, weights)
		neighbor_state = type3!(zerows, neighbor_state, weights, neighbor_constraint)

		if neighbor_state ∉ visited # check if we've already visited
			push!(neighbors, neighbor_state)
		end
	end 
	return neighbors
end

# Adds neighbor states to working_set
function add_neighbor_states(state::Vector{BitVector}, neighbor_indices::Vector{Int64}, working_set, idx2repeat::Dict{Int64,Vector{Int64}}, zerows::Vector{Int64}, weights::Vector{Matrix{Float64}}, state2essential)	
	for idx in neighbor_indices
		neighbor_state = deepcopy(state)
		neighbor_constraint = -normalize_row(neuron_map(get_layer_neuron(idx, neighbor_state)..., neighbor_state, weights))

		neighbor_state = type1!(idx2repeat[idx], neighbor_state, weights)
		neighbor_state = type3!(zerows, neighbor_state, weights, neighbor_constraint)

		if !haskey(state2essential, neighbor_state) && neighbor_state ∉ working_set
			push!(working_set, neighbor_state)
			state2essential[neighbor_state] = idx2repeat[idx] # All of the neurons that define the neighbor constraint
		end
	end 

	return working_set, state2essential
end

# Handle flipping for type 1 neurons
function type1!(set, neighbor_state, weights)
	for neuron_idx in set # Type 1 neurons
		layer, neuron = get_layer_neuron(neuron_idx, neighbor_state)
		new_constraint = -(1-2*neighbor_state[layer][neuron])*neuron_map(layer, neuron, neighbor_state, weights) # flipped

		if isapprox(new_constraint, zeros(length(new_constraint)), atol=ϵ ) # sometimes previous flipping leads to zerow
			neighbor_state[layer][neuron] = 0
		else
			neighbor_state[layer][neuron] = !neighbor_state[layer][neuron]
		end
	end
	return neighbor_state
end

# Handle flipping for type 3 neurons
function type3!(set, neighbor_state, weights, neighbor_constraint)
	for neuron_idx in set # Type 3 neurons
		layer, neuron = get_layer_neuron(neuron_idx, neighbor_state)
		new_constraint = -neuron_map(layer, neuron, neighbor_state, weights) # negative because testing 0->1 flip
		if !isapprox(new_constraint, zeros(length(new_constraint)), atol=ϵ )
			neighbor_state[layer][neuron] = 1
		end
	end
	return neighbor_state
end
















### FUNCTIONS FOR PERFORMING REACHABILITY ###
# Solve a Feasibility LP to check if two polyhedron intersect
function poly_intersection(A₁, b₁, A₂, b₂; presolve=false)
	dim = max(size(A₁,2), size(A₂,2)) # In the case one of them is empty
	model = Model(GLPK.Optimizer)
	presolve ? set_optimizer_attribute(model, "presolve", 1) : nothing
	@variable(model, x[1:dim])
	@objective(model, MOI.FEASIBILITY_SENSE, 0)
	@constraint(model,  A₁*x .≤ b₁)
	@constraint(model,  A₂*x .≤ b₂)
	optimize!(model)

	if termination_status(model) == MOI.INFEASIBLE # no intersection
		return false
	elseif termination_status(model) == MOI.OPTIMAL # intersection
		return true
	elseif termination_status(model) == MOI.NUMERICAL_ERROR && !presolve
		return poly_intersection(A₁, b₁, A₂, b₂, presolve=true)
	else
		@show termination_status(model)
		@show A₁; @show b₁; @show A₂; @show b₂
		error("Intersection LP error!")
	end
end

# Find {y | Ax≤b and y=Cx+d} for the case where C is not invertible
function affine_map(A,b,C,d)
	if rank(C) == length(d)
		return A*inv(C), b + A*inv(C)*d
	end

	xdim = size(A,2)
	ydim = size(C,1)
	A′ = vcat( hcat(-C, I), hcat(C, -I), hcat(A, zeros(length(b), ydim)) )
	b′ = vcat(d, -d, b)

	poly_in = polyhedron(hrep(A′,b′), CDDLib.Library(:float))
	poly_out = eliminate(poly_in, collect(ydim+1:ydim+xdim), FourierMotzkin())
	ine = unsafe_load(poly_out.ine.matrix)

	numrow = ine.rowsize
	numcol = ine.colsize
	Aₒ = Matrix{Float64}(undef, numrow, numcol-1)
	bₒ = Vector{Float64}(undef, numrow)
	for i in 1:numrow
		row = unsafe_load(ine.matrix, i)
		for j in 1:numcol-1
			Aₒ[i,j] = unsafe_load(row, j)
		end
		bₒ[i] = unsafe_load(row, numcol)
	end
	return Aₒ, bₒ
end
















### FUNCTIONS TO VERIFY ACTIVATION PATTERN ###
# Make sure we generate the correct state for a region
function check_state(input, weights, state)
	if state != get_state(input, weights) 
		@show input;  @show state;  @show get_state(input, weights)
		error("NN state not what it seems!")
	end
	return nothing
end

# Solve Chebyshev Center LP
# "an optimal dual variable is nonzero only if its associated constraint in the primal is binding", http://web.mit.edu/15.053/www/AMP-Chapter-04.pdf
function cheby_lp(A, b, Aᵢ, bᵢ, unique_nonzerow_indices; presolve=false)
	dim = max(size(A,2), size(Aᵢ,2)) # In the case one of them is empty
	model = Model(GLPK.Optimizer)
	# set_optimizer_attribute(model, "dual", 1)
	presolve ? set_optimizer_attribute(model, "presolve", 1) : nothing
	@variable(model, r)
	@variable(model, x_c[1:dim])
	@objective(model, Max, r)

	for i in 1:length(b)
		if i ∈ unique_nonzerow_indices
			@constraint(model, dot(A[i,:],x_c) + r*norm(A[i,:]) ≤ b[i])
		else
			@constraint(model, 0*r ≤ 0) # non-constraint so we have a dual variable for each index in A
		end
	end
	for i in 1:length(bᵢ)
		@constraint(model, dot(Aᵢ[i,:],x_c) + r*norm(Aᵢ[i,:]) ≤ bᵢ[i]) # superset constraints
	end
	@constraint(model,  r ≤ 1e4) # prevents unboundedness
	@constraint(model, -r ≤ -1e-15) # Must have r>0
	optimize!(model)

	if termination_status(model) == MOI.OPTIMAL
		constraints = all_constraints(model, GenericAffExpr{Float64,VariableRef}, MOI.LessThan{Float64})
		length(constraints)-2 != length(b)+length(bᵢ) ? (error("Not enough dual variables!")) : nothing
		essential  = [i for i in 1:length(b) if abs(dual(constraints[i])) > 1e-4]
		essentialᵢ = [i-length(b) for i in length(b)+1:length(constraints)-2 if abs(dual(constraints[i])) > 1e-4]
		return value.(x_c), essential, essentialᵢ
	elseif termination_status(model) == MOI.NUMERICAL_ERROR && !presolve
		return cheby_lp(A, b, Aᵢ, bᵢ, unique_nonzerow_indices, presolve=true)
	else
		@show termination_status(model)
		@show A; @show b; @show Aᵢ; @show bᵢ
		@show unique_nonzerow_indices
		println("Dual infeasible => primal unbounded.")
		error("Chebyshev center error!")
	end
end

















### MAIN ALGORITHM ###
# Given input point and weights return state2input, state2output, state2map, plt_in, plt_out
# set reach=false for just cell enumeration
# Supports looking for multiple backward reachable sets at once
function compute_reach(weights, Aᵢ::Matrix{Float64}, bᵢ::Vector{Float64}, Aₒ::Vector{Matrix{Float64}}, bₒ::Vector{Vector{Float64}}; reach=false, back=false, verification=false)
	# Construct necessary data structures #
	state2input    = Dict{Vector{BitVector}, Tuple{Matrix{Float64},Vector{Float64}} }() # Dict from state -> (A,b) input constraints
	state2output   = Dict{Vector{BitVector}, Tuple{Matrix{Float64},Vector{Float64}} }() # Dict from state -> (A′,b′) ouput constraints
	state2backward = [Dict{Vector{BitVector}, Tuple{Matrix{Float64},Vector{Float64}}}() for _ in 1:length(Aₒ)]
	state2map      = Dict{Vector{BitVector}, Tuple{Matrix{Float64},Vector{Float64}} }() # Dict from state -> (C,d) local affine map
	state2essential = Dict{Vector{BitVector}, Vector{Int64}}() # Dict from state to neuron indices we know are essential
	working_set = Set{Vector{BitVector}}() # Network states we want to explore

	# Initialize algorithm #
	input = get_input(Aᵢ, bᵢ)
	state = get_state(input, weights)
	state2essential[state] = Vector{Int64}()
	push!(working_set, state)
	num_neurons = sum([length(state[layer]) for layer in 1:length(state)])
	
	# Begin cell enumeration #
	i, saved_lps, solved_lps, rank_deficient = (1, 0, 0, 0)
	while !isempty(working_set)
		println(i)
		state = pop!(working_set)

		# Get local affine_map
		C, d = local_map(state, weights)
		rank(C) != length(d) ? rank_deficient += 1 : nothing
		state2map[state] = (C,d)

		A, b, idx2repeat, zerows, unique_nonzerow_indices = get_constraints(weights, state, num_neurons)

		# We can check this before removing redundant constraints
		if verification
			for k in 1:length(Aₒ)
				Aᵤ, bᵤ = (Aₒ[k]*C, bₒ[k]-Aₒ[k]*d) # for Aₒy ≤ bₒ and y = Cx+d -> AₒCx ≤ bₒ-Aₒd
				if poly_intersection(A, b, Aᵤ, bᵤ)
					println("Found input that maps to target set!")
					@show state
					return state2input, state2output, state2map, state2backward
				end
			end
		end
		
		# Uncomment these lines to double check generated state is correct
		# center, essential, essentialᵢ = cheby_lp(A, b, Aᵢ, bᵢ, unique_nonzerow_indices) # Chebyshev center
		# check_state(center, weights, state)

		A, b, neighbor_indices, saved_lps_i, solved_lps_i = remove_redundant(A, b, Aᵢ, bᵢ, unique_nonzerow_indices, state2essential[state])
		working_set, state2essential = add_neighbor_states(state, neighbor_indices, working_set, idx2repeat, zerows, weights, state2essential)
		state2input[state] = (A,b)

		reach ? state2output[state] = affine_map(A, b, C, d) : nothing
		if back
			for k in 1:length(Aₒ)
				Aᵤ, bᵤ = (Aₒ[k]*C, bₒ[k]-Aₒ[k]*d) # for Aₒy ≤ bₒ and y = Cx+d -> AₒCx ≤ bₒ-Aₒd
				if poly_intersection(A, b, Aᵤ, bᵤ)
					state2backward[k][state] = (vcat(A, Aᵤ), vcat(b, bᵤ)) # not a fully reduced representation
				end
			end
		end
		
		i += 1;	saved_lps += saved_lps_i; solved_lps += solved_lps_i
	end
	verification ? println("No input maps to the target set.") : nothing
	println("Rank deficient maps: ", rank_deficient)
	total_lps = saved_lps + solved_lps
	println("Total solved LPs: ", solved_lps)
	println("Total saved LPs:  ", saved_lps, "/", total_lps, " : ", round(100*saved_lps/total_lps, digits=1), "% pruned." )
	return state2input, state2output, state2map, state2backward
end

