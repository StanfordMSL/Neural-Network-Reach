using Plots, LinearAlgebra, JuMP, GLPK, LazySets, Polyhedra, CDDLib
include("load_networks.jl")
include("unique_custom.jl")

ϵ = 1e-10 # used for numerical tolerances throughout

### GENERAL PURPOSE FUNCTIONS ###
function normalize_row_old(row::Vector{Float64}; return_zero=false)
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

function normalize_row(row::Vector{Float64})
	scale = norm(row[1:end-1])
	size = length(row)-1
	if scale > ϵ
		return row / scale
	else
		return vcat(zeros(size), [row[end]])
	end
end

# Returns the algorithm initialization point given task and input space constraints
function get_input(Aᵢ, bᵢ, weights)
	input, nothing, nothing = cheby_lp([], [], Aᵢ, bᵢ, [])
	return input + 0.0001*randn(length(input))
	# i = 1
	# while !interior_input(input, weights)
	# 	input += 1e-6*randn(size(Aᵢ,2))
	# 	if i == 100
	# 		error("Couldn't generate interior input.")
	# 	end
	# 	i += 1
	# end
	# return input
end

# checks whether input is on cell boundary
# an input is on a cell boundary if any postactivation variable z is zero
function interior_input(input, weights)
	NN_out = vcat(input, [1.])
    for layer = 1:length(weights)-1
    	ẑ = weights[layer]*NN_out
    	for i in 1:length(ẑ)
    		isapprox(ẑ[i], 0.0, atol=ϵ) ? (return false) : nothing
    	end
        NN_out = max.(0, ẑ)
    end
    return true
end


# Given input point, perform forward pass to get ap.
function get_ap(input, weights)
	L = length(weights)
	ap = Vector{BitVector}(undef, L-1)
	layer_vals = vcat(input, [1])
	for layer in 1:L-1 # No ReLU on last layer
		layer_vals = max.(0, weights[layer]*layer_vals)
		ap[layer] = layer_vals .> ϵ
	end
	return ap
end

# Given constraint index, return associated layer and neuron
function get_layer_neuron(index, ap)
	layer, neuron, prev = (1, 1, 0)
	for l in 1:length(ap)
		if index > length(ap[l]) + prev
			prev += length(ap[l])
		else
			return l, (index - prev)
		end
	end
end

# Get hyperplane equation associated with a given neuron and input
function neuron_map(layer, neuron, ap, weights; normalize=true)
	matrix = I 
	for l in 1:layer-1
		matrix = diagm(0 => ap[l])*weights[l]*matrix
	end
	matrix = weights[layer]*matrix
	if normalize
		return normalize_row(matrix[neuron,:])
	else
		return matrix[neuron,:]
	end
end

# Get affine map associated with an ap
function local_map(ap::Vector{BitVector}, weights::Vector{Matrix{Float64}})
	matrix = I 
	for l in 1:length(weights)-1
		matrix = diagm(0 => ap[l])*weights[l]*matrix
	end
	matrix = weights[end]*matrix
	
	return matrix[:,1:end-1], matrix[:,end] # C,d of y = Cx+d
end















### FUNCTIONS FOR GETTING CONSTRAINTS ###
function positive_zeros(A)
    k = 1
    @inbounds for t in eachindex(A)
        if isequal(A[t], -0.0)
            A[t] = 0.0
            k += 1
        end
    end
    return A
end

# Get redundant A≤b constraints
function get_constraints(weights::Vector{Matrix{Float64}}, ap::Vector{BitVector}, num_neurons)
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
		for neuron in 1:length(ap[layer])
			A[i,:] = (1-2*ap[layer][neuron])*output[neuron,:]
			if !isapprox(A[i,1:end-1], zeros(size(A,2)-1), atol=ϵ) # check nonzero.
				A[i,:] = normalize_row(A[i,:])
			else
				push!(zerows, i)
			end
			i += 1
		end
		lin_map = diagm(0 => ap[layer])*weights[layer]*lin_map
	end

	A = positive_zeros(A)
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

	if bᵢ != []
		return vcat(A[essential,:], Aᵢ[essentialᵢ,:]), vcat(b[essential], bᵢ[essentialᵢ]), sort(essential), saved_lps, solved_lps
	else
		return A[essential,:], b[essential], sort(essential), saved_lps, solved_lps
	end

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
	bᵢ != [] ? @constraint(model, Aᵢ*x .<= bᵢ) : nothing
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
	if bᵢ != []
		for i in length(bᵢ)
			val = sum([Aᵢ[i,j] > 0 ? Aᵢ[i,j]*bounds[j,2] : Aᵢ[i,j]*bounds[j,1] for j in 1:size(Aᵢ,2)])
			val + ϵ < bᵢ[i]  ? push!(redundantᵢ,i) : nothing
		end
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
	bᵢ != [] ? @constraint(model, conₛ[j in non_redundantᵢ], dot(Aᵢ[j,:],x) <= bᵢ[j]) : nothing # tack on non-redundant superset constraints

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

	if bᵢ != []
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
	end
	return essential, essentialᵢ
end
















### FUNCTIONS FOR FINDING NEIGHBORS ###
# Adds neighbor aps to working_set
function add_neighbor_aps(ap::Vector{BitVector}, neighbor_indices::Vector{Int64}, working_set, idx2repeat::Dict{Int64,Vector{Int64}}, zerows::Vector{Int64}, weights::Vector{Matrix{Float64}}, ap2essential)	
	for idx in neighbor_indices
		neighbor_ap = deepcopy(ap)
		l, n = get_layer_neuron(idx, neighbor_ap)
		neighbor_constraint = -(1-2*neighbor_ap[l][n])*normalize_row(neuron_map(l, n, neighbor_ap, weights)) # constraint for c′

		type1 = idx2repeat[idx]
		type2 = zerows
		neighbor_ap = flip_neurons!(type1, type2, neighbor_ap, weights, neighbor_constraint)

		if !haskey(ap2essential, neighbor_ap) && neighbor_ap ∉ working_set
			push!(working_set, neighbor_ap)
			ap2essential[neighbor_ap] = idx2repeat[idx] # All of the neurons that define the neighbor constraint
		end
	end 

	return working_set, ap2essential
end


# Handles flipping of activation pattern from c to c′
function flip_neurons!(type1, type2, neighbor_ap, weights, neighbor_constraint)
	# neuron_idx = what number neuron with top neuron first layer = 1 and bottom neuron last layer = end
	a, b = neighbor_constraint[1:end-1], -neighbor_constraint[end] # a⋅x ≤ b for c′

	for neuron_idx in sort(vcat(type1, type2))
		l, n = get_layer_neuron(neuron_idx, neighbor_ap)
		new_map = (1-2*neighbor_ap[l][n])*normalize_row(neuron_map(l, n, neighbor_ap, weights))
		a′, b′ = new_map[1:end-1], -new_map[end] 
		
		# now we check whether a′⋅x ≤ b′ is valid, or if we need to flip the activation such that a′⋅x ≥ b′
		if isapprox(a′, zeros(length(a′)), atol=ϵ )
			if b′ ≥ 0 # ⟹ 0⋅x ≤ b′ is then always satisfied, thus valid
				nothing
			else # 0⋅x ≤ b′ is then never satisfied, thus invalid
				neighbor_ap[l][n] = !neighbor_ap[l][n]
			end 
		# elseif neuron_idx ∈ type1
		# 	neighbor_ap[l][n] = !neighbor_ap[l][n]
		elseif isapprox(a′, a, atol=ϵ ) && b′ ≥ b 
			nothing
		elseif isapprox(a′, a, atol=ϵ ) && b′ < b
			neighbor_ap[l][n] = !neighbor_ap[l][n]
		elseif isapprox(-a′, a, atol=ϵ ) && -b′ < b
			nothing
		elseif isapprox(-a′, a, atol=ϵ ) && -b′ ≥ b
			neighbor_ap[l][n] = !neighbor_ap[l][n]
		else
			error("Check neuron flipping rules.")
		end
	end

	return neighbor_ap
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
	A′ = vcat( hcat(I, -C), hcat(-I, C), hcat(zeros(length(b), ydim), A) )
	b′ = vcat(d, -d, b)

	poly_in = polyhedron(hrep(A′,b′), CDDLib.Library(:float))
	poly_out = eliminate(poly_in, collect(ydim+1:ydim+xdim), FourierMotzkin())
	ine = unsafe_load(poly_out.ine.matrix)

	numrow = ine.rowsize
	numcol = ine.colsize
	Aₒ = Matrix{Float64}(undef, numrow, numcol-1)
	bₒ = Vector{Float64}(undef, numrow)
	good_idxs = []
	for i in 1:numrow
		row = unsafe_load(ine.matrix, i)
		bₒ[i] = unsafe_load(row, 1)
		for j in 1:numcol-1
			Aₒ[i,j] = -unsafe_load(row, j+1)
		end
		Aₒ[i,:] != zeros(numcol-1) ? push!(good_idxs, i) : nothing
	end

	return Aₒ[good_idxs,:], bₒ[good_idxs]
end
















### FUNCTIONS TO VERIFY ACTIVATION PATTERN ###
# Make sure we generate the correct ap for a region
function check_ap(input, weights, ap)
	if ap != get_ap(input, weights) 
		@show input;  @show ap;  @show get_ap(input, weights)
		error("NN ap not what it seems!")
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
		if value.(r) == 1e4
			println("Unbounded!")
		end
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
# Given input point and weights return ap2input, ap2output, ap2map, plt_in, plt_out
# set reach=false for just cell enumeration
# Supports looking for multiple backward reachable sets at once
function compute_reach(weights, Aᵢ::Matrix{Float64}, bᵢ::Vector{Float64}, Aₒ::Vector{Matrix{Float64}}, bₒ::Vector{Vector{Float64}}; fp = [], reach=false, back=false, verification=false, connected=false)
	# Construct necessary data structures #
	ap2input    = Dict{Vector{BitVector}, Tuple{Matrix{Float64},Vector{Float64}} }() # Dict from ap -> (A,b) input constraints
	ap2output   = Dict{Vector{BitVector}, Tuple{Matrix{Float64},Vector{Float64}} }() # Dict from ap -> (A′,b′) ouput constraints
	ap2backward = [Dict{Vector{BitVector}, Tuple{Matrix{Float64},Vector{Float64}}}() for _ in 1:length(Aₒ)]
	ap2map      = Dict{Vector{BitVector}, Tuple{Matrix{Float64},Vector{Float64}} }() # Dict from ap -> (C,d) local affine map
	ap2essential = Dict{Vector{BitVector}, Vector{Int64}}() # Dict from ap to neuron indices we know are essential
	working_set = Set{Vector{BitVector}}() # Network aps we want to explore
	# Initialize algorithm #
	fp == [] ? input = get_input(Aᵢ, bᵢ, weights) : input = fp # this may fail if initialized on the boundary of a cell
	# check whether input is in interior of cell. If so, find a new input.
	# input = 0.0001*randn(2) # need to test that input is not on cell boundary
	# input = [-1.0899274931805163, -0.12567904584271794]
	ap = get_ap(input, weights)
	ap2essential[ap] = Vector{Int64}()
	push!(working_set, ap)
	num_neurons = sum([length(ap[layer]) for layer in 1:length(ap)])
	
	# Begin cell enumeration #
	i, saved_lps, solved_lps, rank_deficient = (1, 0, 0, 0)
	while !isempty(working_set)
		println(i)
		ap = pop!(working_set)

		# Get local affine_map
		C, d = local_map(ap, weights)
		rank(C) != length(d) ? rank_deficient += 1 : nothing
		ap2map[ap] = (C,d)

		A, b, idx2repeat, zerows, unique_nonzerow_indices = get_constraints(weights, ap, num_neurons)

		# We can check this before removing redundant constraints
		if verification
			for k in 1:length(Aₒ)
				Aᵤ, bᵤ = (Aₒ[k]*C, bₒ[k]-Aₒ[k]*d) # for Aₒy ≤ bₒ and y = Cx+d -> AₒCx ≤ bₒ-Aₒd
				if poly_intersection(A, b, Aᵤ, bᵤ)
					println("Found input that maps to target set!")
					@show ap
					return ap2input, ap2output, ap2map, ap2backward
				end
			end
		end
		
		# Uncomment these lines to double check generated ap is correct
		center, essential, essentialᵢ = cheby_lp(A, b, Aᵢ, bᵢ, unique_nonzerow_indices) # Chebyshev center
		check_ap(center, weights, ap)

		A, b, neighbor_indices, saved_lps_i, solved_lps_i = remove_redundant(A, b, Aᵢ, bᵢ, unique_nonzerow_indices, ap2essential[ap])


		reach ? ap2output[ap] = affine_map(A, b, C, d) : nothing
		if back && connected # only add neighbors of cells that are in the BRS
			for k in 1:length(Aₒ)
				Aᵤ, bᵤ = (Aₒ[k]*C, bₒ[k]-Aₒ[k]*d) # for Aₒy ≤ bₒ and y = Cx+d -> AₒCx ≤ bₒ-Aₒd
				if poly_intersection(A, b, Aᵤ, bᵤ)
					ap2backward[k][ap] = (vcat(A, Aᵤ), vcat(b, bᵤ)) # not a fully reduced representation
					working_set, ap2essential = add_neighbor_aps(ap, neighbor_indices, working_set, idx2repeat, zerows, weights, ap2essential)
				end
			end
		elseif back # add neighbors of all cells
			for k in 1:length(Aₒ)
				Aᵤ, bᵤ = (Aₒ[k]*C, bₒ[k]-Aₒ[k]*d) # for Aₒy ≤ bₒ and y = Cx+d -> AₒCx ≤ bₒ-Aₒd
				if poly_intersection(A, b, Aᵤ, bᵤ)
					ap2backward[k][ap] = (vcat(A, Aᵤ), vcat(b, bᵤ)) # not a fully reduced representation
				end
			end
			working_set, ap2essential = add_neighbor_aps(ap, neighbor_indices, working_set, idx2repeat, zerows, weights, ap2essential)
		else
			# add neighbors of all cells
			working_set, ap2essential = add_neighbor_aps(ap, neighbor_indices, working_set, idx2repeat, zerows, weights, ap2essential)
		end
		# ap2input[ap] = (vcat(A, Aᵢ), vcat(b, bᵢ))
		ap2input[ap] = (A, b)
		
		i += 1;	saved_lps += saved_lps_i; solved_lps += solved_lps_i
		# if i == 20
		# 	break
		# end
	end
	verification ? println("No input maps to the target set.") : nothing
	println("Rank deficient maps: ", rank_deficient)
	total_lps = saved_lps + solved_lps
	println("Total solved LPs: ", solved_lps)
	println("Total saved LPs:  ", saved_lps, "/", total_lps, " : ", round(100*saved_lps/total_lps, digits=1), "% pruned." )
	return ap2input, ap2output, ap2map, ap2backward
end
