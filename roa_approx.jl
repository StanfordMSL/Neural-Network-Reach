#=
The goal of this file is to certify a small region of attraction (local asymptotic stability) exists for a 
given homeomorphic PWA function and to compute a large forward invariant set which is constructed in a 
principled way so as to likely also be an ROA. Though proving the computed invariant set is an ROA is
too hard to perform. 

Specifically, we sample the boundary of the seed ROA and, since the function is invertible, estimate the 
perimeter of the k-step ROA by passing each point through the inverse map.
We then construct an α-complex based on these points.
From this α-complex we can use the associated simplices as a polytopic partition of the approximate k-step ROA.
We can verify this set of polytopes is forward invariant by checking that the 1-step forward reachable set is 
a subset of the α-complex.


I use the following polytopic ROA as my seed set.
fp = [-1.089927713157323, -0.12567755953751042]
A_roa = 370*[-0.20814568962857855 0.03271955855771795; 0.2183098663000297 0.12073669880754853; 0.42582101825227686 0.0789033995762251; 0.14480530852927057 -0.05205811047518554; -0.13634673812819695 -0.1155315084750385; 0.04492020060602461 0.09045775648816877; -0.6124506220873154 -0.12811621541510643]
b_roa = ones(size(A_roa,1)) + A_roa*fp
=#

using LinearAlgebra, JuMP, GLPK, LazySets, FileIO, Plots

# generate uniform random real vector on the unit sphere
rand_sphere(n) = normalize(randn(n))

# find chebyshev center of polytope
function cheby_lp(A, b)
	dim = size(A,2)
	
	# Find chebyshev center
	model = Model(GLPK.Optimizer)
	@variable(model, r)
	@variable(model, x_c[1:dim])
	@objective(model, Max, r)

	for i in 1:length(b)
		@constraint(model, dot(A[i,:],x_c) + r*norm(A[i,:]) ≤ b[i])
	end

	@constraint(model,  r ≤ 1e4) # prevents unboundedness
	@constraint(model, r ≥ 1e-15) # Must have r>0
	optimize!(model)
	return value.(x_c)
end


# sample n perimeter points of polytope
function sample_perimeter(A, b, n)
	dim = size(A,2)
	perimeter = Vector{Vector{Float64}}(undef, n)
	center = cheby_lp(A, b)

	# shoot random ray from center and find nearest halfspace intersection
	perim_point = deepcopy(center)
	for i in 1:n
		d = normalize(rand_sphere(dim))
		t_min = Inf
		
		for i in 1:length(b)
			t = (b[i] - A[i,:]⋅center) / (A[i,:]⋅d)
			if t > 0 && t < t_min
				t_min = t
				perim_point = center + t_min*d
			end
		end
		perimeter[i] = perim_point
	end
	return perimeter
end


#=
find the input that leads to the given output under the PWA function
given y = Cx + d, find corresponding x = inv(C)*(y - d) and see if its in Ax≤b
 =#
function inverse_map(ap2map, ap2input, Ys)
	Xs = Vector{Vector{Float64}}(undef, length(Ys))
	working_indices = collect(1:length(Ys))
	found_indices = Vector{Int64}(undef, 0)

	# for each region, check for inputs matching given outputs
	for ap in keys(ap2map)
		C, d = ap2map[ap]
		A, b = ap2input[ap]

		new_found_indices = Vector{Int64}(undef, 0)
		for i in working_indices
			inverse = inv(C)
			if in_polytope(inverse*(Ys[i] - d), A, b)
				Xs[i] = inverse*(Ys[i] - d)
				push!(new_found_indices, i)
			end
		end
		setdiff!(working_indices, new_found_indices)
		push!(found_indices, new_found_indices)

		# once all outputs accounted for, return inputs
		if isempty(working_indices)
			return Xs
		end
	end

	error("Inverse map error!")
end


function in_polytope(x, A, b)
	for i in 1:length(b)
		if A[i,:]⋅x > b[i]
			return false
		end
	end
	return true
end







### Scripting ###
# load in homeomorphic PWA function (21,606 regions)
pwa_dict = load("models/taxinet/taxinet_pwa_map_large.jld2")
ap2map = pwa_dict["ap2map"]
ap2input = pwa_dict["ap2input"]
ap2neighbors = pwa_dict["ap2neighbors"]



# collect perimeter points of seed ROA
fp = [-1.089927713157323, -0.12567755953751042]
A_roa = 370*[-0.20814568962857855 0.03271955855771795; 0.2183098663000297 0.12073669880754853; 0.42582101825227686 0.0789033995762251; 0.14480530852927057 -0.05205811047518554; -0.13634673812819695 -0.1155315084750385; 0.04492020060602461 0.09045775648816877; -0.6124506220873154 -0.12811621541510643]
b_roa = ones(size(A_roa,1)) + A_roa*fp

n = 500
Hrep = HPolytope(constraints_list(A_roa, b_roa))
perimeter_0 = sample_perimeter(A_roa, b_roa, n)
perimeter_matrix = hcat(perimeter_0...)'

plt = plot(Hrep, label=false)
scatter!(plt, perimeter_matrix[:,1], perimeter_matrix[:,2], label=false)



