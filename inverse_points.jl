<<<<<<< Updated upstream
using LinearAlgebra, JuMP, GLPK, LazySets, Polyhedra, FileIO, Plots, Distributions
include("load_networks.jl")
# pyplot()

# generate uniform random real vector on the unit sphere
rand_sphere(n) = normalize(randn(n))

#= 
Given polytope Ax≤b, associated vertices, n attempts, and tolerance tol, generate
random points on the boundary of the polytope such that no 
point is closer than tol to another point.
Random boundary points are generated using ray shooting.
=#
function sample_perimeter(A, b, vertices, n; tol=0.0)
	dim = size(A,2)
	perimeter = vertices

	# shoot random ray from center and find nearest halfspace intersection
	perim_point = zeros(dim)
	for i in 1:n
		# random center point as random interior point
		weights = rand(Dirichlet(length(vertices), 1.)) # convex combination
		c = sum(weights[i]*vertices[i] for i in 1:length(vertices))
		d = rand_sphere(dim)
		t_min = Inf

		!in_polytope(c, A, b) ? error("center not in polytope!") : nothing
		
		# solve for closest ray intersection
		for i in 1:length(b)
			t = (b[i] - A[i,:]⋅c) / (A[i,:]⋅d)
			if t > 0 && t < t_min
				t_min = t
				perim_point = c + t_min*d
			end
		end

		# check that new point is not within tol of existing points
		below_tol = false
		for j in 1:length(perimeter)
			if norm(perimeter[j] - perim_point) ≤ tol
				below_tol = true
				break
			end
		end

		below_tol ? nothing : push!(perimeter, perim_point)
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
	num_cells = length(ap2map)

	# for each region, check for inputs matching given outputs
	for (i,ap) in enumerate(keys(ap2map))
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
		append!(found_indices, new_found_indices)

		# println(i, "/", num_cells, " regions.")
		# println(length(found_indices), "/", length(Ys), " points.")

		# once all outputs accounted for, return inputs
		if isempty(working_indices)
			return Xs
		end
	end

	println("The following output indices not accounted for!")
	@show working_indices
	return Xs
end


function in_polytope(x, A, b)
	for i in 1:length(b)
		if A[i,:]⋅x > b[i]
			return false
		end
	end
	return true
end






# load in homeomorphic PWA function (21,606 regions)
pwa_dict = load("models/taxinet/taxinet_pwa_map_large.jld2")
ap2map = pwa_dict["ap2map"]
ap2input = pwa_dict["ap2input"]
ap2neighbors = pwa_dict["ap2neighbors"]
weights = taxinet_cl(1)
# eval_net(input, weights, steps)

# define seed ROA #
fp = [-1.089927713157323, -0.12567755953751042]
A_roa = 370*[-0.20814568962857855 0.03271955855771795; 0.2183098663000297 0.12073669880754853; 0.42582101825227686 0.0789033995762251; 0.14480530852927057 -0.05205811047518554; -0.13634673812819695 -0.1155315084750385; 0.04492020060602461 0.09045775648816877; -0.6124506220873154 -0.12811621541510643]
b_roa = ones(size(A_roa,1)) + A_roa*fp
Hrep = HPolytope(constraints_list(A_roa, b_roa))


# sample space around seed ROA #
N = 1_000
Xs = Vector{Vector{Vector{Float64}}}(undef, 2)
Xs[1] = [[bound_r(-2,0), bound_r(-0.25,0.0)] for _ in 1:N]
Xs[2] = Xs[1]


# compute inverse points
steps = 24 # 227 is max for taxinet_pwa_map_large.jld2
for i in 2:steps+1
	println("Inverse step: ", i-1)
	Xs[2] = inverse_map(ap2map, ap2input, Xs[2])
end


# split into points originating from roa and those not #


# plot starting and ending points
plt1 = plot(Hrep, reuse=false, label=false)
scatter!(plt1, hcat(Xs[1]...)'[:,1], hcat(Xs[1]...)'[:,2], label=false)
scatter!(plt1, hcat(Xs[2]...)'[:,1], hcat(Xs[2]...)'[:,2], label=string(steps, "-step"))




# # collect perimeter points of seed ROA
# fp = [-1.089927713157323, -0.12567755953751042]
# A_roa = 370*[-0.20814568962857855 0.03271955855771795; 0.2183098663000297 0.12073669880754853; 0.42582101825227686 0.0789033995762251; 0.14480530852927057 -0.05205811047518554; -0.13634673812819695 -0.1155315084750385; 0.04492020060602461 0.09045775648816877; -0.6124506220873154 -0.12811621541510643]
# b_roa = ones(size(A_roa,1)) + A_roa*fp

# Hrep = HPolytope(constraints_list(A_roa, b_roa))
# Vrep = tovrep(Hrep)

# n = 100_000
# Xs = Vector{Vector{Vector{Float64}}}(undef, 2)
# Xs[1] = sample_perimeter(A_roa, b_roa, Vrep.vertices, n, tol = 1e-3)
# Xs[2] = Xs[1]
# println(length(Xs[1]), " perimeter points")


# # compute inverse points
# steps = 227 # 227 is max for taxinet_pwa_map_large.jld2
# for i in 2:steps+1
# 	println("Inverse step: ", i-1)
# 	Xs[2] = inverse_map(ap2map, ap2input, Xs[2])
# end

# # plot starting and ending points
# plt1 = plot(Hrep, reuse=false, label=false)
# scatter!(plt1, hcat(Xs[1]...)'[:,1], hcat(Xs[1]...)'[:,2], label=false)
# scatter!(plt1, hcat(Xs[2]...)'[:,1], hcat(Xs[2]...)'[:,2], label=string(steps, "-step"))

# save point set
# save(string("models/taxinet/point_set_", steps, "_", length(Xs[1]), ".jld2"), Dict("perim_small" => Xs[1], "perim_large" => Xs[2]))

=======
using LinearAlgebra, JuMP, GLPK, LazySets, Polyhedra, FileIO, Plots, Distributions
include("load_networks.jl")
# pyplot()

# generate uniform random real vector on the unit sphere
rand_sphere(n) = normalize(randn(n))

#= 
Given polytope Ax≤b, associated vertices, n attempts, and tolerance tol, generate
random points on the boundary of the polytope such that no 
point is closer than tol to another point.
Random boundary points are generated using ray shooting.
=#
function sample_perimeter(A, b, vertices, n; tol=0.0)
	dim = size(A,2)
	perimeter = vertices

	# shoot random ray from center and find nearest halfspace intersection
	perim_point = zeros(dim)
	for i in 1:n
		# random center point as random interior point
		weights = rand(Dirichlet(length(vertices), 1.)) # convex combination
		c = sum(weights[i]*vertices[i] for i in 1:length(vertices))
		d = rand_sphere(dim)
		t_min = Inf

		!in_polytope(c, A, b) ? error("center not in polytope!") : nothing
		
		# solve for closest ray intersection
		for i in 1:length(b)
			t = (b[i] - A[i,:]⋅c) / (A[i,:]⋅d)
			if t > 0 && t < t_min
				t_min = t
				perim_point = c + t_min*d
			end
		end

		# check that new point is not within tol of existing points
		below_tol = false
		for j in 1:length(perimeter)
			if norm(perimeter[j] - perim_point) ≤ tol
				below_tol = true
				break
			end
		end

		below_tol ? nothing : push!(perimeter, perim_point)
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
	num_cells = length(ap2map)

	# for each region, check for inputs matching given outputs
	for (i,ap) in enumerate(keys(ap2map))
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
		append!(found_indices, new_found_indices)

		# println(i, "/", num_cells, " regions.")
		# println(length(found_indices), "/", length(Ys), " points.")

		# once all outputs accounted for, return inputs
		if isempty(working_indices)
			return Xs
		end
	end

	println("The following output indices not accounted for!")
	@show working_indices
	return Xs
end


function in_polytope(x, A, b)
	for i in 1:length(b)
		if A[i,:]⋅x > b[i]
			return false
		end
	end
	return true
end






# load in homeomorphic PWA function (21,606 regions)
pwa_dict = load("models/taxinet/taxinet_pwa_map_large.jld2")
ap2map = pwa_dict["ap2map"]
ap2input = pwa_dict["ap2input"]
ap2neighbors = pwa_dict["ap2neighbors"]
weights = taxinet_cl(1)
# eval_net(input, weights, steps)

# define seed ROA #
fp = [-1.089927713157323, -0.12567755953751042]
A_roa = 370*[-0.20814568962857855 0.03271955855771795; 0.2183098663000297 0.12073669880754853; 0.42582101825227686 0.0789033995762251; 0.14480530852927057 -0.05205811047518554; -0.13634673812819695 -0.1155315084750385; 0.04492020060602461 0.09045775648816877; -0.6124506220873154 -0.12811621541510643]
b_roa = ones(size(A_roa,1)) + A_roa*fp
Hrep = HPolytope(constraints_list(A_roa, b_roa))


# sample space around seed ROA #
N = 1_000
Xs = Vector{Vector{Vector{Float64}}}(undef, 2)
Xs[1] = [[bound_r(-2,0), bound_r(-0.25,0.0)] for _ in 1:N]
Xs[2] = Xs[1]


# compute inverse points
steps = 24 # 227 is max for taxinet_pwa_map_large.jld2
for i in 2:steps+1
	println("Inverse step: ", i-1)
	Xs[2] = inverse_map(ap2map, ap2input, Xs[2])
end


# split into points originating from roa and those not #


# plot starting and ending points
plt1 = plot(Hrep, reuse=false, label=false)
scatter!(plt1, hcat(Xs[1]...)'[:,1], hcat(Xs[1]...)'[:,2], label=false)
scatter!(plt1, hcat(Xs[2]...)'[:,1], hcat(Xs[2]...)'[:,2], label=string(steps, "-step"))




# # collect perimeter points of seed ROA
# fp = [-1.089927713157323, -0.12567755953751042]
# A_roa = 370*[-0.20814568962857855 0.03271955855771795; 0.2183098663000297 0.12073669880754853; 0.42582101825227686 0.0789033995762251; 0.14480530852927057 -0.05205811047518554; -0.13634673812819695 -0.1155315084750385; 0.04492020060602461 0.09045775648816877; -0.6124506220873154 -0.12811621541510643]
# b_roa = ones(size(A_roa,1)) + A_roa*fp

# Hrep = HPolytope(constraints_list(A_roa, b_roa))
# Vrep = tovrep(Hrep)

# n = 100_000
# Xs = Vector{Vector{Vector{Float64}}}(undef, 2)
# Xs[1] = sample_perimeter(A_roa, b_roa, Vrep.vertices, n, tol = 1e-3)
# Xs[2] = Xs[1]
# println(length(Xs[1]), " perimeter points")


# # compute inverse points
# steps = 227 # 227 is max for taxinet_pwa_map_large.jld2
# for i in 2:steps+1
# 	println("Inverse step: ", i-1)
# 	Xs[2] = inverse_map(ap2map, ap2input, Xs[2])
# end

# # plot starting and ending points
# plt1 = plot(Hrep, reuse=false, label=false)
# scatter!(plt1, hcat(Xs[1]...)'[:,1], hcat(Xs[1]...)'[:,2], label=false)
# scatter!(plt1, hcat(Xs[2]...)'[:,1], hcat(Xs[2]...)'[:,2], label=string(steps, "-step"))

# save point set
# save(string("models/taxinet/point_set_", steps, "_", length(Xs[1]), ".jld2"), Dict("perim_small" => Xs[1], "perim_large" => Xs[2]))

>>>>>>> Stashed changes
