<<<<<<< Updated upstream
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

using LinearAlgebra, LazySets, Polyhedra, AlphaStructures, FileIO, Plots
include("merge_poly.jl")
# pyplot()


# load point set
perim_dict = load(string("models/taxinet/point_set_", steps, "_", length(Xs[1]), ".jld2"))
Xs = perim_dict["perim_large"]


# find nonconvex triangulation of inverse point set
α = 0.1

# Input array has vertices as columns
filtration = AlphaStructures.alphaFilter(hcat(Xs...))

# FV gives tuples of 3 vertices defining my simplices
VV,EV,FV = AlphaStructures.alphaSimplex(hcat(Vrep.vertices...),filtration, Float64(α))

# Get set of simplices in Vrep
simplices = Vector{Matrix{Float64}}(undef, length(FV))
for i in 1:length(simplices)
	simplices[i] = hcat(Xs[FV[i][1]], Xs[FV[i][2]], Xs[FV[i][3]])
end

plt2 = plot(reuse=false)
scatter!(plt2, hcat(Xs...)'[:,1], hcat(Xs...)'[:,2], label=string(steps, "-step"))
for s in simplices
	plot!(plt2, VPolytope(s), label=false)
end



# # turn simplices into hrep
# simplices_h = Set{ Tuple{Matrix{Float64}, Vector{Float64}} }()
# for s in simplices
# 	h = tohrep(VPolytope(s))
# 	n_cons = length(h.constraints)
# 	A = Matrix{Float64}(undef, n_cons, size(s,1))
# 	b = Vector{Float64}(undef, n_cons)
# 	for i in 1:n_cons
# 		A[i,:] = h.constraints[i].a
# 		b[i] = h.constraints[i].b
# 	end
# 	push!(simplices_h, (A, b))
# end

# # optimally merge simplices into larger overlapping polytopes
# polytopes = merge_polytopes(simplices_h, verbose=true)

# plt3 = plot(reuse=false)
# for (A,b) in polytopes
# 	plot!(plt3, HPolytope(A,b), label=false)
# end


## TODO ##
# find good initial sampling
# compute 227 step inverse and save points
# find good alpha level



## NEW FILE ##
# define PWA function for each polytope domain
# perform forward reach on each polytope domain
=======
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

using LinearAlgebra, LazySets, Polyhedra, AlphaStructures, FileIO, Plots
include("merge_poly.jl")
# pyplot()


# load point set
perim_dict = load(string("models/taxinet/point_set_", steps, "_", length(Xs[1]), ".jld2"))
Xs = perim_dict["perim_large"]


# find nonconvex triangulation of inverse point set
α = 0.1

# Input array has vertices as columns
filtration = AlphaStructures.alphaFilter(hcat(Xs...))

# FV gives tuples of 3 vertices defining my simplices
VV,EV,FV = AlphaStructures.alphaSimplex(hcat(Vrep.vertices...),filtration, Float64(α))

# Get set of simplices in Vrep
simplices = Vector{Matrix{Float64}}(undef, length(FV))
for i in 1:length(simplices)
	simplices[i] = hcat(Xs[FV[i][1]], Xs[FV[i][2]], Xs[FV[i][3]])
end

plt2 = plot(reuse=false)
scatter!(plt2, hcat(Xs...)'[:,1], hcat(Xs...)'[:,2], label=string(steps, "-step"))
for s in simplices
	plot!(plt2, VPolytope(s), label=false)
end



# # turn simplices into hrep
# simplices_h = Set{ Tuple{Matrix{Float64}, Vector{Float64}} }()
# for s in simplices
# 	h = tohrep(VPolytope(s))
# 	n_cons = length(h.constraints)
# 	A = Matrix{Float64}(undef, n_cons, size(s,1))
# 	b = Vector{Float64}(undef, n_cons)
# 	for i in 1:n_cons
# 		A[i,:] = h.constraints[i].a
# 		b[i] = h.constraints[i].b
# 	end
# 	push!(simplices_h, (A, b))
# end

# # optimally merge simplices into larger overlapping polytopes
# polytopes = merge_polytopes(simplices_h, verbose=true)

# plt3 = plot(reuse=false)
# for (A,b) in polytopes
# 	plot!(plt3, HPolytope(A,b), label=false)
# end


## TODO ##
# find good initial sampling
# compute 227 step inverse and save points
# find good alpha level



## NEW FILE ##
# define PWA function for each polytope domain
# perform forward reach on each polytope domain
>>>>>>> Stashed changes
# plot reachable set and visually confirm it's a subset of starting set