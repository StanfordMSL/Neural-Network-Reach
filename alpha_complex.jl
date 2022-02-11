using LinearAlgebra, LazySets, Polyhedra, AlphaStructures, FileIO, Plots
include("merge_poly.jl")
include("inverse_points.jl")
include("load_networks.jl")
# pyplot()


# load point set
perim_dict = load(string("models/taxinet/point_set_", 227, "_", 54, ".jld2"))
Xs = perim_dict["perim_large"]


# find nonconvex triangulation of inverse point set
α = 0.25

# Input array has vertices as columns
filtration = AlphaStructures.alphaFilter(hcat(Xs...))

# FV gives tuples of 3 vertices defining my simplices
VV,EV,FV = AlphaStructures.alphaSimplex(hcat(Xs...),filtration, Float64(α))

# Get set of simplices in Vrep
simplices = Vector{Matrix{Float64}}(undef, length(FV))
for i in 1:length(simplices)
	simplices[i] = hcat(Xs[FV[i][1]], Xs[FV[i][2]], Xs[FV[i][3]])
end

plt2 = plot(reuse=false)
scatter!(plt2, hcat(Xs...)'[:,1], hcat(Xs...)'[:,2])
for s in simplices
	plot!(plt2, VPolytope(s), label=false)
end




# turn simplices into hrep
simplices_h = Set{ Tuple{Matrix{Float64}, Vector{Float64}} }()
for s in simplices
	h = tohrep(VPolytope(s))
	n_cons = length(h.constraints)
	A = Matrix{Float64}(undef, n_cons, size(s,1))
	b = Vector{Float64}(undef, n_cons)
	for i in 1:n_cons
		A[i,:] = h.constraints[i].a
		b[i] = h.constraints[i].b
	end
	push!(simplices_h, (A, b))
end

# optimally merge simplices into larger overlapping polytopes
polytopes = merge_polytopes(simplices_h, verbose=true)

plt3 = plot()
for (A,b) in polytopes
	plot!(plt3, HPolytope(A,b), label=false)
end



# sample boundary of polytope union and visually check that it is an invariant set for the sampled points
samples = Vector{Vector{Float64}}(undef,0)
n = 10_000
for (A,b) in polytopes
	append!(samples, sample_perimeter(A, b, collect(vertices(HPolytope(A,b))), n, tol = 1e-3))
end

scatter!(plt3, hcat(samples...)'[:,1], hcat(samples...)'[:,2], markersize=2, label="original")

# pass samples through dynamics
weights = taxinet_cl(1)
outputs = [eval_net(sample, weights, 1) for sample in samples]

scatter!(plt3, hcat(outputs...)'[:,1], hcat(outputs...)'[:,2], markersize=2, label="forward")

# save point set
# save(string("models/taxinet/point_set_", steps, "_", length(Xs[1]), ".jld2"), Dict("perim_small" => Xs[1], "perim_large" => Xs[2]))