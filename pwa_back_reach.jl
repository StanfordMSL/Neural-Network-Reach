using LinearAlgebra, JuMP, GLPK, FileIO, Plots, LazySets

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


function plot_brs(brs)
	plt = plot(reuse = false, legend=false, xlabel="x₁", ylabel="x₂")
	for (A, b) in brs
		reg = HPolytope(constraints_list(A, b))
		if isempty(reg)
			@show reg
			error("Empty polyhedron.")
		end
		plot!(plt,  reg, fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
	end
	return plt
end


#=
4-step brs is largest seed roa I found
=#



brs_dict = load(string("models/taxinet/BRS/taxinet_brs_", 20, "_step.jld2"))
out_set = brs_dict["brs"]
plt = plot_brs(out_set)

#=
# select step i
for i in 51:100
	# i = 10
	println("Computing ", i, "-step BRS.")



	# load in PWA function
	pwa_dict = load("models/taxinet/taxinet_pwa_map.jld2")
	ap2map = pwa_dict["ap2map"]
	ap2input = pwa_dict["ap2input"]
	ap2neighbors = pwa_dict["ap2neighbors"]


	# define the fixed point
	fp = [-1.089927713157323, -0.12567755953751042]


	# load in set to find preimage of
	# 0-step brs
	# A_roa = 370*[-0.20814568962857855 0.03271955855771795; 0.2183098663000297 0.12073669880754853; 0.42582101825227686 0.0789033995762251; 0.14480530852927057 -0.05205811047518554; -0.13634673812819695 -0.1155315084750385; 0.04492020060602461 0.09045775648816877; -0.6124506220873154 -0.12811621541510643]
	# b_roa = ones(size(A_roa,1)) + A_roa*fp
	# out_set = Set{Tuple{Matrix{Float64},Vector{Float64}}}([(A_roa, b_roa)])

	# i-step brs
	brs_dict = load(string("models/taxinet/BRS/taxinet_brs_", i-1, "_step.jld2"))
	out_set = brs_dict["brs"]


	# find ap for cell that fp lives in
	ap = pwa_dict["ap_fp"]
	working_set = Set{Vector{BitVector}}() # APs we want to explore
	explored_set = Set{Vector{BitVector}}() # APs we have already explored
	brs = Set{Tuple{Matrix{Float64},Vector{Float64}}}() # backward reachable set, stored as a collection of polytopes
	push!(working_set, ap)


	# traverse connected input space to enumerate brs
	j = 0
	while !isempty(working_set)
		println(j)
		in_brs = false

		ap = pop!(working_set)
		push!(explored_set, ap)
		C, d = ap2map[ap]
		A, b = ap2input[ap]
		
		# check intersection with preimage of each polytope in output set
		for (Aₒ, bₒ) in out_set
			Aᵤ, bᵤ = (Aₒ*C, bₒ-Aₒ*d) # for Aₒy ≤ bₒ and y = Cx+d -> AₒCx ≤ bₒ-Aₒd
			if poly_intersection(A, b, Aᵤ, bᵤ)
				push!(brs, (vcat(A, Aᵤ), vcat(b, bᵤ)))
				in_brs = true
			end
		end

		# if a subset of Ax≤b is in the brs then add neighbor APs to the working set
		if in_brs
			for neighbor_ap in ap2neighbors[ap]
				if neighbor_ap ∉ explored_set && neighbor_ap ∉ working_set
					push!(working_set, neighbor_ap)
				end
			end
		end
		# j += 1
	end

	println("# Polytopes in BRS: ", length(brs))

	# save the i-step brs
	save(string("models/taxinet/BRS/taxinet_brs_", i, "_step.jld2"), Dict("brs" => brs))

	
end

# plot brs
plt = plot_brs(brs)

=#