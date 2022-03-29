#=
This file is used for performing traditional backward reachability of homeomorphic PWA functions.
We can't use other implementations, such as from MPT, because they don't take advantage of 
the homeomorphic property of the PWA function.
=#

using LinearAlgebra, JuMP, GLPK, LazySets, MATLAB, FileIO, Plots
include("merge_poly.jl")


# Solve a feasibility LP to check if two polyhedron intersect
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


# Plot union of polytopes
function plot_polytopes(polytopes)
	# plt = plot(reuse = false, legend=false, xlabel="x₁", ylabel="x₂", xlims=[-1.175, -1.0], ylims=[-0.5, 0.2])
	plt = plot(reuse = false, legend=false, xlabel="x₁", ylabel="x₂")

	for (A, b) in polytopes
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
This function computes an i-step BRS, given a saved (i-1)-step step BRS.
merge flag controls whether BRS polytopes are merged before returning result.
save flag controls whether the resulting i-step BRS is saved to file
=#
function backward_reach(pwa_dict, pwa_info, i; Save=false, verbose=false)
	verbose ? println("Computing ", i, "-step BRS.") : nothing

	# load in homeomorphic PWA function
	ap2map = pwa_dict["ap2map"]
	ap2input = pwa_dict["ap2input"]
	ap2neighbors = pwa_dict["ap2neighbors"]

	# load in set to find preimage of
	brs_dict = load(string("models/taxinet/5_15/taxinet_brs2_", i-1, "_step_overlap.jld2"))
	# brs_dict = load(string("models/taxinet/5_15/taxinet_brs3_", i-1, "_step_overlap.jld2"))
	output_polytopes = brs_dict["brs"]

	# find ap for cell that fp lives in
	ap = pwa_info["ap_fp2"]
	# ap = pwa_info["ap_fp3"]
	working_set = Set{Vector{BitVector}}() # APs we want to explore
	explored_set = Set{Vector{BitVector}}() # APs we have already explored
	brs_polytopes = Set{Tuple{Matrix{Float64},Vector{Float64}}}() # backward reachable set, stored as a collection of polytopes
	push!(working_set, ap)

	# traverse connected input space to enumerate brs_polytopes
	while !isempty(working_set)
		in_brs = false

		ap = pop!(working_set)
		push!(explored_set, ap)
		C, d = ap2map[ap]
		A, b = ap2input[ap]
		
		# check intersection with preimage of each polytope in output set
		for (Aₒ, bₒ) in output_polytopes
			Aᵤ, bᵤ = (Aₒ*C, bₒ-Aₒ*d) # compute preimage polytope
			if poly_intersection(A, b, Aᵤ, bᵤ)
				push!(brs_polytopes, (vcat(A, Aᵤ), vcat(b, bᵤ)))
				in_brs = true
			end
		end

		# if a subset of Ax≤b is in the brs_polytopes then add neighbor APs to the working set
		if in_brs
			for neighbor_ap in ap2neighbors[ap]
				if neighbor_ap ∉ explored_set && neighbor_ap ∉ working_set
					push!(working_set, neighbor_ap)
				end
			end
		end
	end

	# optimal merging of polytopes in BRS
	if length(brs_polytopes) > 1
		brs_polytopes = merge_polytopes(brs_polytopes; verbose=verbose)
	end

	# save the i-step brs
	if Save
		save(string("models/taxinet/5_15/taxinet_brs2_", i, "_step_overlap.jld2"), Dict("brs" => brs_polytopes))
		# save(string("models/taxinet/5_15/taxinet_brs3_", i, "_step_overlap.jld2"), Dict("brs" => brs_polytopes))
	end

	verbose ? println(length(brs_polytopes), " in the BRS.") : nothing

	return brs_polytopes
end






# ap2map = pwa_dict["ap2map"]
# ap2input = pwa_dict["ap2input"]
# ap2neighbors = pwa_dict["ap2neighbors"]
# Aᵢ = pwa_dict["Aᵢ"]
# bᵢ = pwa_dict["bᵢ"]
# save("models/taxinet/taxinet_pwa_map_5_15.jld2", Dict("ap2map" => ap2map, "ap2input" => ap2input, "ap2neighbors" => ap2neighbors, "ap_fp2" => ap2, "ap_fp3" => ap3, "fp2" => fp2, "fp3" => fp3, "Aᵢ" => Aᵢ, "bᵢ" => bᵢ))
# save("models/taxinet/5_15/back_reach_info.jld2", Dict("ap_fp2" => ap2, "ap_fp3" => ap3, "fp2" => fp2, "fp3" => fp3, "Aᵢ" => Aᵢ, "bᵢ" => bᵢ))







# init 0-step sets
# save("models/taxinet/5_15/stats3.jld2", Dict("times" => [0.0], "poly_counts" => [1]))
# save("models/taxinet/5_15/taxinet_brs3_0_step.jld2", Dict("brs" => [(A3_roa, b3_roa)]))






### Scripting ###
pwa_dict = load("models/taxinet/taxinet_pwa_map_5_15.jld2")
pwa_info = load("models/taxinet/5_15/back_reach_info.jld2")
start_steps = 20
end_steps = 500

times, poly_counts = Vector{Float64}(undef, 0), Vector{Int64}(undef, 0)
Save = true
stats = load("models/taxinet/5_15/stats2_overlap.jld2")
# stats = load("models/taxinet/5_15/stats3_overlap.jld2")

times = stats["times"]
poly_counts = stats["poly_counts"]

# To compute multiple steps #
for i in start_steps:end_steps
	println("\n")
	brs_polytopes, t, bytes, gctime, memallocs = @timed backward_reach(pwa_dict, pwa_info, i; Save=Save, verbose=true)
	
	if length(times) == length(poly_counts)
		if i ≤ length(times)
			times[i] = t
			poly_counts[i] = length(brs_polytopes)
		elseif i == length(times)+1
			push!(times, t)
			push!(poly_counts, length(brs_polytopes))
		end
	end

	if Save
		save("models/taxinet/5_15/stats2_overlap.jld2", Dict("times" => times, "poly_counts" => poly_counts))
		# save("models/taxinet/5_15/stats3_overlap.jld2", Dict("times" => times, "poly_counts" => poly_counts))
	end
	println("total time: ", t)
end








## To plot a BRS ##
# i = 19
# brs_dict = load(string("models/taxinet/5_15/taxinet_brs2_", i, "_step_overlap.jld2"))
# brs_dict = load(string("models/taxinet/5_15/taxinet_brs3_", i, "_step_overlap.jld2"))
# output_polytopes = brs_dict["brs"]
# plt = plot_polytopes(output_polytopes)

# savefig(plt, string(i, ".png"))
