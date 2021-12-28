#=
This file is used for performing traditional backward reachability of homeomorphic PWA functions.
We can't use other implementations, such as from MPT, because they don't take advantage of 
the homeomorphic property of the PWA function.

I use the following polytopic ROA as my 0-step brs
fp = [-1.089927713157323, -0.12567755953751042]
A_roa = 370*[-0.20814568962857855 0.03271955855771795; 0.2183098663000297 0.12073669880754853; 0.42582101825227686 0.0789033995762251; 0.14480530852927057 -0.05205811047518554; -0.13634673812819695 -0.1155315084750385; 0.04492020060602461 0.09045775648816877; -0.6124506220873154 -0.12811621541510643]
b_roa = ones(size(A_roa,1)) + A_roa*fp
out_set = Set{Tuple{Matrix{Float64},Vector{Float64}}}([(A_roa, b_roa)])
=#

using LinearAlgebra, JuMP, GLPK, LazySets, MATLAB, FileIO, Plots

#=
Given a set of polytopes, interface with the Matlab MPT toolbox to merge
the given union of polytopes into a union of fewer polytopes.
Greedy merging wasn't working for me so I'm using optimal merging.
=#
function merge_polytopes(polytopes::Set{ Tuple{Matrix{Float64}, Vector{Float64}} }; verbose=false)
	verbose ? println("Started with ", length(polytopes), " polytopes") : nothing

	# Send polytopes to Matlab variables
	As = Vector{Matrix{Float64}}(undef, length(polytopes))
	bs = Vector{Vector{Float64}}(undef, length(polytopes))
	for (i,polytope) in enumerate(polytopes)
		As[i] = polytope[1]
		bs[i] = polytope[2]
	end 
	As_mat = mxarray(As)
	bs_mat = mxarray(bs)
	println("Here 0")

	# interface with Matlab MPT merge() method for PolyUnion objects
	# optimal merging creates a polytopic covering an hence we set overlaps to true.
	mat"""
	disp('Here 0.5')
	abc = mptopt('lpsolver', 'GLPK');
	clear P;
	clear U;
	clear merged;
	for i = 1:length($As_mat)
		P(i) = Polyhedron($As_mat{i}, $bs_mat{i});
	end
	disp('Here 1')
	U = PolyUnion('Set',P,'convex',false,'overlaps',true,'Connected',true,'fulldim',true,'bounded',true);
	merged = U.merge('optimal', true);
	disp('Here 2')
	$len_m = length(merged.Set);
	$As_merged = cell($len_m,1);
	$bs_merged = cell($len_m,1);

	for i = 1:$len_m
		$As_merged{i,1} = merged.Set(i).A;
	 	$bs_merged{i,1} = merged.Set(i).b;
	end
	"""

	# Send merged polytopes to Julia variables
	verbose ? println("Ended with ", length(As_merged), " polytopes") : nothing
	merged_set = Set{ Tuple{Matrix{Float64}, Vector{Float64}} }()
	for i in 1:length(As_merged)
		push!(merged_set, (As_merged[i], bs_merged[i]))
	end

	return merged_set
end


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
function backward_reach(pwa_dict, i; merge=false, optMerge=false, Save=false, verbose=false)
	verbose ? println("Computing ", i, "-step BRS.") : nothing

	# load in homeomorphic PWA function
	ap2map = pwa_dict["ap2map"]
	ap2input = pwa_dict["ap2input"]
	ap2neighbors = pwa_dict["ap2neighbors"]

	# load in set to find preimage of
	if merge
		brs_dict = load(string("models/taxinet/BRS_merge/taxinet_brs_", i-1, "_step.jld2"))
	else
		brs_dict = load(string("models/taxinet/BRS_no_merge/taxinet_brs_", i-1, "_step.jld2"))
	end
	output_polytopes = brs_dict["brs"]

	# find ap for cell that fp lives in
	ap = pwa_dict["ap_fp"]
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
	if optMerge && length(brs_polytopes) > 1
		brs_polytopes = merge_polytopes(brs_polytopes; verbose=verbose)
	end

	# save the i-step brs
	if Save
		if merge
			save(string("models/taxinet/BRS_merge/taxinet_brs_", i, "_step.jld2"), Dict("brs" => brs_polytopes))
		else
			save(string("models/taxinet/BRS_no_merge/taxinet_brs_", i, "_step.jld2"), Dict("brs" => brs_polytopes))
		end
	end

	verbose ? println(length(brs_polytopes), " in the BRS.") : nothing

	return brs_polytopes
end
















# save("models/taxinet/taxinet_pwa_map_large.jld2", Dict("ap2map" => ap2map, "ap2input" => ap2input, "ap2neighbors" => ap2neighbors, "ap_fp" => ap_fp, "fp" => fp, "Aᵢ" => Aᵢ, "bᵢ" => bᵢ))















### Scripting ###
# pwa_dict = load("models/taxinet/taxinet_pwa_map_large.jld2")
# start_steps = 136
# end_steps = 136
# stats = load("models/taxinet/BRS_merge/stats.jld2")
# times = stats["times"]
# poly_counts = stats["poly_counts"]
# Save = true
# Merge = true
# optMerge = true

# # To compute multiple steps #
# for i in start_steps:end_steps
# 	println("\n")
# 	brs_polytopes, t, bytes, gctime, memallocs = @timed backward_reach(pwa_dict, i; merge=Merge, optMerge=optMerge, Save=Save, verbose=true)
	
# 	if length(times) == length(poly_counts)
# 		if i ≤ length(times)
# 			times[i] = t
# 			poly_counts[i] = length(brs_polytopes)
# 		elseif i == length(times)+1
# 			push!(times, t)
# 			push!(poly_counts, length(brs_polytopes))
# 		end
# 	end

# 	Save ? save("models/taxinet/BRS_merge/stats.jld2", Dict("times" => times, "poly_counts" => poly_counts)) : nothing
# end



# To compute one step #
# brs_polytopes, t, bytes, gctime, memallocs = @timed backward_reach(pwa_dict, 6; merge=true, Save=false, verbose=true)
# plt = plot_polytopes(brs_polytopes)






## To compute multiple steps ##
# max_steps = 100
# times = Matrix{Float64}(undef, max_steps, 2)
# poly_counts = Matrix{Int64}(undef, max_steps, 2)

# for i in 1:max_steps
# 	println("\n")
# 	# compute brs union of polytopes
# 	brs_polytopes, t, bytes, gctime, memallocs = @timed backward_reach(i; merge=false, Save=true, verbose=true)
# 	times[i,1], poly_counts[i,1] = t, length(brs_polytopes)

# 	brs_polytopes, t, bytes, gctime, memallocs = @timed backward_reach(i; merge=true, Save=true, verbose=true)
# 	times[i,2], poly_counts[i,2] = t, length(brs_polytopes)

# 	# plot brs
# 	# plt = plot_polytopes(brs_polytopes)
# end









## To plot a BRS ##
merge = true
i = 135
if merge
	brs_dict = load(string("models/taxinet/BRS_merge/taxinet_brs_", i, "_step.jld2"))
else
	brs_dict = load(string("models/taxinet/BRS_no_merge/taxinet_brs_", i, "_step.jld2"))
end
output_polytopes = brs_dict["brs"]
plt = plot_polytopes(output_polytopes)
