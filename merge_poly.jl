using MATLAB

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

	# interface with Matlab MPT merge() method for PolyUnion objects
	# optimal merging creates a polytopic covering an hence we set overlaps to true.
	mat"""
	temp = mptopt('lpsolver', 'GLPK');
	clear P;
	clear U;
	clear merged;
	for i = 1:length($As_mat)
		P(i) = Polyhedron($As_mat{i}, $bs_mat{i});
	end
	U = PolyUnion('Set',P,'convex',false,'overlaps',true,'Connected',true,'fulldim',true,'bounded',true);
	merged = U.merge('optimal', true);
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


