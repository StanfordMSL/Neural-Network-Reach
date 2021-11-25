using LinearAlgebra, MATLAB, FileIO, Plots, LazySets


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


# Load in polytope collection
brs_dict = load(string("models/taxinet/BRS/taxinet_brs_", 10, "_step.jld2"))
out_set = brs_dict["brs"]
plt = plot_brs(out_set)
out_set_mat = mxarray(collect(out_set))
l_out = length(out_set)

As = Vector{Matrix{Float64}}(undef, 0)
bs = Vector{Vector{Float64}}(undef, 0)
for polytope in out_set
	push!(As, polytope[1])
	push!(bs, polytope[2])
end 

As_mat = mxarray(As)
bs_mat = mxarray(bs)

# Create data structures
in_set = Set{ Tuple{Matrix{Float64}, Vector{Float64}} }()



# interface with Matlab
mat"""
for i = 1:length($As_mat)
	% disp($bs_mat{i})
	P(i) = Polyhedron($As_mat{i}, $bs_mat{i})
end

U = PolyUnion('Set',P,'convex',false,'overlaps',false,'Connected',true,'fulldim',true,'bounded',true)
merged = U.merge
$len = length(merged)
%disp(U)

% disp(merged(1).Set(6).A)

for i = 1:length(merged)
	merged(1).Set(i).A
	merged(1).Set(i).b
end
merged.plot
"""
# @mget merged