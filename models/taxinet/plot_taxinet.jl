using LazySets, FileIO, Plots
include("../../load_networks.jl")
include("../../reach.jl")
include("../../invariance.jl")



# Plot union of polytopes
function plot_polytopes(polytopes)
	# plt = plot(reuse = false, legend=false, xlabel="x₁", ylabel="x₂", xlims=[-1.175, -1.0], ylims=[-0.5, 0.2])
	plt = plot(reuse = false, legend=false, xlabel="Crosstrack Position (m.)", ylabel="Heading Angle (deg.)")

	for (A, b) in polytopes
		reg = HPolytope(constraints_list(A, b))
		if isempty(reg)
			@show reg
			error("Empty polyhedron.")
		end
		plot!(plt,  reg, color="paleturquoise1", fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
	end
	return plt
end



### Scripting ###
weights = taxinet_cl(1)
pwa_dict = load("models/taxinet/taxinet_pwa_map_5_15.jld2")
pwa_info = load("models/taxinet/5_15/back_reach_info.jld2")
# fp2, fp3 = pwa_info["fp2"], pwa_info["fp3"]

# find any fixed points if they exist
# fixed_points, fp_dict = find_fixed_points(pwa_dict["ap2map"], pwa_dict["ap2input"], weights)


i = 51
brs_dict2 = load(string("models/taxinet/5_15/taxinet_brs2_", i, "_step_overlap.jld2"))
brs_polytopes2 = brs_dict2["brs"]
brs_dict3 = load(string("models/taxinet/5_15/taxinet_brs3_", i, "_step_overlap.jld2"))
brs_polytopes3 = brs_dict3["brs"]

@show length(brs_polytopes2)
@show length(brs_polytopes3)

# plt = plot_polytopes(brs_polytopes2 ∪ brs_polytopes3)

# scatter!(plt, [fixed_points[1][1]], [fixed_points[1][2]], color="red", label="unstable")
# scatter!(plt, [fixed_points[2][1], fixed_points[3][1]], [fixed_points[2][2], fixed_points[3][2]], color="yellow", label="stable")


## To plot a BRS ##
# i = 17
# brs_dict = load(string("models/taxinet/5_15/taxinet_brs3_", i, "_step_overlap.jld2"))
# brs_polytopes = brs_dict["brs"]
# plt = plot_polytopes(brs_polytopes)

# scatter!(plt, [fp2[1], fp3[1]], [fp2[2], fp3[2]], label="fixed points")

# savefig(plt, string(i, ".png"))

# brs_polytopes2 = merge_polytopes(brs_polytopes; verbose=true)
# plt2 = plot_polytopes(brs_polytopes2)


