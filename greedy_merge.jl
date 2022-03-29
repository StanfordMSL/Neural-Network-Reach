using LazySets, FileIO
include("merge_poly.jl")



# sample a polytope

# find its essential constraints

# generate point across neighbor constraint







### Scripting ###
pwa_info = load("models/taxinet/5_15/back_reach_info.jld2")
fp2, fp3 = pwa_info["fp2"], pwa_info["fp3"]

## To plot a BRS ##
i = 40
brs_dict = load(string("models/taxinet/5_15/taxinet_brs3_", i, "_step_overlap.jld2"))
brs_polytopes = brs_dict["brs"]


brs_polytopes = merge_polytopes(brs_polytopes; verbose=true)

nothing