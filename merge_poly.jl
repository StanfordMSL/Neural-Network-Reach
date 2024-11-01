using MAT, FileIO, JLD2

BRS = load("models/taxinet/taxinet_brs_10_step.jld2")

polytopes = BRS["brs"]

# Send polytopes to Matlab variables
As = Vector{Matrix{Float64}}(undef, length(polytopes))
bs = Vector{Vector{Float64}}(undef, length(polytopes))
for (i,key) in enumerate(keys(polytopes))
	polytope = polytopes[key]
	As[i] = polytope[1]
	bs[i] = polytope[2]
end 

# save matlab data structures
# matwrite("models/taxinet/taxinet_brs_10_step.mat", Dict("As_mat" => As, "bs_mat" => bs))

# save .mat as .jld2
# vars = matread("taxinet_brs_10_step_overlap.mat")
# As = vars["As_merged"]
# bs = vars["bs_merged"]
# polytopes = Set{Tuple{Matrix{Float64}, Vector{Float64}}}()
# for i in 1:length(As)
# 	push!(polytopes, (As[i], bs[i][:,1]))
# end
# save("taxinet_brs_10_step_overlap.jld2", Dict("brs" => polytopes))