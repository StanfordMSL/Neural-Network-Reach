using Plots, LazySets, FileIO
include("load_networks.jl")


function sample_forward(weights, N, X1, X2, roa2, roa3)
	converged, unconverged = Vector{Tuple{Int64, Int64}}(undef, 0), Vector{Tuple{Int64, Int64}}(undef, 0)
	# check if initial condition gets to roa in N time steps
	for (i, x₁) in enumerate(X1)
		for (j, x₂) in enumerate(X2)
			x′ = [x₁, x₂]
			for k in 1:N
				x′ = eval_net(x′, weights, 1)
				in_domain = all(x′ .≤ maximum.([X1, X2])) && all(x′ .≥ minimum.([X1, X2]))

				if !in_domain
					push!(unconverged, (i,j))
					break
				elseif k == N && !(x′ ∈ roa2 || x′ ∈ roa3)
					push!(unconverged, (i, j))
				elseif k == N
					push!(converged, (i, j))
				end

			end
		end
	end
	return converged, unconverged
end


function reconstruct(converged, unconverged, N, X1, X2)
	plt = scatter(xlabel="x₁", ylabel="x₂", title = string(N, " Steps"), legend = false)

	if length(converged) > 0
		converged_dat = hcat([ [X1[p[1]], X2[p[2]]] for p in converged]...)'
		scatter!(plt, converged_dat[:,1], converged_dat[:,2], color = "blue", markersize=1, markerstrokewidth=0.1)
	end

	if length(unconverged) > 0
		unconverged_dat = hcat([ [X1[p[1]], X2[p[2]]] for p in unconverged]...)'
		scatter!(plt, unconverged_dat[:,1], unconverged_dat[:,2], color = "red", markersize=1, markerstrokewidth=0.1)
	end

	return plt
end

function run_trials(weights, roa1, roa2)
	X1 = -5:.1:5
	X2 = -15:.1:15
	results = Dict() # Dict of time-step -> converged, unconverged data
	for N in 1000:1000
		converged, unconverged = sample_forward(weights, N, X1, X2, roa1, roa2)
		results[string(N)] = (converged, unconverged)
	end
	return results, X1, X2
end


copies = 1 # copies = 1 is original network
weights = taxinet_cl(copies)


# Define ROA
fp2dict = load("models/taxinet/5_15/taxinet_brs2_0_step.jld2")
fp3dict = load("models/taxinet/5_15/taxinet_brs3_0_step.jld2")

roa2 = HPolytope(fp2dict["brs"][1][1], fp2dict["brs"][1][2])
roa3 = HPolytope(fp3dict["brs"][1][1], fp3dict["brs"][1][2])


# run large trials
# results, X1, X2 = run_trials(weights, roa2, roa3)
# save("models/taxinet/sample_roas_4_22.jld2", Dict("X1" => X1, "X2" => X2, "results" => results))


# # plot results
# sample_dict = load("models/taxinet/sample_roas_4_22.jld2")
# results = sample_dict["results"]
# X1, X2 = sample_dict["X1"], sample_dict["X2"]

# N = 1000
# converged, unconverged = results[string(N)]
# plt = reconstruct(converged, unconverged, N, X1, X2)




# scripting
N = 500 # how many steps to consider

X1 = -5:.1:5
X2 = -15:.1:15

@time begin
converged, unconverged = sample_forward(weights, N, X1, X2, roa2, roa3)
end

plt = reconstruct(converged, unconverged, N, X1, X2)
plot!(plt, roa2)
plot!(plt, roa3)

