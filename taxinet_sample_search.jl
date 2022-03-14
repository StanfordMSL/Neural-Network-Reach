using Plots, LazySets, FileIO
include("load_networks.jl")


function sample_forward(weights, N, X1, X2, reg_roa)
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
				elseif k == N && x′ ∉ reg_roa
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

function run_trials()
	X1 = -5:.1:5
	X2 = -15:.2:15
	results = Dict() # Dict of time-step -> converged, unconverged data
	for N in 50:50:1000
		converged, unconverged = sample_forward(weights, N, X1, X2, reg_roa)
		results[string(N)] = (converged, unconverged)
	end
	return results, X1, X2
end


copies = 1 # copies = 1 is original network
weights = taxinet_cl(copies)


# Define ROA
fp = [-1.089927713157323, -0.12567755953751042]
A_roa = 370*[-0.20814568962857855 0.03271955855771795; 0.2183098663000297 0.12073669880754853; 0.42582101825227686 0.0789033995762251; 0.14480530852927057 -0.05205811047518554; -0.13634673812819695 -0.1155315084750385; 0.04492020060602461 0.09045775648816877; -0.6124506220873154 -0.12811621541510643]
b_roa = ones(size(A_roa,1)) + A_roa*fp
reg_roa = HPolytope(A_roa, b_roa)


# large trials
# results, X1, X2 = run_trials()
# save("models/taxinet/sample_roas_2.jld2", Dict("X1" => X1, "X2" => X2, "results" => results))

sample_dict = load("models/taxinet/sample_roas_2.jld2")
results = sample_dict["results"]
X1, X2 = sample_dict["X1"], sample_dict["X2"]

N = 1000
converged, unconverged = results[string(N)]

plt = reconstruct(converged, unconverged, N, X1, X2)




# scripting
# N = 5 # how many steps to consider

# X1 = -10:.1:10
# X2 = -30:.2:30

# @time begin
# converged, unconverged = sample_forward(weights, N, X1, X2, reg_roa)
# end

# plt = reconstruct(converged, unconverged, N, X1, X2)
