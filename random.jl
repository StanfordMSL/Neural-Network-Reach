include("forward_reachability.jl")
pyplot()

bound_r(a,b) = (b-a)*(rand()-1) + b # Generates a uniformly random number on [a,b]

function input_constraints_random(weights, type::String; net_dict=[])
	if type == "big box"
		in_dim = size(weights[1],2) - 1
		Aᵢ_pos = Matrix{Float64}(I, in_dim, in_dim)
		Aᵢ_neg = Matrix{Float64}(-I, in_dim, in_dim)
		Aᵢ = vcat(Aᵢ_pos, Aᵢ_neg)
		bᵢ = 1e8*ones(2*in_dim)
	
	elseif type == "hexagon"
		Aᵢ = [1 0; -1 0; 0 1; 0 -1; 1 1; -1 1; 1 -1; -1 -1]
		bᵢ = [5, 5, 5, 5, 8, 8, 8, 8]
	
	else
		error("Invalid input constraint specification.")
	end
	
	return Aᵢ, bᵢ
end

# Plots all polyhedra
function plot_hrep_random(state2constraints; space = "input", net_dict = [])
	plt = plot(reuse = false)
	for state in keys(state2constraints)
		A, b = state2constraints[state]
		if !isempty(net_dict)
			if space == "input"				
				C = Diagonal(vec(net_dict["X_std"]))
				d = vec(net_dict["X_mean"])
			elseif space == "output"
				C = Diagonal(vec(net_dict["Y_std"]))
				d = vec(net_dict["Y_mean"])
			else
				error("Invalid arg given for space")
			end
			reg = Float64.(C)*HPolytope(constraints_list(A,b)) + Float64.(d)
		else
			reg = HPolytope(constraints_list(A,b))
		end
		
		if isempty(reg)
			@show reg
			error("Empty polyhedron.")
		end
		plot!(plt,  reg)
	end
	return plt
end

######################################################################
# weights = test_pyramid() # get NN

## Random Examples ##
flux_net, weights = test_random_flux(2, 2, 30, 3) # (in_d, out_d, hdim, layers; Aₒ=[], bₒ=[], value=false)
Aᵢ, bᵢ = input_constraints_random(weights, "hexagon")
Aₒ = []
bₒ = []

## CartPole Examples ##
# weights = haruki_net("no")
# Aᵢ, bᵢ = input_constraints(weights, "big box")


@time begin
state2input, state2output, state2map, state2backward = forward_reach(weights, Aᵢ, bᵢ, Aₒ, bₒ, reach=false, back=false, verification=false)
end
@show length(state2input)

# Plot all regions #
# plt_in  = plot_hrep_random(state2input, space="input")
# plt_out = plot_hrep_random(state2output, space="output")





# Total saved LPs: 24065
# 5.338476 seconds (24.01 M allocations: 1.467 GiB, 7.27% gc time)
# length(state2input) = 378
# 4.709716 seconds (17.89 M allocations: 1.069 GiB, 6.01% gc time)
# 3.711627 seconds (13.74 M allocations: 800.876 MiB, 5.85% gc time)
# 1.312521 seconds (9.18 M allocations: 560.882 MiB, 9.92% gc time) time when in profile_rand(weights) function
# with ^^, there are 378 regions and 100 nodes, so a worst case LP # of 37,800
# We save 24,065, thus we solve 13,735 LPs. 
# 1.312521s/13,735 LPs = 0.0001s / LP


# 2127     70,4
# Total saved LPs: 430365 (72% of LPs saved)
# 23.160349 seconds (210.16 M allocations: 12.213 GiB, 12.67% gc time)
# 0.00014s / LP

# 4277
# Total saved LPs: 1255831 (73% of LPs saved)
# 161.055837 seconds (765.55 M allocations: 43.953 GiB, 9.29% gc time)
# 0.00035 s / LP
