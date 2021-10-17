using Plots
include("reach.jl")

# Returns H-rep of various input sets
function input_constraints_random(weights, type::String)
	if type == "big box"
		in_dim = size(weights[1],2) - 1
		Aᵢ_pos = Matrix{Float64}(I, in_dim, in_dim)
		Aᵢ_neg = Matrix{Float64}(-I, in_dim, in_dim)
		Aᵢ = vcat(Aᵢ_pos, Aᵢ_neg)
		bᵢ = 1e2*ones(2*in_dim)
	elseif type == "hexagon"
		Aᵢ = [1. 0.; -1. 0.; 0. 1.; 0. -1.; 1. 1.; -1. 1.; 1. -1.; -1. -1.]
		bᵢ = [5., 5., 5., 5., 8., 8., 8., 8.]
	else
		error("Invalid input constraint specification.")
	end
	return Aᵢ, bᵢ
end

# Plots all polyhedra
function plot_hrep_random(state2constraints; space = "input")
	plt = plot(reuse = false)
	for state in keys(state2constraints)
		A, b = state2constraints[state]
		reg = HPolytope(constraints_list(A,b))
		if isempty(reg)
			@show reg
			error("Empty polyhedron.")
		end
		plot!(plt,  reg, fontfamily=font(14, "Computer Modern"), tickfont = (12))
	end
	return plt
end

###########################
######## SCRIPTING ########
###########################
weights = random_net(2, 2, 20, 5) # (in_d, out_d, hdim, layers)
Aᵢ, bᵢ = input_constraints_random(weights, "hexagon")
Aₒ = Matrix{Float64}(undef,0,0)
bₒ = Vector{Float64}()

@time begin
state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], reach=false, back=false, verification=false)
end
@show length(state2input)

# Plot all regions (only 2D input) #
plt_in  = plot_hrep_random(state2input, space="input")

