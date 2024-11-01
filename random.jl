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
	elseif type == "box"
		in_dim = size(weights[1],2) - 1
		Aᵢ_pos = Matrix{Float64}(I, in_dim, in_dim)
		Aᵢ_neg = Matrix{Float64}(-I, in_dim, in_dim)
		Aᵢ = vcat(Aᵢ_pos, Aᵢ_neg)
		bᵢ = 2*ones(2*in_dim)
	elseif type == "hexagon"
		Aᵢ = [1. 0.; -1. 0.; 0. 1.; 0. -1.; 1. 1.; -1. 1.; 1. -1.; -1. -1.]
		bᵢ = [5., 5., 5., 5., 8., 8., 8., 8.]
	else
		error("Invalid input constraint specification.")
	end
	return Aᵢ, bᵢ
end

# Plots all polyhedra
function plot_hrep_random(ap2constraints; limit=Inf)
	plt = plot(reuse = false)
	for (i,ap) in enumerate(keys(ap2constraints))
		A, b = ap2constraints[ap]
		reg = HPolytope(constraints_list(A,b))
		if isempty(reg)
			@show reg
			error("Empty polyhedron.")
		end
		plot!(plt,  reg, fontfamily=font(14, "Computer Modern"), tickfont = (12))
		
		i == limit ? (return plt) : nothing
	end
	return plt
end





###########################
######## SCRIPTING ########
###########################

in_d, out_d, hdim, layers = 2, 2, 10, 5
weights = random_net(in_d, out_d, hdim, layers)

Aᵢ, bᵢ = input_constraints_random(weights, "box")
Aₒ = Matrix{Float64}(undef,0,0)
bₒ = Vector{Float64}()

@time begin
ap2input, ap2output, ap2map, ap2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], reach=true)
end
@show length(ap2input)


# Plot all regions (only 2D input) #
if in_d == 2
	plt_in  = plot_hrep_random(ap2input, limit=Inf)
end