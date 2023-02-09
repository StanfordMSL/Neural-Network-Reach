using Plots
include("reach.jl")
include("invariance.jl")

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
function plot_hrep_random(ap2constraints)
	plt = plot(reuse = false)
	for ap in keys(ap2constraints)
		A, b = ap2constraints[ap]
		reg = HPolytope(constraints_list(A,b))
		if isempty(reg)
			@show reg
			error("Empty polyhedron.")
		end
		plot!(plt,  reg, fontfamily=font(14, "Computer Modern"), tickfont = (12))
	end
	return plt
end

function get_vrep(ap2input)
	ap2vertices = Dict{Vector{BitVector}, Vector{Vector{Float64}} }() # Dict from ap -> vector of vertices for input polytope
	for key in keys(ap2input)
		A, b = ap2input[key]
		ap2vertices[key] = tovrep(HPolytope(A,b)).vertices
	end
	return ap2vertices
end

###########################
######## SCRIPTING ########
###########################

make_fig_1 = false

in_d, out_d, hdim, layers = 2, 2, 10, 3

if make_fig_1
	in_d, out_d, hdim, layers = 2, 2, 10, 3
	weights = random_net(in_d, out_d, hdim, layers)
	
	# modify weights to scale up output for nicer plot
	for (i,w) in enumerate(weights)
		if i == 1
			n_rows = size(w,1)
			Γ = diagm(50*randn(n_rows))
			Γ[end,end] = 1
			weights[i] = Γ*w
		end
	end
else
	in_d, out_d, hdim, layers = 2, 2, 10, 5
	weights = random_net(in_d, out_d, hdim, layers)
end

Aᵢ, bᵢ = input_constraints_random(weights, "big box")
Aₒ = Matrix{Float64}(undef,0,0)
bₒ = Vector{Float64}()

@time begin
ap2input, ap2output, ap2map, ap2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], reach=true, back=false, verification=false)
end
@show length(ap2input)


# Plot all regions (only 2D input) #
plt_in  = plot_hrep_random(ap2input)


homeomorph = is_homeomorphism(ap2map, size(Aᵢ,2))
println("PWA function is a homeomorphism: ", homeomorph)





# Make Figure 1 for paper (requires ap2input and ap2output to be OrderedDict type)
if make_fig_1
	plt_fig1 = make_figure_1(ap2input, ap2output, [1, 60, Inf])
end
