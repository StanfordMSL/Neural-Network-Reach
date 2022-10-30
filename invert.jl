using Plots, FileIO
include("reach.jl")

# Returns H-rep of various input sets
function input_constraints_invert(weights, type::String)
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
function plot_hrep_invert(ap2constraints; limit=Inf)
	plt = plot(reuse = false)
	for (i, ap) in enumerate(keys(ap2constraints))
		A, b = ap2constraints[ap]
		reg = HPolytope(constraints_list(A,b))
		if isempty(reg)
			@show reg
			error("Empty polyhedron.")
		end
		plot!(plt, reg, fontfamily=font(14, "Computer Modern"), tickfont = (12), xlims=(-10,10), ylims=(-10,10))
		
		i == limit ? (return plt) : nothing
	end
	return plt
end

function plot_centers(plt, ap2centers)
	for ap in keys(ap2centers)
		center = ap2centers[ap]
		scatter!(plt, [center[1]], [center[2]], label=string(ap))
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

function get_centers(ap2input)
	ap2centers = Dict{Vector{BitVector},Vector{Float64} }() # Dict from ap -> chebyshev center of polytope
	for key in keys(ap2input)
		A, b = ap2input[key]
		ap2centers[key], essential, essentialᵢ = cheby_lp(A, b, [], [], collect(1:length(b))) # Chebyshev center
	end
	return ap2centers
end

###########################
######## SCRIPTING ########
###########################
test_dict = load("inverse/test_net.jld2")
weights = test_dict["weights"]
ap2input = test_dict["ap2input"]
ap2map = test_dict["ap2map"]
ap2vertices = get_vrep(ap2input)
ap2centers = get_centers(ap2input)

# Find Uninterrupted hyperplanes
# for each region
# for each essential hyperplane
# for each region
# check whether vertices are all on one side

# function find_hyperplane_arrangement(Hrep_dict, Vrep_dict)
# 	for ap in keys(Hrep_dict)
# 		A, b = Hrep_dict[ap] # matrix, vector
# 		n_cons = length(b)
# 		for hyp in 1:n_cons
# 			uninterrupted = true
# 			for ap in keys(Vrep_dict)
# 				V = Vrep_dict[ap] # vector of vectors
# 				A[hyp,:]⋅v - b
# 			end
# 		end
# 	end
# end


# Plot all regions (only 2D input) #
if in_d == 2
	plt_in  = plot_hrep_random(ap2input, limit=Inf)
	plt_in = plot_centers(plt_in, ap2centers)
end





# Affine Map Difference Between
# [[0, 0, 0, 1], [1, 0, 1, 1]] and [[1, 0, 0, 1], [1, 0, 1, 1]]
0

[[0, 1, 1, 1], [1, 0, 1, 1]]
[[1, 1, 1, 1], [1, 0, 1, 1]]

[[0, 1, 1, 1], [1, 1, 1, 1]]
[[1, 1, 1, 1], [1, 1, 1, 1]]

# Level 1 hyperplanes
[0.94649   -0.322733] x ≤ -0.18480912288396656
[0.753287   0.657692] x ≤ -0.6548318149352409
[0.878853   0.477092] x ≤ -0.6750564184070789

# with normalized 
1.0169337841227133* [0.9307292321068047, -0.3173589126832036] x ≤ -0.1817317172163745
1.1953262220349647* [0.6301936543461568, 0.5502196704765018] x ≤ -0.5478268633816403
1.206524156453402* [0.7284172432845466, 0.39542681134741636] x ≤ -0.5595050996669794


# NN Layer 1 Weights
0.057135430430609827* [0.9307292772755126, -0.31735878477444585, 0.18173170925484927]
0.12059911905473929* [0.6301937529197602, 0.5502195439787257, 0.547826877037583]
0.13476924089489178* [0.72841737350433, 0.39542674221438984, 0.5595049789937244]

# NN Layer 2 Weights
0.1394534787243155* [0.4361899853628132, 0.5790545502416388, 0.5869861057240967, 0.3603906716335787]
0.13384587052518848* [0.7594656690174297, 0.43735950859465933, 0.4733128920979539, -0.08890142855779526]
0.08381868874745202* [0.20079338198426586, 0.3921625594980648, 0.5524153220621422, 0.7076212663778748]