using Plots, FileIO
include("reach.jl")
include("invariance.jl")

# Returns H-rep of various input sets
function input_constraints_mpc(weights, type::String)
	# Each input specification is in the form Ax≤b
	# The network takes normalized inputs: xₙ = Aᵢₙx + bᵢₙ
	# Thus the input constraints for raw network inputs are: A*inv(Aᵢₙ)x ≤ b + A*inv(Aᵢₙ)*bᵢₙ
	if type == "box"
		in_dim = size(weights[1],2) - 1
		A_pos = Matrix{Float64}(I, in_dim, in_dim)
		A_neg = Matrix{Float64}(-I, in_dim, in_dim)
		A = vcat(A_pos, A_neg)
		b = [5.0, 5.0, 5.0, 5.0]
	elseif type == "hexagon"
		A = [1. 0; -1 0; 0 1; 0 -1; 1 1; -1 1; 1 -1; -1 -1]
		b = [5., 5, 5, 5, 8, 8, 8, 8]
	else
		error("Invalid input constraint specification.")
	end
	return A, b
end

# Returns H-rep of various output sets
function output_constraints_mpc(weights, type::String)
	# Each output specification is in the form Ayₒᵤₜ≤b
	# The raw network outputs are unnormalized: yₒᵤₜ = Aₒᵤₜy + bₒᵤₜ
	# Thus the output constraints for raw network outputs are: A*Aₒᵤₜ*y ≤ b - A*bₒᵤₜ
	if type == "origin"
		A = [1. 0.; -1. 0.; 0. 1.; 0. -1.]
		b = [1., 1., 1., 1.]
 	else 
 		error("Invalid input constraint specification.")
 	end
 	return A, b
end


# Plot all polytopes
function plot_hrep_mpc(state2constraints; type="normal")
	plt = plot(reuse = false, legend=false, xlabel="x₁", ylabel="x₂")
	for state in keys(state2constraints)
		A, b = state2constraints[state]
		reg = HPolytope(constraints_list(A,b))
		
		if isempty(reg)
			@show reg
			error("Empty polyhedron.")
		end

		if type == "normal"
			plot!(plt, reg, xlims=(-5., 5.), ylims=(-5., 5.), fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
		elseif type == "gif"
			plot!(plt, reg, xlims=(-5., 5.), ylims=(-5., 5.), fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
		end
	end
	return plt
end


# make gif of backwards reachable set over time
function BRS_gif(model, Aᵢ, bᵢ, Aₛ, bₛ, steps)
	plt = plot(HPolytope(constraints_list(Aₛ, bₛ)), xlims=(-2.5, 2.5), ylims=(-3, 3))
	anim = @animate for Step in 2:steps
		weights = pytorch_mpc_net("mpc", Step)
		state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₛ], [bₛ], reach=false, back=true, verification=false)
    	plt = plot_hrep_pendulum(state2backward[1], type="gif")
	end
	gif(anim, string("mpc_brs_",steps  ,".gif"), fps = 2)
end


###########################
######## SCRIPTING ########
###########################
# save("models/mpc/mpc_pwa.jld2", Dict("state2input" => state2input, "state2map" => state2map, "Ai" => Aᵢ, "bi" => bᵢ))
# matwrite("models/mpc/mpc_seed.mat", Dict("A_roa" => A_roa, "b_roa" => b_roa, "fp" => fp))
# Also need to save fp

# Given a network representing discrete-time autonomous dynamics and state constraints,
# ⋅ find fixed points
# ⋅ verify the fixed points are stable equilibria
# ⋅ compute invariant polytopes around the fixed points
# ⋅ perform backwards reachability to estimate the maximal region of attraction in the domain

copies = 1 # copies = 1 is original network

weights = pytorch_mpc_net("mpc", copies)
# weights = pytorch_net("mpc", copies)


Aᵢ, bᵢ = input_constraints_mpc(weights, "box")
Aₒ, bₒ = output_constraints_mpc(weights, "origin")
# A_roa = Matrix{Float64}(matread("models/mpc/mpc_seed.mat")["A_roa"])
# b_roa = Vector{Float64}(matread("models/mpc/mpc_seed.mat")["b_roa"])

# Run algorithm
@time begin
state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ])
# state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [A_roa], [b_roa], fp=fp, reach=false, back=true, connected=true)
end
@show length(state2input)
# @show length(state2backward[1])


# Plot all regions #
plt_in1  = plot_hrep_mpc(state2input)
# plt_in2  = plot_hrep_mpc(state2backward[1])



# homeomorph = is_homeomorphism(state2map, size(Aᵢ,2))
# println("PWA function is a homeomorphism: ", homeomorph)

# fixed_points, fp_dict = find_fixed_points(state2map, state2input, weights)
# fp = fixed_points[1]
# @show fp


# A_roa, b_roa, state2backward_chain, plt_in2 = find_roa("mpc", 40, 3) # num_constraints, num_steps
# 10 steps is ~35k polytopes with ~300 polytopes in the BRS
# 15 steps is 88,500 polytopes with 895 polytopes in the BRS
# algorithm does ~1000 polytopes per minute.
# Create gif of backward reachable set
# BRS_gif(model, Aᵢ, bᵢ, A_roa, b_roa, 5)
# nothing


