include("forward_reachability.jl")
pyplot()

bound_r(a,b) = (b-a)*(rand()-1) + b # Generates a uniformly random number on [a,b]

### INPUT CONSTRAINT FUNCTIONS ###
# acas properties defined in original reluplex paper appendix
# For neural networks we often normalize the data before input to the network.
# This normalization is an affine map: x_net = (x - x_mean) ./ x_std --> x = Cx_net + d where C = Diagonal(x_std), d = x_mean
# We can then take our original input constraint, Ax≤b and substitute the above identity so it is properly defined in light of normalization: A(Cx_net + d)≤b
function input_constraints_acas(weights, type::String; net_dict=[])
	# ACAS input  = [ρ, θ, ψ, v_own, v_int]
	if type == "acas property 1"
		nothing
	elseif type == "acas property 2"
		nothing
		nothing
	elseif type == "acas property 3"
		A = [-1  0  0  0  0; # ρ
			   1  0  0  0  0; # ρ
			   0 -1  0  0  0; # θ
			   0  1  0  0  0; # θ
			   0  0 -1  0  0; # ψ
			   0  0  1  0  0; # ψ
			   0  0  0 -1  0; # v_own
			   0  0  0  1  0; # v_own
			   0  0  0  0 -1; # v_int
 			   0  0  0  0  1] # v_int
 		b = [-1500, 1800, 0.06, 0.06, -3.1, 3.14, -980, 1200, -960, 1200]
		σ = Diagonal(vec(net_dict["range_for_scaling"][1:end-1]))
		μ = vec(net_dict["means_for_scaling"][1:end-1])
		Aᵢ = A*σ
		bᵢ = b - A*μ

	elseif type == "acas property 4"
		A = [-1  0  0  0  0; # ρ
			   1  0  0  0  0; # ρ
			   0 -1  0  0  0; # θ
			   0  1  0  0  0; # θ
			   0  0 -1  0  0; # ψ
			   0  0  1  0  0; # ψ
			   0  0  0 -1  0; # v_own
			   0  0  0  1  0; # v_own
			   0  0  0  0 -1; # v_int
 			   0  0  0  0  1] # v_int
 		b = [-1500, 1800, 0.06, 0.06, 0.0, 0.0, -1000, 1200, -700, 800]
		σ = Diagonal(vec(net_dict["range_for_scaling"][1:end-1]))
		μ = vec(net_dict["means_for_scaling"][1:end-1])
		Aᵢ = A*σ
		bᵢ = b - A*μ
	
	else
		error("Invalid input constraint specification.")
	end
	
	return Aᵢ, bᵢ
end

function output_constraints_acas(weights, type::String; net_dict=[])
	if type == "acas property 3" || type == "acas property 4"
		A = [1 -1 0 0 0;
			 1 0 -1 0 0;
			 1 0 0 -1 0;
			 1 0 0 0 -1]
 		b = [0, 0, 0, 0]
		σ = net_dict["range_for_scaling"][end]
		μ = net_dict["means_for_scaling"][end]
		Aₒ = σ*A
		bₒ = b - A*μ*ones(size(A,2))
 	else 
 		error("Invalid input constraint specification.")
 	end
 	return Aₒ, bₒ
end


######################################################################
# ADD IN THAT NEIGHBOR CONTSTRAINT IS ESSENTIAL #
## ACAS Examples ##
weights, net_dict = acas_net(5,5)

property = "acas property 3"
Aᵢ, bᵢ = input_constraints_acas(weights, property, net_dict=net_dict)
Aₒ, bₒ = output_constraints_acas(weights, property, net_dict=net_dict)

function profile(weights, Aᵢ, bᵢ, Aₒ, bₒ)
	@time begin
	try
		state2input, state2output, state2map, state2backward = forward_reach(weights, Aᵢ, bᵢ, Aₒ, bₒ, reach=false, back=false, verification=true)
	catch termination_status
		@show termination_status
		return state2input, state2output, state2map, state2backward
	end
	end
	@show length(state2input)
	return nothing
end

@time profile(weights, Aᵢ, bᵢ, Aₒ, bₒ)

# using ProfileView
# @profview profile(weights, Aᵢ, bᵢ, Aₒ, bₒ)


## SPEED ##
#=
(5,5)
parallel
length(visited) = 1305
No input maps to the target set.
33.410441 seconds (29.15 M allocations: 1.299 GiB, 0.89% gc time)
length(state2input) = 1305
33.901310 seconds (30.22 M allocations: 1.350 GiB, 0.88% gc time)

not parallel
1305
No input maps to the target set.
Total solved LPs: 35831
Total saved LPs:  381361/417192 : 91.4% pruned.
 40.567237 seconds (56.27 M allocations: 6.748 GiB, 3.15% gc time)




@inbounds everywhere saves me ~0 seconds on 5,5. Not worth it for the risk incurred.

1,7: 0.36s :: them: 7.4s
1,8: 0.38s :: them: 5.4s
1,9: 0.38s :: them: 3.9s
5,5: 40.6s :: them: 15.0s
5,8: 76.4s :: them: 13.8s


 10.246097 seconds (14.75 M allocations: 1.769 GiB, 2.80% gc time)
length(state2input) = 351
 12.381916 seconds (17.91 M allocations: 1.913 GiB, 2.63% gc time)


351
No input maps to the target set.
Total solved LPs: 9075
Total saved LPs:  103137/112212 : 91.9% pruned.
 12.067553 seconds (30.27 M allocations: 3.761 GiB, 4.34% gc time)
length(state2input) = 351
 13.519152 seconds (32.72 M allocations: 3.874 GiB, 4.08% gc time)


Total saved LPs:  103137/112212 : 91.9% pruned.
 12.224731 seconds (30.27 M allocations: 3.761 GiB, 4.39% gc time)
length(state2input) = 351
 13.669203 seconds (32.71 M allocations: 3.874 GiB, 4.09% gc time)


Total saved LPs:  103137/112212 : 91.9% pruned.
 12.797318 seconds (62.17 M allocations: 4.586 GiB, 5.72% gc time)
length(state2input) = 351
 14.343512 seconds (64.67 M allocations: 4.702 GiB, 5.29% gc time)







Total saved LPs:  31335/34165 : 91.7% pruned.
  3.551085 seconds (18.97 M allocations: 1.345 GiB, 7.84% gc time)
length(state2input) = 107
  5.129195 seconds (21.49 M allocations: 1.462 GiB, 5.43% gc time)


Total saved LPs:  31335/34165 : 91.7% pruned.
  3.515708 seconds (18.93 M allocations: 1.341 GiB, 7.06% gc time)
length(state2input) = 107
  5.090892 seconds (21.43 M allocations: 1.457 GiB, 5.53% gc time)



Total solved LPs: 2202
Total saved LPs:  25958/28160 : 92.2% pruned.
  2.951224 seconds (15.56 M allocations: 1.096 GiB, 6.63% gc time)
length(state2input) = 88
  4.527843 seconds (18.06 M allocations: 1.212 GiB, 4.88% gc time)


No input maps to the target set.
Total solved LPs: 35831
Total saved LPs:  381343/417174 : 91.4% pruned.
 48.397591 seconds (231.44 M allocations: 17.261 GiB, 6.10% gc time)
length(state2input) = 1305
 49.951297 seconds (233.94 M allocations: 17.377 GiB, 5.97% gc time)


Total solved LPs: 35831
Total saved LPs:  381343/417174 : 91.4% pruned.
 48.510647 seconds (231.45 M allocations: 17.258 GiB, 6.13% gc time)
length(state2input) = 1305
 50.138788 seconds (233.94 M allocations: 17.374 GiB, 6.03% gc time)

Total solved LPs: 35831
Total saved LPs:  381343/417174 : 91.4% pruned.
 47.128435 seconds (112.83 M allocations: 14.190 GiB, 4.25% gc time)
length(state2input) = 1305
 48.582996 seconds (115.28 M allocations: 14.303 GiB, 4.16% gc time)

Total saved LPs:  381343/417174 : 91.4% pruned.
 46.950540 seconds (112.83 M allocations: 14.190 GiB, 4.27% gc time)
length(state2input) = 1305
 48.431624 seconds (115.28 M allocations: 14.303 GiB, 4.21% gc time)

Total saved LPs:  381343/417174 : 91.4% pruned.
 46.318695 seconds (112.83 M allocations: 14.190 GiB, 4.34% gc time)
length(state2input) = 1305
 47.682091 seconds (115.28 M allocations: 14.303 GiB, 4.22% gc time)

# using unique_custom()
Total saved LPs:  381361/417192 : 91.4% pruned.
 40.038496 seconds (55.39 M allocations: 6.778 GiB, 2.49% gc time)
length(state2input) = 1305
 41.801792 seconds (58.36 M allocations: 6.915 GiB, 2.48% gc time)






Total solved LPs: 2937
Total saved LPs:  30233/33170 : 91.1% pruned.
3.025980 seconds (17.27 M allocations: 1.241 GiB, 6.67% gc time)
length(state2input) = 107
4.587160 seconds (19.87 M allocations: 1.361 GiB, 4.92% gc time)


Total solved LPs: 37136
Total saved LPs:  367414/404550 : 90.8% pruned.
 43.877379 seconds (211.22 M allocations: 16.031 GiB, 5.83% gc time)
length(state2input) = 1305
 45.471109 seconds (213.90 M allocations: 16.156 GiB, 5.69% gc time)

Total solved LPs: 37136
Total saved LPs:  367414/404550 : 90.8% pruned.
 44.955126 seconds (211.17 M allocations: 16.029 GiB, 5.99% gc time)
length(state2input) = 1305
 46.531892 seconds (213.68 M allocations: 16.145 GiB, 5.86% gc time)

Total solved LPs: 35831
Total saved LPs:  368719/404550 : 91.1% pruned.
 44.233829 seconds (211.07 M allocations: 16.017 GiB, 5.21% gc time)
length(state2input) = 1305
 45.777152 seconds (213.56 M allocations: 16.133 GiB, 5.08% gc time)

Total saved LPs:  381343/417174 : 91.4% pruned.
 43.444595 seconds (211.07 M allocations: 16.017 GiB, 5.23% gc time)
length(state2input) = 1305
 45.003554 seconds (213.56 M allocations: 16.133 GiB, 5.10% gc time)



Property 3
N₅₉
Recorded solved LPs: 6521
Total solved LPs: 7163
Total saved LPs:  26649/33812 : 78.8% pruned.
3.796642 seconds (20.07 M allocations: 1.412 GiB, 5.26% gc time)
length(state2input) = 107

Adding input set constraints in bounding box heuristic
Recorded solved LPs: 2657
Total solved LPs: 3299
Total saved LPs:  30513/33812 : 90.2% pruned.
3.441163 seconds (19.65 M allocations: 1.382 GiB, 5.55% gc time)
length(state2input) = 107

Not solving the cheby lp
Total solved LPs: 3043
Total saved LPs:  30127/33170 : 90.8% pruned.
2.970665 seconds (17.24 M allocations: 1.239 GiB, 5.60% gc time)
length(state2input) = 107

Fixed type stability issues
Total solved LPs: 3043
Total saved LPs:  30127/33170 : 90.8% pruned.
2.901200 seconds (17.24 M allocations: 1.239 GiB, 6.05% gc time)
length(state2input) = 107
This is as fast as face lattice

N₅₈
Total solved LPs: 70606
Total saved LPs:  663474/734080 : 90.4% pruned.
80.453692 seconds (383.75 M allocations: 29.465 GiB, 5.49% gc time)
length(state2input) = 2368
This is 2x slower than face lattice


Total solved LPs: 2377
Total saved LPs:  24903/27280 : 91.3% pruned.
length(state2input) = 88
5.674426 seconds (19.34 M allocations: 1.278 GiB, 5.44% gc time)

With ray shooting (slower)
Total solved LPs: 1856
Total saved LPs:  25424/27280 : 93.2% pruned.
length(state2input) = 88
6.135284 seconds (21.55 M allocations: 1.416 GiB, 4.83% gc time)


N₅₅
With neighbor constraint cache
Total solved LPs: 37136
Total saved LPs:  367414/404550 : 90.8% pruned.
48.723185 seconds (231.57 M allocations: 17.271 GiB, 5.20% gc time)
length(state2input) = 1305
50.303510 seconds (234.17 M allocations: 17.391 GiB, 5.10% gc time)


=#



