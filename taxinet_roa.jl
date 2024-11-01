using Plots, FileIO
include("reach.jl")
include("invariance.jl")

# Returns H-rep of various input sets
function input_constraints_taxinet()
	A = [1. 0.; -1. 0.; 0. 1.; 0. -1.]
	b = [10., 10., 25., 25.] # taxinet_pwa_map_full.jld  +- 10 meters, +- 25 degrees
	return A, b
end

# Returns H-rep of various output sets
function output_constraints_taxinet()
	A = [1. 0.; -1. 0.; 0. 1.; 0. -1.]
	b = [1., 1., 1., 1.]
 	return A, b
end


# Plot all polytopes
function plot_hrep_taxinet(ap2constraints; type="normal")
	plt = plot(reuse = false, legend=false, xlabel="x₁", ylabel="x₂")
	for ap in keys(ap2constraints)
		A, b = ap2constraints[ap]
		reg = HPolytope(A, b)
	
		if isempty(reg)
			@show reg
			error("Empty polyhedron.")
		end

		if type == "normal"
			plot!(plt,  reg, fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
		elseif type == "gif"
			plot!(plt,  reg, xlims=(-3, 3), ylims=(-3, 3), fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
		end
	end
	return plt
end



###########################
######## SCRIPTING ########
###########################


# Load in saved function #
pwa_dict = load("models/taxinet/taxinet_pwa_map.jld2")
ap2input = pwa_dict["ap2input"]
plt_in1 = plot_hrep_taxinet(ap2input)



# Or solve from scratch:
# Given a network representing discrete-time autonomous dynamics and state constraints,
# ⋅ find fixed points
# ⋅ verify the fixed points are stable equilibria
# ⋅ compute invariant polytopes around the fixed points
# ⋅ perform backwards reachability to estimate the maximal region of attraction in the domain

# load network and domain constraints
# copies = 1 # copies = 1 is original network
# weights = taxinet_cl(copies)

# Aᵢ, bᵢ = input_constraints_taxinet()
# Aₒ, bₒ = output_constraints_taxinet()

# Load fixed point and ROA
# fp = [-1.231008720985839, 0.29785040070928537]
# A_roa = [-1.8983520194303938 37.509406817395785; -7.009461824024306 -7.450182394356705; 44.025034229240774 39.844054302383746; -27.947559360616175 -12.42511925401813; -27.63319921521314 23.442616936278394; 0.9015223818050797 -4.849875664991969; 43.51363803842881 41.92687778397991; 31.062203435580233 43.24579088827325; -0.5071988145561469 -5.264280599457316; 32.46117602704302 42.69801279908387; 6.915225467252936 -3.2643737649364564; -1.357715851928735 -5.5311707060289566; 5.768699221485058 -3.5833680040031406; -34.03458642377201 20.18603138815306; 10.073134723915624 -2.4304342556831044; -6.537649840415557 34.66751818766581; 2.8695803343533655 -4.331704651553559; 38.1153316855168 41.97923864268696; 0.7846838757510037 -4.8867746153247955; 21.750511616023335 43.46050208438424; -3.617732747965628 -6.222827064131699; -9.083833436574343 32.944036624344626; 0.06491977165667807 -0.007369974395176735; -10.23527126906095 32.38001202125704; 61.35313552482967 29.199889515225358; -19.419765310872204 -10.639136735099052; -43.88525871789746 15.28087807829831; 2.716213741768975 -4.373750453209372; -33.33970054313621 20.53707882837384; 4.595841591682565 -3.8780254079493917; 57.78088329548463 33.548149636984895; 6.66152531340165 40.89210645662412; -7.694341829943389 -7.605893867643146; -87.41145957707782 -5.886877098246038; 6.2837349740544015 -3.4430336940489736; -39.40351532765211 17.539452594229925; 0.1158242653372329 0.02057995645049188; 10.076293651259903 41.36207548187913; 1.066655936194925 -4.808274084523969; 51.79022457052009 40.45541682882823; -28.217763175038247 23.0668953870438; -1.0774759678063235 -5.444373778677145; 0.3137887376186603 -5.017566458201944; -4.313731620540087 36.787428705937494; 53.28135257589212 38.92074744024701; 37.08940847759919 41.960474727664526; -61.586241890047035 6.582357973871666; -9.31468555003701 -8.007368982370945; -76.94579396238674 -0.8320244087562776; -23.411125971713098 25.50664593135508; 59.06605382169619 9.112082823859868; -105.50023543667292 -22.36364651357535; 0.0 -5.128751842906363; -31.481203931623956 21.705656166404804; -111.0169967995597 -17.149664991152655; 66.78668386564645 22.572986219565596]
# b_roa = [14.509079742348828, 7.409668823274817, -41.327633538047934, 31.702862554517967, 41.99910207081121, -1.554319324374048, -40.07773055867934, -24.357067177173406, 0.056398077997817375, -26.24237056104908, -8.484997891722939, 1.0238986431628052, -7.168626646273493, 48.909290240501186, -12.1240225097582, 19.37363815209593, -3.822678383377955, -33.416772656944026, -1.421480471616908, -12.83034152436794, 3.599989028345539, 21.9946726899482, 0.9178880450993995, 23.24410774937271, -65.82904609811467, 21.737029317431823, 59.57455186326248, -3.646406129270016, 48.15843928208631, -5.812592501349194, -60.13644143057554, 4.979334538086896, 8.206383358676643, 106.85086035084504, -7.76084151885275, 54.73020398925182, 0.8635490675434472, 0.9157053965302246, -1.7452091225428035, -50.704555994797644, 42.606796589335296, 0.7047734014321274, -0.8807608527166266, 17.267391626004972, -52.997249465862026, -32.15944108054399, 78.77375881952094, 10.081461085310046, 95.47312460732964, 37.41646495068087, -68.99678984834408, 124.21068881321439, -0.527600791548146, 46.21867497378441, 132.55485664827015, -74.49161729363222]

# fp = [-1.089927713157323, -0.12567755953751042]
# A_roa = 370*[-0.20814568962857855 0.03271955855771795; 0.2183098663000297 0.12073669880754853; 0.42582101825227686 0.0789033995762251; 0.14480530852927057 -0.05205811047518554; -0.13634673812819695 -0.1155315084750385; 0.04492020060602461 0.09045775648816877; -0.6124506220873154 -0.12811621541510643]
# b_roa = ones(size(A_roa,1)) + A_roa*fp
# reg_roa = HPolytope(A_roa, b_roa)

# Run algorithm
# @time begin
# ap2input, ap2output, ap2map, ap2backward, ap2neighbors = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], graph=true)
# # ap2input, ap2output, ap2map, ap2backward = compute_reach(weights, Aᵢ, bᵢ, [A_roa], [b_roa], fp=fp, reach=false, back=true, connected=true)
# end
# @show length(ap2input)
# @show length(ap2backward[1])


# save pwa map
# save("models/taxinet/1Hz_2nd/pwa_map.jld2", Dict("ap2map" => ap2map, "ap2input" => ap2input, "ap2neighbors" => ap2neighbors, "Aᵢ" => Aᵢ, "bᵢ" => bᵢ))
# save("models/taxinet/1Hz_2nd/taxinet_brs_0_step_overlap.jld2", Dict("brs" => aa))


# Plot all regions #
# plt_in1 = plot_hrep_taxinet(ap2input)
# scatter!(plt_in1, [fp[1]], [fp[2]])
# plot!(plt_in1, reg_roa)

# plt_in2  = plot_hrep_taxinet(ap2backward[1])

# net_dict = load("models/taxinet/taxinet_pwa_map_5_15.jld2")
# ap2map = net_dict["ap2map"]
# ap2input = net_dict["ap2input"]


# determine if function is a homeomorphism
# @time begin
# homeomorph = is_homeomorphism(ap2map, size(Aᵢ,2))
# end
# println("PWA function is a homeomorphism: ", homeomorph)


# find any fixed points if they exist
# @time begin
# fixed_points, fp_dict = find_fixed_points(ap2map, ap2input, weights)
# end
# @show fixed_points


# find an attractive fixed point and return it's affine function y=Cx+d and polytope Aₓx≤bₓ
# A1, b1 =  fp_dict[fixed_points[1]][1]
# C1, d1 = fp_dict[fixed_points[1]][2]
# fp2, C2, d2, A2, b2 = find_attractor([fixed_points[2]], fp_dict)
# fp3, C3, d3, A3, b3 = find_attractor([fixed_points[3]], fp_dict)

# plt = plot(reuse = false, xlabel="x₁", ylabel="x₂")
# scatter!(plt, [fp2[1]], [fp2[2]], label="fp2")
# scatter!(plt, [fp3[1]], [fp3[2]], label="fp3")
# plot!(HPolytope(A1, b1), label="unstable")
# plot!(HPolytope(A2, b2), label="stable")
# plot!(HPolytope(A3, b3), label="stable")

# @time begin
# n_constraints = 50
# A1_roa, b1_roa = polytope_roa_sdp(A1, b1, C1, fixed_points[1], n_constraints)
# end
# A3_roa, b3_roa = polytope_roa_sdp(A3, b3, C3, fp3, n_constraints)

# plot!(HPolytope(A2_roa, b2_roa), label="stable_roa")
# plot!(HPolytope(A3_roa, b3_roa), label="stable_roa")

# save("models/taxinet/1Hz_2nd/fp_info.jld2", Dict("stable fps" => fixed_points[1], "roas" => [(A1_roa, b1_roa)]))









# Getting mostly suboptimal SDP here
# A_roa, b_roa, fp, ap2backward_chain, plt_in2 = find_roa("taxinet", weights, 40, 1)
# @show plt_in2




