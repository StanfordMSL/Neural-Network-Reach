using MATLAB, FileIO, Plots, LazySets

function plot_hrep(inv_set)
	plt = plot(reuse = false, legend=false, xlabel="Angle (deg.)", ylabel="Angular Velocity (deg./s.)")
	for (A,b) in inv_set
		reg = (180/π)*HPolytope(constraints_list(A,b)) # Convert from rad to deg for plotting
		
		if isempty(reg)
			@show reg
			error("Empty polyhedron.")
		end
		plot!(plt, reg, fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
	end
	return plt
end

#=
interface with Matlab MPT to find control invariant set
=#
function ctrl_inv(max_iters)
	mat"""
	addpath(genpath('../tbxmanager/toolboxes'));
	temp = mptopt('lpsolver', 'GLPK');

	load('models/Pendulum/pendulum_controlled_pwa.mat'); % loads in A,b, C,d cell arrays
	X = Polyhedron('lb', [-pi; -pi], 'ub', [pi; pi]);
	U = Polyhedron('lb', -5, 'ub', 5);

	num_regions = length(A);
	for i = 1:num_regions
	    systems(i) = LTISystem('A', C{i}(:,1:2), 'B', C{i}(:,3), 'f', d{i}, 'domain', Polyhedron(A{i}, b{i}));
	end

	pwa = PWASystem(systems);
	tic;
	S = pwa.invariantSet('X', X, 'U', U, 'maxIterations', $max_iters);
	t = toc;
	disp(toc)
	disp(t)

	$len_S = length(S);
	$As = cell($len_S,1);
	$bs = cell($len_S,1);

	for i = 1:$len_S
		$As{i,1} = S(i).A;
	 	$bs{i,1} = S(i).b;
	end
	"""

	# Send polytopes to Julia variables
	println("Ended with ", length(As), " polytopes")
	inv_set = Set{ Tuple{Matrix{Float64}, Vector{Float64}} }()
	for i in 1:length(As)
		push!(inv_set, (As[i], bs[i]))
	end

	return inv_set
end


# inv_set = ctrl_inv(100)
# save("models/Pendulum/pendulum_controlled_inv_set.jld2", Dict("inv_set" => inv_set) )

inv_dict = load("models/Pendulum/pendulum_controlled_inv_set.jld2",)
inv_set = inv_dict["inv_set"]

plt = plot_hrep(inv_set)




