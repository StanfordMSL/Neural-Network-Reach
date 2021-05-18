using Plots, MATLAB
include("reach.jl")
include("invariance.jl")

# Returns H-rep of various input sets
function input_constraints_pendulum(weights, type::String; net_dict=[])
	if type == "pendulum"
		# Square. ⨦1 rad , ⨦ 1 rad/s
		A = [1 0; -1 0; 0 1; 0 -1]
		b = [deg2rad(90), deg2rad(90), deg2rad(90), deg2rad(90)]
		C = Float64.(Diagonal(vec(net_dict["X_std"])))
		d = vec(net_dict["X_mean"])
		Aᵢ = A*C
		bᵢ = b - A*d
	elseif type == "box"
		in_dim = size(weights[1],2) - 1
		Aᵢ_pos = Matrix{Float64}(I, in_dim, in_dim)
		Aᵢ_neg = Matrix{Float64}(-I, in_dim, in_dim)
		Aᵢ = vcat(Aᵢ_pos, Aᵢ_neg)
		bᵢ = 0.01*ones(2*in_dim)
	elseif type == "hexagon"
		Aᵢ = [1 0; -1 0; 0 1; 0 -1; 1 1; -1 1; 1 -1; -1 -1]
		bᵢ = [5, 5, 5, 5, 8, 8, 8, 8]
	else
		error("Invalid input constraint specification.")
	end
	return Aᵢ, bᵢ
end

# Returns H-rep of various output sets
function output_constraints_pendulum(weights, type::String; net_dict=[])
	if type == "origin"
		A = [1 0; -1 0; 0 1; 0 -1]
		b = [5, 5, 2, 2]
		σ = Diagonal(vec(net_dict["Y_std"]))
		μ = vec(net_dict["Y_mean"])
		Aₒ = (180/π)*A*σ
		bₒ = b - (180/π)*A*μ
 	else 
 		error("Invalid input constraint specification.")
 	end
 	return Aₒ, bₒ
end


# Plot all polytopes
function plot_hrep_pendulum(state2constraints, net_dict; space = "input")
	plt = plot(reuse = false, legend=false, xlabel="Angle (deg.)", ylabel="Angular Velocity (deg./s.)")
	for state in keys(state2constraints)
		A, b = state2constraints[state]
		if space == "input"				
			C = Diagonal(vec(net_dict["X_std"]))
			d = vec(net_dict["X_mean"])
		elseif space == "output"
			C = Diagonal(vec(net_dict["Y_std"]))
			d = vec(net_dict["Y_mean"])
		else
			error("Invalid arg given for space")
		end
		temp_reg = HPolytope(constraints_list(A,b))
		if isempty(temp_reg) || isempty(Float64.(C)*temp_reg + Float64.(d))
			println("Empty polyhedron.")
		else
			reg = Float64.(C)*temp_reg + Float64.(d)
			reg = (180/π)*reg # Convert from rad to deg
			# @show reg
			plot!(plt,  reg, fontfamily=font(40, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
		end
		
	end
	return plt
end



###########################
######## SCRIPTING ########
###########################
copies = 0 # copies = 0 is original network
model = "models/Pendulum/NN_params_pendulum_0_1s_1e7data_a15_12_2_L1.mat"

weights, net_dict = pendulum_net(model, copies)
Aᵢ, bᵢ = input_constraints_pendulum(weights, "pendulum", net_dict=net_dict)
Aₒ, bₒ = output_constraints_pendulum(weights, "origin", net_dict=net_dict)
# Aₒ, bₒ = A_, b_

# Run algorithm
# @time begin
# state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], reach=false, back=false, verification=false)
# end
# @show length(state2input)
# @show length(state2backward[1])

# Plot all regions #
# plt_in1  = plot_hrep_pendulum(state2input, net_dict, space="input")
# plt_in2  = plot_hrep_pendulum(state2backward[1], net_dict, space="input")
# plt_out = plot_hrep_pendulum(state2output, net_dict, space="output")


# Fixed point stuff
C_in = Float64.(Diagonal(vec(1 ./ net_dict["X_std"])))
d_in = vec(net_dict["X_mean"])
C_out = Diagonal(vec(net_dict["Y_std"]))
d_out = vec(net_dict["Y_mean"])


fp = [-0.001989079271247683, 0.017466775805123395]
eval_net(fp, net_dict, 0)


weights, net_dict = pendulum_net(model, 0)
C, d = local_map(get_state(fp, weights), weights)
G = svd(C) # U, S, V = G

aa = C_out*C*C_in

# for t in 1:200
# 	# weights, net_dict = pendulum_net(model, t-1)
# 	# C, d = local_map(get_state(fp, weights), weights)
# 	F = eigen(C^t)
# 	println(t, "-step eigenvalues: ", norm.(F.values))
# end


# fixed_points = find_fixed_points(state2map) # Fixed point =  [-0.001989079271247683, 0.017466775805123395]
# i, state, A_, b_ = i_step_invariance(fixed_points[1], 1)

# plt_seed = plot_hrep_pendulum(Dict(state => (A_, b_)), net_dict; space = "input")






























# # Export to matlab data
# A_dat = Vector{Matrix{Float64}}(undef, length(state2input))
# b_dat = Vector{Vector{Float64}}(undef, length(state2input))
# C_dat = Vector{Matrix{Float64}}(undef, length(state2input))
# d_dat = Vector{Vector{Float64}}(undef, length(state2input))

# for (i,key) in enumerate(keys(state2input))
# 	C_dat[i], d_dat[i] = state2map[key]
# 	A_dat[i], b_dat[i] = state2input[key]
# end


# A = mxcellarray(A_dat)  # creates a MATLAB cell array
# b = mxcellarray(b_dat)  # creates a MATLAB cell array
# C = mxcellarray(C_dat)  # creates a MATLAB cell array
# d = mxcellarray(d_dat)  # creates a MATLAB cell array
# write_matfile("region_dat.mat"; A = A, b = b, C = C, d = d)



# mat"""
# Q = Polyhedron([1 -1;0.5 -2;-1 0.4; -1 -2],[1;2;3;4])

# sys1 = LTISystem('A', 0.5, 'f', 0);
# sys1.setDomain('x', Polyhedron('lb',0, 'ub', 1));

# sys2 = LTISystem('A', -0.5, 'f', 0);
# sys2.setDomain('x', Polyhedron('lb',-1, 'ub', 0));

# bb = 33


# pwa = PWASystem([sys1, sys2])
# S = pwa.invariantSet()
# size(S)
# $ii

# """












# for region in regions, search each one for locally invariant set
# for ap in keys(state2map)
# 	C, d = state2map[ap]
# 	A, b = state2input[ap]
# 	println(rank(I - C))
# 	p = (I - C) \ d
# 	inside = true
# 	for i in 1:length(b)
# 		A[i,:]⋅p > b[i] ? inside = false : nothing
# 	end
# 	if inside
# 		@show p
# 		scatter!(plt_in1, [p[1]], [p[2]])
# 	end
# end


# fixed_point = [-0.001989079271247683, 0.0174667758051234] # in the normalized input space

# Then compose network 50 times and find all inputs that lead to the fixed point
# for region in regions, search each one for locally invariant set
# for ap in keys(state2map)
# 	feasible = false
# 	C, d = state2map[ap]
# 	A, b = state2input[ap]
# 	rank(C) > 2 ? println(rank(C)) : nothing

# 	# solve feasibility LP
# 	model = Model(GLPK.Optimizer)
# 	@variable(model, x[1:size(C,2)])
# 	@objective(model, Max, 0)
# 	@constraint(model, A*x .≤ b)
# 	@constraint(model, C*x + d .== fixed_point)
# 	optimize!(model)
# 	if termination_status(model) == MOI.OPTIMAL
# 		A_ = vcat(A, C, -C)
# 		b_ = vcat(b, d, -d)
# 		state2invariant[ap] = (A_, b_) # feasible
# 	end
# end



# plt_in3  = plot_hrep_pendulum(state2invariant, net_dict, space="input")






















## mp-LP ##


# mat"""
# Q = Polyhedron([1 -1;0.5 -2;-1 0.4; -1 -2],[1;2;3;4])

# sys1 = LTISystem('A', 1, 'f', 1);
# sys1.setDomain('x', Polyhedron('lb',0));

# sys2 = LTISystem('A', -2, 'f', 1);
# sys2.setDomain('x', Polyhedron('ub',0));

# pwa = PWASystem([sys1,sys2]);
# S = pwa.invariantSet();
# """






# mat"""
# % printing parameters
# label_font_size = 14;
# tick_font_size = 10;
# line_width = 0.8;
# axeswidth=0.2;
# figure_name = ['.',filesep,'figures',filesep,'ex5_2'];
# % use laprint to produce pictures
# set(0,'defaulttextinterpreter','none')
# width = 10;

# % core code
# % matrices of MPLP formulation G*x <= w + S*th
# G=zeros(12,4);
# w=zeros(12,1);
# S=zeros(12,2);
# G(1,1)=-1;
# G(2,1)=-1;
# G(2,3)=-1;
# G(3,1)=-1;
# G(4,1)=-1;
# G(4,3)=+1;
# G(5,2:3)=[-1 -1];
# G(6,2:4)=[-1 -1 -1];
# G(7,2:3)=[-1 1];
# G(8,2:4)=[-1 1 1];
# G(9,3)=1;
# G(10,3)=-1;
# G(11,4)=1;
# G(12,4)=-1;
# w(9:12)=[1;1;1;1];
# S(1,:)=[1 1];
# S(2,:)=[0 1];
# S(3,:)=[-1 -1];
# S(4,:)=[0 -1];
# S(5,:)=[1 2];
# S(6,:)=[0 1];
# S(7,:)=[-1 -2];
# S(8,:)=[0 -1];
# Ath = [eye(2); -eye(2)]; bth = 2.5*ones(4,1); 

# % formulate MPLP, min f'*x s.t. A*x <= b + pB*th, Ath*th <= bth
# problem = Opt('f',[1 1 0 0],'A',G,'b',w,'pB',S,'Ath',Ath,'bth',bth);
# solution=problem.solve; 

# % plot partition
# figure 
# hold on
# plot(solution.xopt,'linewidth',0.8,'wirestyle','--','color',[0.9 0.9 0.9])
# grid on

# % list active sets
# for j=1:length(solution.xopt.Set)
#     % get a point inside each polytope
#     chebC=solution.xopt.Set(j).chebyCenter;
    
#     % solve optimization problem for a fixed point to get the active set
#     problem_fixed = Opt('f',[1 1 0 0],'A',G,'b',w+S*chebC.x);
#     solution_fixed=problem_fixed.solve;
    
#     % display active set
# %     disp(['Active set in Region ',num2str(j),' is:'])
# %     ActiveSet=find(G*solution_fixed.xopt-(w+S*chebC.x)>=-1e-10);
# %     ActiveSet'
#     % nonzero Lagrange multpliers for inequality constraints 
# %     lam = solution.xopt.feval(chebC.x, 'dual-ineqlin')';
# %     find(lam >= 1e-10)
    
#     % cost function is stored as x = F*th + g
# %     disp(['Cost function is in Region ',num2str(j),' is:'])
# %     [solution.xopt.Set(j).getFunction('obj').F solution.xopt.Set(j).getFunction('obj').g]
#     ht3 = text(chebC.x(1)-0.1,chebC.x(2),num2str(j));
#     set(ht3, 'FontSize', label_font_size);
# end


# % save book figures
# % font size of tick labels
# set(gca,'LineWidth',axeswidth)
# set(gca,'FontSize', tick_font_size);

# title('')
# hx=xlabel('x_1');
# set(hx, 'FontSize', label_font_size);
# hy=ylabel('x_2');
# set(hy, 'FontSize', label_font_size);


# xl = transpose([-2.5,2.5]);
# yl = transpose([-2.5,2.5]);
# set(gca,'XTick',xl);
# set(gca,'YTick',yl);
# set(gca,'XTickLabel',num2str(xl));
# set(gca,'YTickLabel',num2str(yl));
# """