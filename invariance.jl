function in_polytope(x, A, b)
	for i in 1:length(b)
		A[i,:]⋅x > b[i] ? (return false) : nothing
	end
	return true
end

function compute_traj(init, steps::Int64, net_dict)
	t = collect(0:0.1:0.1*steps) # make sure this is the right dt
	state_traj = zeros(2,length(t))
	state_traj[:,1] = init
	for i in 2:length(t)
		state_traj[:,i] = eval_net(state_traj[:,i-1], net_dict, 0)
	end
	return state_traj
end

function plot_traj(state_traj)
	plt = plot(reuse = false)
	plot!(plt, t, rad2deg.(state_traj[1,:]), linewidth=3, legend=false, xlabel="Time (s.)", ylabel="Angle (deg.)", fontfamily=font(14, "Computer Modern"), yguidefont=(14) , xguidefont=(14), tickfont = (12))
	return plt
end

function find_fixed_points(state2map)
	fixed_points = Vector{Vector{Float64}}(undef, 0)

	for ap in keys(state2map)
		C, d = state2map[ap]
		A, b = state2input[ap]
		p = inv(I - C) * d # make more general for rank deficient cases
		if in_polytope(p, A, b)
			push!(fixed_points, p) # fixed
		end
	end
	return fixed_points
end

# Find an i-step invariant set given a fixed point p
function i_step_invariance(fixed_point, max_steps)
	for i in 0:max_steps
		println("\n", i+1)
		# load neural network
		copies = i # copies = 0 is original network
		model = "models/Pendulum/NN_params_pendulum_0_1s_1e7data_a15_12_2_L1.mat"
		weights, net_dict = pendulum_net(model, copies)
		state = get_state(fixed_point, weights)
		num_neurons = sum([length(state[layer]) for layer in 1:length(state)])
		
		# get Cx+d, Ax≤b
		C, d = local_map(state, weights)
		println("opnorm(C'C): ", opnorm(C'C))
		F = eigen(C)
		println("eigen(C): ", F.values)
		A, b, nothing, nothing, unique_nonzerow_indices = get_constraints(weights, state, num_neurons)
		A, b, nothing, nothing, nothing = remove_redundant(A, b, [], [], unique_nonzerow_indices, [])

		# get forward reachable set
		Af, bf = affine_map(A, b, C, d)

		# check if forward reachable set is a strict subset of Ax≤b
		unique_nonzerow_indices = 1:(length(b) + length(bf)) # all indices unique & nonzero
		A_, b_, nothing, nothing, nothing = remove_redundant(vcat(A, Af), vcat(b, bf), [], [], unique_nonzerow_indices, [])
		if A_ == Af && b_ == bf
			println(i+1, "-step invariant set found!")
			return i, state, A_, b_
		end
	end
	return nothing, nothing, nothing, nothing
end






















## mp-LP ##








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