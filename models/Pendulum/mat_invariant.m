clear all 
clc

% Load in A,b, C,d cell arrays
load('pendulum_controlled_pwa.mat');
X = Polyhedron('lb', [-pi/3; -pi/6], 'ub', [pi/3; pi/6]);
U = Polyhedron('lb', -1, 'ub', 1);

% Constructor for PWA systems
%
%   x^+ = A{i}*x + B{i}*u + f{i}   if   [x; u] \in P(i)
%     y = C{i}*x + D{i}*u + g{i}
% 
% s = PWASystem('A', A, 'B', B, 'C', C, 'D', D, 'f', f, 'domain', P)

num_regions = length(A);
% dyn_A = cell(num_regions);
% dyn_B = cell(num_regions);
% dyn_f = cell(num_regions);
% dyn_P = cell(num_regions);
% systems = [];
for i = 1:num_regions
%     dyn_A{i} = C{i}(:,1:2);
%     dyn_B{i} = C{i}(:,3);
%     dyn_f{i} = d{i};
%     dyn_P{i} = Polyhedron(A{i}, b{i});
    
    systems(i) = LTISystem('A', C{i}(:,1:2), 'B', C{i}(:,3), 'f', d{i}, 'domain', Polyhedron(A{i}, b{i}));
%     systems(i) = system_i;
end

% pwa = PWASystem('A', dyn_A, 'B', dyn_B, 'f', dyn_f, 'domain', dyn_P);
pwa = PWASystem(systems);
S = pwa.invariantSet('X', X, 'U', U, 'maxIterations', 20);
plot(S)
xlabel("Angle (rad)")
ylabel("Angular Velocity (rad/s)")
% xlim([-2.5 2.5])
% ylim([-3 3])

% Save concactenated Ab H-rep of the control invariant set
Ab = S.H;
save("cntrl_invariant.mat", 'Ab');




