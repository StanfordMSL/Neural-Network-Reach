clear all 
clc

% Q = Polyhedron([1 -1;0.5 -2;-1 0.4; -1 -2],[1;2;3;4])
% 
% sys1 = LTISystem('A', 0.5, 'f', 0);
% sys1.setDomain('x', Polyhedron('lb',0, 'ub', 1));
% 
% sys2 = LTISystem('A', -0.5, 'f', 0);
% sys2.setDomain('x', Polyhedron('lb',-1, 'ub', 0));
% dd = [sys1, sys2];
% 
% for iter = 1:5
%     array(iter) = sys1;
% end

% 
% 
% pwa = PWASystem([sys1, sys2])
% S = pwa.invariantSet()

% Load in A,b, C,d cell arrays
load('region_dat.mat');

num_regions = length(A);
for i = 1:num_regions
    P_i = Polyhedron(A{i}, b{i});
    C_i = C{i};
    d_i = d{i};
    system_i = LTISystem('A', C_i, 'f', d_i);
    system_i.setDomain('x', P_i);
    systems(i) = system_i;
end

pwa = PWASystem(systems);
S = pwa.invariantSet();