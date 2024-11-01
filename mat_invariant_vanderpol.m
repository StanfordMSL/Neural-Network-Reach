clear all 
clc

% Load in A,b, C,d cell arrays
load('models/vanderpol/vanderpol_pwa.mat');
load('models/vanderpol/vanderpol_seed.mat');
P_seed = Polyhedron(A_roa, b_roa);

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

tic
S = pwa.invariantSet('maxIterations', 200); % Can add arguments of 'X' (seed set) and maxIterations
toc

plot(S)
xlim([-2.5 2.5])
ylim([-3 3])
