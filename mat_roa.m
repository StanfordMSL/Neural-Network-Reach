clear all 
clc

% Example code from MPT3 Wiki
% model = LTISystem('A', [1 1; 0 1], 'B', [1; 0.5]);
% model.x.min = [-5; -5];
% model.x.max = [5; 5];
% model.u.min = -1;
% model.u.max = 1;
% model.x.penalty = QuadFunction(eye(2));
% model.u.penalty = QuadFunction(1);
% % add LQR terminal set and terminal penalty
% model.x.with('terminalSet');
% model.x.terminalSet = model.LQRSet();
% model.x.with('terminalPenalty');
% model.x.terminalPenalty = model.LQRPenalty();
% 
% N = 5; % prediction horizon
% empc = MPCController(model, N).toExplicit();
% 
% % create a closed-loop system
% loop = ClosedLoop(empc, model);
% 
% % convert the closed-loop system into an autonomous PWA system
% autpwa = loop.toSystem();
% 
% % construct the PWQ Lyapunov function
% L = autpwa.lyapunov('pwq');
% 
% % plot the PWQ Lyapunov function
% L.fplot('lyapunov');ly




% My code
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

% construct the PWQ Lyapunov function
L = pwa.lyapunov('pwq');

% plot the PWQ Lyapunov function
L.fplot('lyapunov');
