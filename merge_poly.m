clear all 
clc

temp = mptopt('lpsolver', 'GLPK');

% Load in A,b, C,d cell arrays
load('models/taxinet/1Hz_2nd/taxinet_brs_10_step.mat');

for i = 1:length(As_mat)
	P(i) = Polyhedron(As_mat{i}, bs_mat{i});
end
U = PolyUnion('Set',P,'convex',false,'overlaps',true,'Connected',true,'fulldim',true,'bounded',true);

merged = U.merge('optimal', false);
len_m = length(merged.Set);
As_merged = cell(len_m,1);
bs_merged = cell(len_m,1);

for i = 1:len_m
	As_merged{i,1} = merged.Set(i).A;
 	bs_merged{i,1} = merged.Set(i).b;
end

save("models/taxinet/1Hz_2nd/taxinet_brs_10_step_overlap.mat", "As_merged", "bs_merged")

