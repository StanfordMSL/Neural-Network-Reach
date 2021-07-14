using FileIO, MAT, MATLAB

# pwa_dict = load("models/Pendulum/pendulum_pwa.jld2")
pwa_dict = load("models/vanderpol/vanderpol_pwa.jld2")

state2input = pwa_dict["state2input"]
state2map = pwa_dict["state2map"]
net_dict = pwa_dict["net_dict"]
A_norm_in, b_norm_in = net_dict["input_norm_map"]
A_norm_out, b_norm_out = net_dict["output_unnorm_map"]


"""
Each region has an affine map ȳ = C*x̄ + d given by state2map
We have that x̄ = A_norm_in*x + b_norm_in
⟹			 ȳ = C*A_norm_in*x + C*b_norm_in + d
and that     y = A_norm_out*ȳ + b_norm_out
⟹     		 y = A_norm_out*C*A_norm_in*x + A_norm_out*C*b_norm_in + A_norm_out*d + b_norm_out
⟹            C_eff = A_norm_out*C*A_norm_in
			 d_eff = A_norm_out*C*b_norm_in + A_norm_out*d + b_norm_out

Each region has polytopic constraints A*x̄ ≤ b
We have that x̄ = A_norm_in*x + b_norm_in
⟹			 A*A_norm_in*x ≤ b - A*b_norm_in
⟹            A_eff = A*A_norm_in
			 b_eff = b - A*b_norm_in
"""


# Export PWA function to matlab
A_dat = Vector{Matrix{Float64}}(undef, length(state2input))
b_dat = Vector{Vector{Float64}}(undef, length(state2input))
C_dat = Vector{Matrix{Float64}}(undef, length(state2input))
d_dat = Vector{Vector{Float64}}(undef, length(state2input))

for (i,key) in enumerate(keys(state2input))
	C, d = state2map[key]
	C_dat[i], d_dat[i] = A_norm_out*C*A_norm_in, A_norm_out*C*b_norm_in + A_norm_out*d + b_norm_out

	A, b = state2input[key]
	A_dat[i], b_dat[i] = A*A_norm_in, b - A*b_norm_in
end


A = mxcellarray(A_dat)  # creates a MATLAB cell array
b = mxcellarray(b_dat)  # creates a MATLAB cell array
C = mxcellarray(C_dat)  # creates a MATLAB cell array
d = mxcellarray(d_dat)  # creates a MATLAB cell array
write_matfile("vanderpol_pwa.mat"; A = A, b = b, C = C, d = d)
