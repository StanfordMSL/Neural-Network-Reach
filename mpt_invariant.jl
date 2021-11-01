using FileIO, MAT, MATLAB

pwa_dict = load("models/Pendulum/pendulum_controlled_pwa.jld2")
# pwa_dict = load("models/vanderpol/vanderpol_pwa.jld2")

state2input = pwa_dict["state2input"]
state2map = pwa_dict["state2map"]

# Export PWA function to matlab
A_dat = Vector{Matrix{Float64}}(undef, length(state2input))
b_dat = Vector{Vector{Float64}}(undef, length(state2input))
C_dat = Vector{Matrix{Float64}}(undef, length(state2input))
d_dat = Vector{Vector{Float64}}(undef, length(state2input))

for (i,key) in enumerate(keys(state2input))
	C_dat[i], d_dat[i] = state2map[key]
	A_dat[i], b_dat[i] = state2input[key]
end


A = mxcellarray(A_dat)  # creates a MATLAB cell array
b = mxcellarray(b_dat)  # creates a MATLAB cell array
C = mxcellarray(C_dat)  # creates a MATLAB cell array
d = mxcellarray(d_dat)  # creates a MATLAB cell array
write_matfile("models/Pendulum/pendulum_controlled_pwa.mat"; A = A, b = b, C = C, d = d)
