using FileIO, MAT, MATLAB

# pwa_dict = load("models/Pendulum/pendulum_controlled_pwa.jld2")
# pwa_dict = load("models/vanderpol/vanderpol_pwa.jld2")
pwa_dict = load("models/quadratic/quadratic_pwa.jld2")

ap2input = pwa_dict["ap2input"]
ap2map = pwa_dict["ap2map"]

# Export PWA function to matlab
A_dat = Vector{Matrix{Float64}}(undef, length(ap2input))
b_dat = Vector{Vector{Float64}}(undef, length(ap2input))
C_dat = Vector{Matrix{Float64}}(undef, length(ap2input))
d_dat = Vector{Vector{Float64}}(undef, length(ap2input))

for (i,key) in enumerate(keys(ap2input))
	C_dat[i], d_dat[i] = ap2map[key]
	A_dat[i], b_dat[i] = ap2input[key]
end


A = mxcellarray(A_dat)  # creates a MATLAB cell array
b = mxcellarray(b_dat)  # creates a MATLAB cell array
C = mxcellarray(C_dat)  # creates a MATLAB cell array
d = mxcellarray(d_dat)  # creates a MATLAB cell array

write_matfile("models/quadratic/quadratic_pwa.mat"; A = A, b = b, C = C, d = d)
