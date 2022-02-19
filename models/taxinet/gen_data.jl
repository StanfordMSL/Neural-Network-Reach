using NPZ, LinearAlgebra

# generates a uniformly random number on [a,b]
bound_r(a,b) = (b-a)*(rand()-1) + b

### Generate data like in SISL paper ##
#=
from https://arxiv.org/pdf/2105.07091.pdf Eq. 8
x = [p, θ] where p in meters and θ in degrees
=#
function dynamics(x, u; dt=0.05)
	v, L= 5, 5
	x′ = [x[1] + v*sind(x[2])*dt, x[2] + rad2deg((v/L)*tand(u))*dt]
	return x′
end

# generate input data
function gen_data(n::Int)
	X, Y = Matrix{Float64}(undef, n, 3), Matrix{Float64}(undef, n, 2)
	for i in 1:n
		X[i,:] = [bound_r(-10., 10.), bound_r(-11.,11.), bound_r(-30.,30.)] # [u; x] input bounds
		Y[i,:] = dynamics(X[i,2:3], X[i,1])
	end
	npzwrite("models/taxinet/X_dynamics_10hz.npy", X)
	npzwrite("models/taxinet/Y_dynamics_10hz.npy", Y)
	return nothing
end


## Generate data with RK integration ##
RK_f(S, u) = [5*sind(S[2]), rad2deg((5/5)*tand(u))] # continuous time dynamics assuming constant u over the time period. 

function RK_update(S, u, dt)
	k1 = RK_f(S, u)
	k2 = RK_f(S + dt*0.5*k1, u)
	k3 = RK_f(S + dt*0.5*k2, u)
	k4 = RK_f(S + dt*k3, u)
	return S + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
end

# generate input data
function gen_data_RK(n::Int)
	X, Y = Matrix{Float64}(undef, n, 3), Matrix{Float64}(undef, n, 2)
	for i in 1:n
		X[i,:] = [bound_r(-20., 20.), bound_r(-11.,11.), bound_r(-30.,30.)] # [u; x] input bounds
		Y[i,:] = RK_update(X[i,2:3], X[i,1], 0.2)
	end
	npzwrite("models/taxinet/X_dynamics_5hz.npy", X)
	npzwrite("models/taxinet/Y_dynamics_5hz.npy", Y)
	return nothing
end

