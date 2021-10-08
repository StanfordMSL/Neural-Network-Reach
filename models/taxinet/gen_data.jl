using NPZ, LinearAlgebra

```generates a uniformly random number on [a,b]```
bound_r(a,b) = (b-a)*(rand()-1) + b

```
from https://arxiv.org/pdf/2105.07091.pdf Eq. 8
x = [p, θ] where p in meters and θ in degrees
```
function dynamics(x; dt=0.05)
	v, L= 5, 5
	u = [-0.74, -0.44]⋅x
	x′ = [x[1] + v*sind(x[2]*dt), x[2] + (v/L)*tand(u)*dt]
	return x′
end

```generate input data```
function gen_data(n::Int)
	X, Y = Matrix{Float64}(undef, n, 2), Matrix{Float64}(undef, n, 2)
	for i in 1:n
		X[i,:] = [bound_r(-11.,11.), bound_r(-30.,30.)] # input bounds
		Y[i,:] = dynamics(X[i,:])
	end
	npzwrite("models/taxinet/X_dynamics.npy", X)
	npzwrite("models/taxinet/Y_dynamics.npy", Y)
	return nothing
end