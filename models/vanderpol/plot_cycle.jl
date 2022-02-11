using Plots

# van der Pol Oscillator #
# dynamics from http://control.asu.edu/Publications/2018/Colbert_CDC_2018.pdf


# RK_f(S; γ=1., ϵ=-1., ω=1.) = [S[2], (-γ - ϵ*S[1]^2)*S[2] + ω^2*S[1]] # forward time

RK_f(S) = [-S[2], S[1] + S[2]*(S[1]^2 - 1)] # reverse time. ROA is a nonconvex subset of a square of +- 3 around the origin


function RK_update(S, dt)
	k1 = RK_f(S)
	k2 = RK_f(S + dt*0.5*k1)
	k3 = RK_f(S + dt*0.5*k2)
	k4 = RK_f(S + dt*k3)
	return S + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
end


function rollout(x0, steps, dt)
	traj = Matrix{Float64}(undef, steps, 2)
	traj[1,:] = x0
	for i in 2:steps
		traj[i,:] = RK_update(traj[i-1,:], dt)
	end
	return traj
end


traj = rollout([-2.0086212, 0.0], 135, 0.05)
plt = plot(traj[:,1], traj[:,2], xlims=(-2.5, 2.5), ylims=(-3, 3), label=false)