<<<<<<< Updated upstream
using NPZ, HDF5

f_train = h5open("models/taxinet/morning_train_downsampled.h5", "r")
f_test = h5open("models/taxinet/morning_test_downsampled.h5", "r")
f_val = h5open("models/taxinet/morning_validation_downsampled.h5", "r")

x_train = read(f_train["X_train"])
y_train = read(f_train["y_train"])

x_test = read(f_test["X_train"])
y_test = read(f_test["y_train"])

x_val = read(f_val["X_train"])
y_val = read(f_val["y_train"])

X = cat(x_train, x_test, x_val, dims=3)
X_flat = reshape(X, 16*8, 63141)
Y = cat(y_train, y_test, y_val, dims=2)
Y = Y[1:2,:]


println("Size x_train: ", size(x_train))
println("Size y_train: ", size(y_train))

println("Size x_test: ", size(x_test))
println("Size y_test: ", size(y_test))

println("Size x_val: ", size(x_val))
println("Size y_val: ", size(y_val))

println("Size X: ", size(X))
println("Size X_flat: ", size(reshape(X, 16*8, 63141)))
println("Size Y: ", size(Y))

# npzwrite("models/taxinet/X_image.npy", X_flat')
# npzwrite("models/taxinet/Y_image.npy", Y')

# save("data.jld2", Dict("X" => X, "Y" => Y))
# dat_dict = load("data.jld2")


using Colors, Plots
plt = plot(Gray.(X[:,:,1]))

=======
using NPZ, HDF5

f_train = h5open("models/taxinet/morning_train_downsampled.h5", "r")
f_test = h5open("models/taxinet/morning_test_downsampled.h5", "r")
f_val = h5open("models/taxinet/morning_validation_downsampled.h5", "r")

x_train = read(f_train["X_train"])
y_train = read(f_train["y_train"])

x_test = read(f_test["X_train"])
y_test = read(f_test["y_train"])

x_val = read(f_val["X_train"])
y_val = read(f_val["y_train"])

X = cat(x_train, x_test, x_val, dims=3)
X_flat = reshape(X, 16*8, 63141)
Y = cat(y_train, y_test, y_val, dims=2)
Y = Y[1:2,:]


println("Size x_train: ", size(x_train))
println("Size y_train: ", size(y_train))

println("Size x_test: ", size(x_test))
println("Size y_test: ", size(y_test))

println("Size x_val: ", size(x_val))
println("Size y_val: ", size(y_val))

println("Size X: ", size(X))
println("Size X_flat: ", size(reshape(X, 16*8, 63141)))
println("Size Y: ", size(Y))

# npzwrite("models/taxinet/X_image.npy", X_flat')
# npzwrite("models/taxinet/Y_image.npy", Y')

# save("data.jld2", Dict("X" => X, "Y" => Y))
# dat_dict = load("data.jld2")


using Colors, Plots
plt = plot(Gray.(X[:,:,1]))

>>>>>>> Stashed changes
