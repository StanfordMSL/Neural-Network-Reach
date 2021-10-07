# Data
Dynamics model taken from [this work](https://github.com/sisl/VerifyGAN).  
```X_image.npy``` are flattened images from the h5 datasets.  
```Y_image.npy``` are associated states (crosstrack position, heading error) from the h5 datasets.  
```X_dynamics.npy``` are states within the domain of our dynamics.  
```Y_dynamics.npy``` are next states according to our dynamics model.  
All .npy data files are numpy arrays where each data point is a column.  

# Files
```gen_data.jl``` is used to generate dynamics data.  
```reformat_data.jl``` is used to extract and reformat the image-state data from the h5 files.
