# MPT3
I set ```Opt.tol_minR = 1e-10``` in ```mpt/modules/compatibility/optmerge/mpt_exHyperAdv.m``` (Line 124).
When this tolerance is too large and MPT treats a PolyUnion of 2 polytopes as just 1 polytope, some errors can occur.
I set ```Algorithm = Inf``` in ```mpt/modules/geometry/unions/@PolyUnion/merge.m``` (Line 158). This is to ensure no overlaps when merging polytopes.

# Data
Dynamics model taken from [this work](https://github.com/sisl/VerifyGAN).  
```X_image.npy``` are flattened images from the h5 datasets.  
```Y_image.npy``` are associated states (crosstrack position, heading error) from the h5 datasets.  
```X_dynamics.npy``` are states within the domain of our dynamics.  
```Y_dynamics.npy``` are next states according to our dynamics model.  
All .npy data files are numpy arrays where each data point is a row.  

# Files
```gen_data.jl``` is used to generate dynamics data.  
```reformat_data.jl``` is used to extract and reformat the image-state data from the h5 files.
