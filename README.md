# Neural-Network-Reach
Exact forward and backward reachability of deep neural networks with ReLU activation. Associated with the papers:

- [Reachable Polyhedral Marching (RPM): A Safety Verification Algorithm for Robotic Systems with Deep Neural Network Components](https://ieeexplore.ieee.org/document/9561956) (conference version)
- [Reachable Polyhedral Marching (RPM): An Exact Analysis Tool for Deep-Learned Control Systems](https://arxiv.org/abs/2210.08339) (journal version)

## Requirements ##
- Julia 1.10.4
- See Project.toml
- Some experiments use [MPT3](https://www.mpt3.org/) (version 3.2.1) with MATLAB R2024a. Typically MATLAB should be run with Administrator priviledges for these.


## Installation ##
Clone the repository in the desired location:
```
git clone https://github.com/StanfordMSL/Neural-Network-Reach
```
In the Neural-Network-Reach directory activate the project:
```
julia # launch julia REPL
julia> ] # switch to package manager
(@v1.8) pkg> activate .
(Neural-Network-Reach) pkg> instantiate
```
This creates a Julia environment for the package and sets up all necessary packages.

## Setting up the venv
You shouldn't need to setup the python venv unless you want to replicate the training of the neural networks used in the examples.
But if that is the case, then the venv can be set up as follows.
- Install ``pip`` and ``virtualenv`` if you don't have them already.

- Clone the repository and navigate to the folder in a terminal.

- Run ``virtualenv venv`` to make a new virtual environment for the project in the repo folder (the ``./venv/`` folder is git ignored so it won't push those files to the repo).

- Activate the virtual environment with: ``source venv/bin/activate``.

- Install the requirements with: ``pip install -r requirements.txt``.

- If you've installed new packages and want to add them to the ``requirements.txt`` just do: ``pip freeze > requirements.txt``.

## Examples ##

### Random Network Example ###
This file is used to test the algorithm on ReLU networks with randomly initialized weights.
In ```random.jl``` the following should be specified:
- ```in_d, out_d, hdim, layers```: Arguments are input dimension, output dimension, hidden layer dimension, and number of layers.
Then run the file:
```
julia> include("random.jl")
```

### Dampled Pendulum Example ###
This example appears in the paper [Reachable Polyhedral Marching (RPM): A Safety Verification Algorithm for Robotic Systems with Deep Neural Network Components](https://ieeexplore.ieee.org/document/9561956).

In ```pendulum.jl``` the following should be specified:
- ```copies```: How many times the dynamics network should be concatenated.
- keyword arguments of ```compute_reach()```: Specify whether to compute forward and/or backward reachable sets and/or solve a verification problem.
Then run the file:
```
julia> include("pendulum.jl")
```

### ACAS Xu Example ###
This example appears in the paper [Reachable Polyhedral Marching (RPM): A Safety Verification Algorithm for Robotic Systems with Deep Neural Network Components](https://ieeexplore.ieee.org/document/9561956).

In ```acas.jl``` the following should be specified:
- ```acas_net_nnet(a,b)```: Arguments ```a,b``` specify which ACAS network to analyze.
- ```property```: Current implementation supports ```"acas property 3"```.
- keyword arguments of ```compute_reach()```: Specify whether to compute forward and/or backward reachable sets and/or solve a verification problem.
Then run the file:
```
julia> include("acas.jl")
```
This file also includes functionality to verify all ACAS networks, as well as find the exact partition of the input space corresponding to each decision output by the network.






### Quadratic Example ###
This example appears as Figure 1 in [Reachable Polyhedral Marching (RPM): An Exact Analysis Tool for Deep-Learned Control Systems](https://arxiv.org/abs/2210.08339)

The PWA representation of the ReLU network used to approximate a quadratic function can be obtained by running:
```
julia> include("quadratic.jl")
```
We then save the affine regions and use MATLAB to plot the function in 3D using the file ```plot_3D.m```.


### Vanderpol Region of Attraction Example ###
This example appears in [Reachable Polyhedral Marching (RPM): An Exact Analysis Tool for Deep-Learned Control Systems](https://arxiv.org/abs/2210.08339)

In ```vanderpol_roa.jl``` the following should be specified:
- ```connected```: Whether to restrict RPM to enumerate a connected ROA.
- ```steps```: A list of the number of steps to compute for the ROA.

Then run the file:
```
julia> include("vanderpol_roa.jl")
```
Using the standard Lyapunov approach to find an ROA is implemented in the file ```mat_invariant_vanderpol.m```.
This approach fails to find an invariant domain, and thus cannot search for a Lyapunov function.

### Controlled Pendulum Example ###
This example appears in [Reachable Polyhedral Marching (RPM): An Exact Analysis Tool for Deep-Learned Control Systems](https://arxiv.org/abs/2210.08339)

The plot in the paper can be reproduced by running the file
```
julia> include("pendulum_controlled.jl")
```
This file also solves for the PWA representation of the dynamics network.
The control invariant set is found in MATLAB with the file ```mat_invariant_pendulum.m```.


### Runway Taxiing Example ###
This example appears in [Reachable Polyhedral Marching (RPM): An Exact Analysis Tool for Deep-Learned Control Systems](https://arxiv.org/abs/2210.08339)

 - ```taxinet_roa.jl``` was used to solve for the explicit PWA representation of the closed loop dynamics as well as find the stable fixed point and find the seed ROA.
 - ```pwa_back_reach.jl``` was used to compute each backward reachability step.
 - ```merge_poly.jl``` was used to convert polyhedra between jld2 and mat variables so we can use both Julia and MATLAB.
 - ```merge_poly.m``` was used to merge polyhedra in order to reduce complexity of the next backward reachability step.
 




