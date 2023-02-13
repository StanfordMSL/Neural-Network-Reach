# Neural-Network-Reach
Exact forward and backward reachability of deep neural networks with ReLU activation. Associated with the papers:

- [Reachable Polyhedral Marching (RPM): A Safety Verification Algorithm for Robotic Systems with Deep Neural Network Components](https://ieeexplore.ieee.org/document/9561956) (conference version)
- [Reachable Polyhedral Marching (RPM): An Exact Analysis Tool for Deep-Learned Control Systems](https://arxiv.org/abs/2210.08339) (journal version)

## Requirements ##
- Julia 1.8.2
- See Project.toml
- For the Taxinet example I use [MPT3](https://www.mpt3.org/) (version 3.2.1) with MATLAB R2020a. See the ReadMe for the Taxinet example for more details.

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

- Install ``pip`` and ``virtualenv`` if you don't have them already.

- Clone the repository and navigate to the folder in a terminal.

- Run ``virtualenv venv`` to make a new virtual environment for the project in the repo folder (the ``./venv/`` folder is git ignored so it won't push those files to the repo).

- Activate the virtual environment with: ``source venv/bin/activate``.

- Install the requirements with: ``pip install -r requirements.txt``.

- If you've installed new packages and want to add them to the ``requirements.txt`` just do: ``pip freeze > requirements.txt``.

## Examples ##
### Dampled Pendulum Example ###
In ```pendulum.jl``` the following should be specified:
- ```copies```: How many times the dynamics network should be concatenated.
- keyword arguments of ```compute_reach()```: Specify whether to compute forward and/or backward reachable sets and/or solve a verification problem.
Then run the file:
```
julia> include("pendulum.jl")
```

### ACAS Xu Example ###
In ```acas.jl``` the following should be specified:
- ```acas_net(a,b)```: Arguments ```a,b``` specify which ACAS network to analyze.
- ```property```: Current implementation supports ```"acas property 3"``` and ```"acas property 4"```.
- keyword arguments of ```compute_reach()```: Specify whether to compute forward and/or backward reachable sets and/or solve a verification problem.
Then run the file:
```
julia> include("acas.jl")
```

### Random Network Example ###
In ```random.jl``` the following should be specified:
- ```in_d, out_d, hdim, layers```: Arguments are input dimension, output dimension, hidden layer dimension, and number of layers.
- keyword arguments of ```compute_reach()```: Specify whether to compute forward and/or backward reachable sets and/or solve a verification problem.
Then run the file:
```
julia> include("random.jl")
```


### Vanderpol Region of Attraction Example ###
In ```vanderpol_roa.jl``` the following should be specified:
- ```steps```: A list of the number of steps to compute for the BRS.

Then run the file:
```
julia> include("vanderpol_roa.jl")
```
