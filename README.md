# Neural-Network-Reach
Exact forward and backward reachability of deep neural networks with ReLU activation. Associated [paper](https://arxiv.org/abs/2011.11609).

## Requirements ##
- Julia 1.4
- See Project.toml

## Installation ##
Clone the repository in the desired location:
```
git clone https://github.com/StanfordMSL/Neural-Network-Reach
```
In the Neural-Network-Reach directory activate the project:
```
julia # launch julia REPL
julia> ] # switch to package manager
(@v1.4) pkg> activate .
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
- ```model```: Which trained dynamics model to use.
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
- ```test_random_flux(a, b, c, d)```: Arguments are input dim, output dim, hidden layer dim, num layers.
- keyword arguments of ```compute_reach()```: Specify whether to compute forward and/or backward reachable sets and/or solve a verification problem.
Then run the file:
```
julia> include("random.jl")
```
