# Neural-Network-Reach
Exact forward and backward reachability of deep neural networks with ReLU activation. Associated paper available on arXiv:
```
https://arxiv.org/abs/2011.11609
```

## Requirements ##
- Julia xxx

## Installation ##
Clone the repository in the desired location:
```
git clone https://github.com/StanfordMSL/Neural-Network-Reach
```
In the Neural-Network-Reach directory activate the project:
```
julia
(@v1.4) pkg> activate .
(Neural-Network-Reach) pkg> instantiate
```

## Examples ##
### Dampled Pendulum Example ###
In ```pendulum.jl``` the following should be specified:
- ```copies```: How many times the dynamics network should be concatenated.
- ```model```: Which trained dynamics model to use.
- keyword arguments of ```compute_reach()```: Specify whether to compute forward and/or backward reachable sets and/or solve a verification problem.

### ACAS Xu Example ###
In ```acas.jl``` the following should be specified:
- ```acas_net(a,b)```: Arguments ```a,b``` specify which ACAS network to analyze.
- ```property```: Current implementation supports ```"acas property 3"``` and ```"acas property 4"```.
- keyword arguments of ```compute_reach()```: Specify whether to compute forward and/or backward reachable sets and/or solve a verification problem.

### Random Network Example ###
In ```random.jl``` the following should be specified:
- ```test_random_flux(2, 2, 50, 3)```: Arguments are input dim, output dim, hidden layer dim, num layers.
- keyword arguments of ```compute_reach()```: Specify whether to compute forward and/or backward reachable sets and/or solve a verification problem.

### Loading a Network ###


### Forward Reachability ###


### Backward Reachability ###


### Verification ###


### Tips ###
