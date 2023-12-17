# Conjugate Gradient Method Library
Conjugate Gradient Method Library implement the linear CG method and 
non-linear CG menthod by numpy and C++. 
Furthermore, Conjugate Gradient Method Library supports 
std::thread and OpenMP operations, that can further accelerate the c++ library.

## Introduction to the Conjugate Gradient Method
Conjugate gradient method is a numerical method that can find the 
minima of the function in the hyper-dimensional space, and conjugate 
gradient method includes linear conjugate method and non-linear conjugate method. 

Linear CG algorithm can precisely calculate the step length in every iteration,
but in order to calculate the precise step length, the objective function could
only be quadratic function. Compared with the steepest gradient method, the 
minima could be found in the 
finite step by conjugate gradient method in theory (if we don't consider the 
floating-point error and ill-conditioned), and the convergence iteration of 
conjugate gradient method is less than those of steepest gradient method.

Non-linear CG algorithm can approximately calculate the step length by line 
search, or gradient descent method, and the advantage of non-linear CG 
algorithm is that the target function could be convex nonlinear objective 
functions.

## Getting Started

1. Clone this repo.

2. Install [numpy](https://numpy.org/install/), 
[autograd](https://github.com/HIPS/autograd), 
and [OpenMP](https://www.openmp.org/), 
or build and run the dockerfile in /contrib/docker.

3. ```make test``` for run the pytest 
and ```make demo``` for run the simple example and analysis.

## User Tutorial
<a href="./python">API Introduction</a>

