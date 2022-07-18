# Gradient-based dimension reduction for Bayesian inference

This repository examines data and parameter dimension reduction for Bayesian inference problems. We consider three applications with forward models that include an elliptic partial differential equation, an optical imaging technique, and a stochastic differential equation with a nonlinear drift. The code in this repository reproduces the results available in the [preprint](https://arxiv.org). Our approach derives bounds for the posterior approximation error (in KL divergence) that can be tractably minimized to identify the relevant subspaces of the parameter and data for each problem.

## Authors

Ricardo Baptista (MIT), Youssef Marzouk and Olivier Zahm
E-mails: rsb@mit.edu, ymarz@mit.edu, olivier.zahm@inria.fr

## Installation

The code is implemented in MATLAB and the results can be reproduced by calling the main script `run_experiments.m`. One of the experiments requires installing the [ATM package](https://github.com/baptistar/ATM) to learn parametric transport maps. These maps are used to sample reduced posterior distributions in Bayesian inference problems.