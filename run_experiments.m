clear; close all; clc

cd LinearExample
linear_joint_paper
cd ..

cd WrenchMark
plot_solutions
compute_subspaces
cd ..

cd WrenchMark_ATM
build_adaptive_maps
evaluate_negativeloglikelihood
cd ..

cd Imaging
sample_images
compute_eigenvectors
convergence_withincreasingdim
cd ..

cd ConditionedDiffusion
plot_prior_samples
compute_basis
plot_convergence_paper
evaluate_permutations
cd ..
