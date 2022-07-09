function [model, obs, prior] = setup_model(N, k_obs)

model = struct;
model.N = N;
model.Tend = 1;%10;
model.d = 10;
model.dt = model.Tend/N;
model.tt = linspace(0, model.Tend, model.N+1);
model.k = k_obs;
model.sensors = (model.k+1):model.k:(model.N+1);

% define standard Gaussian prior 
prior = struct;
prior.C = eye(model.N);
prior.L = eye(model.N);
prior.NP = model.N;
prior.dof = model.N;
prior.type = 'Field';
prior.mean_u = zeros(model.N,1);

% define obs model
obs = struct;
obs.std = 0.5;
obs.like = 'normal';
obs.n_data = length(model.sensors);
obs.jacobian = true;