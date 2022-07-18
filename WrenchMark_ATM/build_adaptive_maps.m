clear; close all; clc
sd = 10; rng(sd);

% check for ATM code and add to path
ATMdir = '/path/to/ATM/sr'
if ~exist(ATMdir, 'dir')
    error('Install ATM code from: https://github.com/baptistar/ATM')
end
addpath(genpath(ATMdir))

%% Define parameters

hmax = 0.15; % mesh size: 0.5=coarse , 0.15=fine , 0.08 = ULTRA fine
precomputeEverything = 1; % you need it to be 1 in order to compute gradients, and 2 for parallel precomputation
wrench = Wrench(hmax,precomputeEverything);

%% Set the observation and its gradient

% Vertical displacement of the solution on the left of the wrench
% type = 1;
% Von Mises stress at some point
% type = 2;
% Displacement along the line
type = 3;

logE = wrench.logE();
[y,g,XObs,~] = wrench.evalObs(logE,type);

%% Evaluate Norms on the parameter and data space

% Data
RX = wrench.M + wrench.K;
[~,~,~,L] = wrench.evalObs(wrench.logE(),type);
L = wrench.B'*L;
RYinv = full(L'*(RX\L)) * 10;
RY = inv(RYinv);

[Uobs,Dobs] = svd(RYinv);
Cobs12 = Uobs*diag(sqrt(diag(Dobs)))*Uobs';
Cobsm12 = Uobs*diag(1./sqrt(diag(Dobs)))*Uobs';

%% Define model

model.wrench  = wrench;
model.d       = wrench.dimParam;
model.m       = size(y);
model.type    = type;
model.Cobs    = RYinv;
model.Cobsinv = RY;
model.Cobs12  = Cobs12;
model.Cobsm12 = Cobsm12;
model.Cpr     = wrench.Sigma;
model.Cpr12   = wrench.Sigma12;
model.Cprm12  = wrench.Sigma12*diag(1./diag(wrench.Sigma12'*wrench.Sigma12));

%% Generate samples for gradients

% define model
K = 500;
Hx = zeros(wrench.dimParam);
Hy = zeros(size(XObs,2));
data_y = zeros(size(XObs,2),K);
data_x = zeros(wrench.dimParam,K);
for k=1:K
    logE = wrench.logE();
    [y,g] = wrench.evalObs(logE,type);
    
    data_x(:,k) = logE;
    data_y(:,k) = y + model.Cobs12*randn(size(y,1),1);
    
    Hx = Hx + g'*model.Cobsinv*g;
    Hy = Hy + g*model.Cpr*g.';
end
Hx = Hx/K;
Hy = Hy/K;
HxBar = model.Cpr12.' * Hx * model.Cpr12;
HyBar = model.Cobsm12 * Hy * model.Cobsm12.';

%% Compute eigenvectors (CMI, PCA, CCA)

% CMI eigenvectors
[Ubar_cmi,Sx_cmi,~] = svd(HxBar);
[Vbar_cmi,Sy_cmi,~] = svd(HyBar);

% compute covariances
C_xx = model.Cpr;
C_yx = data_y*data_x'/K; % using that x is zero-mean
C_yy = cov(data_y');

% PCA eigenvectors
[U_pca,~] = svd(C_xx);
[V_pca,~] = svd(C_yy);

% run CCA
tol = 1e-10;
[U_cca, V_cca] = my_cca(C_xx,C_yy,C_yx,tol);

%% Create orthogonal vectors

[Ubar_cca,~] = qr(model.Cpr12.'*U_cca,0);
[Vbar_cca,~] = qr(model.Cobs12'*V_cca,0);

% Orthogonalization
[Ubar_pca,~] = qr(model.Cpr12.'*U_pca,0);
[Vbar_pca,~] = qr(model.Cobs12'*V_pca,0);

%% Generate samples for training map

% set parameters
Ntrain = 2000;
Nvalid = 500;

% generate training data
Xtrain = zeros(Ntrain, wrench.dimParam);
Ytrain = zeros(Ntrain, size(XObs,2));
for i=1:Ntrain
    
    logE = wrench.logE();
    y = wrench.evalObs(logE, model.type);
    
    Xtrain(i,:) = logE;
    Ytrain(i,:) = y + model.Cobs12*randn(size(y,1),1);
    
end

% generate testing data
Xvalid = zeros(Nvalid, wrench.dimParam);
Yvalid = zeros(Nvalid, size(XObs,2));
for i=1:Nvalid
    
    logE = wrench.logE();
    y = wrench.evalObs(logE, model.type);
    
    Xvalid(i,:) = logE;
    Yvalid(i,:) = y + model.Cobs12*randn(size(y,1),1);
    
end

%% Optimize map for pairs of reduced dimensions

r_vect     = 1:5;
s_vect     = 1:2;

PB_cmi = cell(length(r_vect), length(s_vect));
G_cmi  = cell(length(r_vect), length(s_vect));
PB_pca = cell(length(r_vect), length(s_vect));
G_pca  = cell(length(r_vect), length(s_vect));
PB_cca = cell(length(r_vect), length(s_vect));
G_cca  = cell(length(r_vect), length(s_vect));

for j = 1:length(r_vect)
    for k = 1:length(s_vect)
           
        % extract basis
        r = r_vect(j);
        s = s_vect(k);
        
        Ur_cmi = model.Cprm12 * Ubar_cmi(:,1:r);
        Vs_cmi = model.Cobsm12 * Vbar_cmi(:,1:s);
        
        Ur_pca = model.Cprm12 * Ubar_pca(:,1:r);
        Vs_pca = model.Cobsm12 * Vbar_pca(:,1:s);
        
        Ur_cca = model.Cprm12 * Ubar_cca(:,1:r);
        Vs_cca = model.Cobsm12 * Vbar_cca(:,1:s);
        
        % compute maps
        [PB_cmi{j,k}, G_cmi{j,k}] = construct_map(Xtrain, Ytrain, Xvalid, Yvalid, Ur_cmi, Vs_cmi);
        [PB_pca{j,k}, G_pca{j,k}] = construct_map(Xtrain, Ytrain, Xvalid, Yvalid, Ur_pca, Vs_pca);
        [PB_cca{j,k}, G_cca{j,k}] = construct_map(Xtrain, Ytrain, Xvalid, Yvalid, Ur_cca, Vs_cca);
        
        % print test error
        fprintf('r = %d, s = %d\n', r, s)
        fprintf('CMI: Train error %f\n', compute_testerr(PB_cmi{j,k}, G_cmi{j,k}, Xvalid, Yvalid, Ur_cmi, Vs_cmi));
        fprintf('CMI: Test  error %f\n', compute_testerr(PB_cmi{j,k}, G_cmi{j,k}, Xvalid, Yvalid, Ur_cmi, Vs_cmi));
        fprintf('PCA: Train error %f\n', compute_testerr(PB_pca{j,k}, G_pca{j,k}, Xvalid, Yvalid, Ur_pca, Vs_pca));
        fprintf('PCA: Test  error %f\n', compute_testerr(PB_pca{j,k}, G_pca{j,k}, Xvalid, Yvalid, Ur_pca, Vs_pca));
        fprintf('CCA: Train error %f\n', compute_testerr(PB_cca{j,k}, G_cca{j,k}, Xvalid, Yvalid, Ur_cca, Vs_cca));
        fprintf('CCA: Test  error %f\n', compute_testerr(PB_cca{j,k}, G_cca{j,k}, Xvalid, Yvalid, Ur_cca, Vs_cca));
        
    end
end

clear model wrench
save('adptive_transport_map_objects')

%% -- Helper Functions --

function [PB, G] = construct_map(Xtrain, Ytrain, Xvalid, Yvalid, Ur, Vs)

    % project data
    Xtrain = (Ur.'*Xtrain.').';
    Ytrain = (Vs.'*Ytrain.').';
    Xvalid = (Ur.'*Xvalid.').';
    Yvalid = (Vs.'*Yvalid.').';

    % find dimensions of data
    dx = size(Xtrain,2);
    dy = size(Ytrain,2);
    
    % concatenate data
    Ztrain = [Ytrain, Xtrain];
    Zvalid = [Yvalid, Xvalid];

    % standardize data
    G = GaussianPullbackDensity(dx+dy, true);
    G = G.optimize(Ztrain);
    Ztrain = G.S.evaluate(Ztrain);
    Zvalid = G.S.evaluate(Zvalid);

    % set basis and define TM
    basis = cell(dy+dx,1);
    for kk=1:(dy+dx)
        basis{kk} = HermiteProbabilistPolyWithLinearization();
        basis{kk}.bounds = quantile(Ztrain(:,kk),[0.01,0.99]).';
    end

    % define reference
    ref = IndependentProductDitribution(repmat({Normal()},1,dx+dy));

    % define map
    S = identity_map(1:(dy+dx), basis);
    TM = TriangularTransportMap(S);
    PB = PullbackDensity(TM, ref);

    % optimize map using cross-validation
    comp_idx = (dy+1):(dy+dx);
    PB = cross_validate_map(PB, ref, Ztrain, Zvalid, comp_idx);
    
end

function PB = cross_validate_map(PB, ref, Ztrain, Zvalid, comp_idx)
    % set defaults
    max_terms = 100;
    max_patience = 10;
    for k=comp_idx
        fprintf('Optimizing component %d\n',k)
        % run greedy approximation on S_valid
        Sk_valid = PB.S.S{k};
        [Sk_valid, output] = greedy_fit(Sk_valid, ref.factors{k}, max_terms, ...
                            Ztrain(:,1:k), Zvalid(:,1:k), max_patience);
        % find optimal number of terms (adding terms originally in S)
        % remove one to account for initial condition
        [~, n_added_terms] = min(output.valid_err);
        n_added_terms = n_added_terms(1) - 1;
        opt_terms = PB.S.S{k}.n_coeff + n_added_terms;
        fprintf('Final map: %d terms\n', opt_terms);
        % extract optimal multi-indices
        midx_opt = Sk_valid.multi_idxs();
        midx_opt = midx_opt(1:opt_terms,:);
        PB.S.S{k} = PB.S.S{k}.set_multi_idxs(midx_opt);
        % run greedy_fit up to opt_terms with all data
        a0 = zeros(opt_terms,1);
        PB.S.S{k} = optimize_component(PB.S.S{k}, ref.factors{k}, a0, Ztrain(:,1:k), []);
    end
end

function nll = compute_testerr(PB, G, X, Y, Ur, Vs)
    Z = [(Vs.'*Y.').', (Ur.'*X.').'];
    Z = G.S.evaluate(Z);
    nll = 0;
    for k = (size(Vs,2) + 1) : (size(Vs,2) + size(Ur,2))
         nll = nll + negative_log_likelihood(PB.S.S{k}, Normal(), Z(:,1:k));
    end
end

% -- END OF FILE --
