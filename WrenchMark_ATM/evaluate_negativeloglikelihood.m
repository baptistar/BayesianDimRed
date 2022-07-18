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

%% Draw joint samples

% set number of samples
n_samples = 1e6;

x_st = zeros(n_samples, model.d);
y_st = zeros(n_samples, model.m(1));
    
for i=1:n_samples
    % draw sample
    logE = wrench.logE();
    y = wrench.evalObs(logE, model.type);
    y = y + model.Cobs12*randn(size(y,1),1);
    % save sample
    x_st(i,:) = logE;
    y_st(i,:) = y;
end

%% Load adaptive maps

load('adptive_transport_map_objects',...
        'PB_cmi','PB_pca','PB_cca', ...
        'G_cmi','G_pca','G_cca',...
        'Ubar_cmi','Ubar_pca','Ubar_cca',...
        'Vbar_cmi','Vbar_pca','Vbar_cca',...
        'Vs_cmi','Vs_pca','Vs_cca');

%% Sample posterior using map

% set r, s
r_vect = 1:5;
s_vect = 1:2;

%set q lower bound
q_lb = 0.001;

% set batch size
batch_size = 10000;

% define cells to store samples
anll_cmi = zeros(length(r_vect), length(s_vect), 2);
anll_pca = zeros(length(r_vect), length(s_vect), 2);
anll_cca = zeros(length(r_vect), length(s_vect), 2);

% draw
for j = 1:length(r_vect)
    for k = 1:length(s_vect)
            
        % extract basis
        r = r_vect(j);
        s = s_vect(k);

        % extract Cprm12
        Cprm12r = model.Cprm12(:,1:size(Ubar_cmi,1));

        Ur_cmi = Cprm12r * Ubar_cmi(:,1:r);
        [Ubar_all,~] = qr(Ubar_cmi(:,1:r));
        Uperp_cmi = Cprm12r * Ubar_all(:,r+1:end);
        Vs_cmi = model.Cobsm12 * Vbar_cmi(:,1:s);

        Ur_pca = Cprm12r * Ubar_pca(:,1:r);
        [Ubar_all,~] = qr(Ubar_pca(:,1:r));
        Uperp_pca = Cprm12r * Ubar_all(:,r+1:end);
        Vs_pca = model.Cobsm12 * Vbar_pca(:,1:s);

        Ur_cca = Cprm12r * Ubar_cca(:,1:r);
        [Ubar_all,~] = qr(Ubar_cca(:,1:r));
        Uperp_cca = Cprm12r * Ubar_all(:,r+1:end);
        Vs_cca = model.Cobsm12 * Vbar_cca(:,1:s);

        % evaluate \pi(x_r|y_s) and \pi(x_\perp|x_r) for CMI
        reduced_logpost_cmi = evaluate_logposterior(PB_cmi{j,k}, G_cmi{j,k}, y_st, x_st, Ur_cmi, Vs_cmi, batch_size);
        idx = (reduced_logpost_cmi > quantile(reduced_logpost_cmi,q_lb));
        comp_logprior_cmi = sum(log(normpdf((Uperp_cmi.' * x_st.').')),2);
        log_post_cmi = (reduced_logpost_cmi(idx) + comp_logprior_cmi(idx));
        anll_cmi(j,k,1) = -1 * mean(log_post_cmi);
        anll_cmi(j,k,2) = 1.96 * std(log_post_cmi) / sqrt(n_samples);

        % evaluate \pi(x_r|y_s) and \pi(x_\perp|x_r) for PCA
        reduced_logpost_pca = evaluate_logposterior(PB_pca{j,k}, G_pca{j,k}, y_st, x_st, Ur_pca, Vs_pca, batch_size);
        idx = (reduced_logpost_pca > quantile(reduced_logpost_pca,q_lb));
        comp_logprior_pca = sum(log(normpdf((Uperp_pca.' * x_st.').')),2);
        log_post_pca = (reduced_logpost_pca(idx) + comp_logprior_pca(idx));
        anll_pca(j,k,1) = -1 * mean(log_post_pca);
        anll_pca(j,k,2) = 1.96 * std(log_post_pca) / sqrt(n_samples);

        % evaluate \pi(x_r|y_s) and \pi(x_\perp|x_r) for CCA
        reduced_logpost_cca = evaluate_logposterior(PB_cca{j,k}, G_cca{j,k}, y_st, x_st, Ur_cca, Vs_cca, batch_size);
        idx = (reduced_logpost_cca > quantile(reduced_logpost_cca,q_lb));
        comp_logprior_cca = sum(log(normpdf((Uperp_cca.' * x_st.').')),2);
        log_post_cca = (reduced_logpost_cca(idx) + comp_logprior_cca(idx));
        anll_cca(j,k,1) = -1 * mean(log_post_cca);
        anll_cca(j,k,2) = 1.96 * std(log_post_cca) / sqrt(n_samples);

    end
end

%% Save results

clearvars -except nll_cmi nll_pca nll_cca anll_cmi anll_pca anll_cca r_vect s_vect order_vect
save('post_process_results')

%% Print table of results

anll_all = {anll_cmi, anll_pca, anll_cca};
alg_all = {'CMI','PCA','CCA'};

offset = anll_cca(1,1,1);

fprintf('       ')
for k=1:length(s_vect)
    for j=1:(length(r_vect)-1)
        fprintf('     (%d,%d)   ',r_vect(j),s_vect(k))
    end
end

for i=1:length(anll_all)
    anll_i = anll_all{i};
    fprintf('\n\\text{%s} ', alg_all{i});
    for k=1:length(s_vect)
        for j = 1:(length(r_vect)-1)
            fprintf('& $%.3f \\pm %.2f$ ', anll_i(j,k,1) - offset, anll_i(j,k,2));
        end
    end
end

%% -- Helper Functions --

function log_pdf = evaluate_logposterior(PB, G, y_test, x_test, Ur, Vs, batch_size)
    assert(size(x_test,1) == size(y_test,1))
    if nargin < 7
        batch_size = 1000;
    end
    
    % divide data
    n = size(x_test,1);
    n_batches = ceil(n/batch_size);
    splits = cvpartition(n,'kFold',n_batches);
    
    % define array to store results
    log_pdf = zeros(n,1);
    
    for i=1:n_batches
        % project data
        x_r = (Ur.'*x_test(splits.test(i),:).').';
        y_s = (Vs.'*y_test(splits.test(i),:).').';

        % concatenate data
        yx_sr = [y_s, x_r];

        % define map components
        r = size(Ur,2);
        s = size(Vs,2);
        comp_idx = (s+1):(s+r);

        % evaluate Gaussian map
        Gz_sr = G.S.evaluate(yx_sr);
        logdet_Gz = G.S.logdet_Jacobian(yx_sr, comp_idx);

        % evaluate non-linear map
        Sz_sr = PB.S.evaluate(Gz_sr, comp_idx);
        logdet_PB_sr = PB.S.logdet_Jacobian(Gz_sr, comp_idx);

        % define reference
        ref = IndependentProductDistribution(repmat({Normal()},1,r));

        % compile terms
        log_pdf_i = ref.log_pdf(Sz_sr) + logdet_PB_sr + logdet_Gz;
        log_pdf(splits.test(i)) = log_pdf_i;
    end
    
end

% -- END OF FILE --
