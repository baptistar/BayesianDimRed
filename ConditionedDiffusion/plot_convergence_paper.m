clear; close all; clc
sd = 2; rng(sd)

addpath('../tools')

% setup model
N = 100;
k_obs = 1;
[model, obs, prior] = setup_model(N, k_obs);
obs.std = 0.1;
obs.Cobs = obs.std^2 * speye(obs.n_data);

% define sample sizes
n_samples_vect = 1e4;
n_samples = max(n_samples_vect);

%% Compute projector using prior samples

% Generate prior samples
np = model.N;
v_pr = randn(np, n_samples);

% transform prior samples to correlated space
u_pr = matvec_prior_L(prior, v_pr) + prior.mean_u;

% define inverse of observational noise covariance
Cobs = obs.Cobs;
Lobs = obs.std * eye(obs.n_data);
I = speye(obs.n_data, obs.n_data)./obs.std^2;

% estimate the LIS diagnostic matrix
ui = zeros(model.N+1,n_samples);
Hx = zeros(np, np);
Hy = zeros(obs.n_data, obs.n_data);
for i=1:n_samples
    if mod(i,1e5) == 0
        disp(i)
    end
    sol = forward_solve(model, u_pr(:,i));
    ui(:,i) = sol.G;
    Ju = explicit_jacobian(model, sol);
    Hx = Hx + Ju' * I * Ju;
    Hy = Hy + Ju * prior.C * Ju.';
end
Hx = Hx/n_samples;
Hy = Hy/n_samples;

% apply transformation to Hx, Hy
THx = prior.L.' * Hx * prior.L;
THy = inv(Lobs) * Hy * inv(Lobs).';

% compute eigenvectors of matrices
[Ux,Dx,~] = svd(THx);
[Uy,Dy,~] = svd(THy);

% apply inverse transformation to Ux, Uy
%Ux = prior.L * Ux;
%Uy = inv(Lobs).' * Uy;

% compute upper bound - sum of trailing eigenvalues
Rx = cumsum(diag(Dx(2:end,2:end)),'reverse');
Ry = cumsum(diag(Dy(2:end,2:end)),'reverse');

%% Generate prior samples

% sample from joint density
sol_pr = zeros(obs.n_data, n_samples);
y_pr   = zeros(obs.n_data, n_samples);
for i=1:n_samples
    sol = forward_solve(model, u_pr(:,i));
    sol_pr(:,i) = sol.d;
    y_pr(:,i) = sol.d + obs.std * randn(length(sol.d),1);
end

% evaluate log-likelihood at all samples
misfit = (y_pr - sol_pr);
loglik = -0.5*dot(misfit, Cobs \ misfit).';

%% Compute KL divergence for projecting parameters

% fix n_outer
n_inner_vect = [10,100,1000];

% define arrays to store KL and MI
r_vect = 5:5:100;
MI_par = zeros(length(n_inner_vect), length(r_vect), 2);
MI_par_condexp = zeros(1, length(r_vect), 2);

for k=1:length(n_inner_vect)
    for i=1:length(r_vect)
        
        % define n_inner
        n_inner = n_inner_vect(k);
        
        % define rank
        r = r_vect(i);
        Ur = Ux(:,1:r);
        Pr = (prior.L \ Ur) * (Ur.' * prior.L);

        % evaluate log-likelihood approximation
        log_approx_lik = zeros(n_samples,1);
        for j=1:n_samples
            v_pr_inner = randn(model.N, n_inner);
            u_pr_inner = matvec_prior_L(prior, v_pr_inner) + prior.mean_u;
            % project sample and transform to the correlated prior space
            utilde = Pr*u_pr(:,j) + (eye(np) - Pr)*u_pr_inner(:,1:n_inner);
            obs.data = y_pr(:,j);
            log_approx_lik(j) = logsumexp(log_likelihood(model, obs, utilde)) - log(n_inner);
        end

        % evaluate ratio of likelihoods
        loglik_ratio = loglik - log_approx_lik;

        % estimate MI and its standard error
        MI_par(k,i,1) = mean(loglik_ratio);
        MI_par(k,i,2) = 1.96*std(loglik_ratio)/sqrt(length(loglik_ratio));

        % evaluate log-likelihood approximation
        log_approx_lik = zeros(n_samples,1);
        for j=1:n_samples
            % project sample and transform to the correlated prior space
            utilde = Pr*u_pr(:,j) + (eye(np) - Pr)*prior.mean_u;
            obs.data = y_pr(:,j);
            log_approx_lik(j) = log_likelihood(model, obs, utilde);
        end

        % evaluate ratio of likelihoods
        loglik_ratio = loglik - log_approx_lik;

        % estimate MI and its standard error
        MI_par_condexp(k,i,1) = mean(loglik_ratio);
        MI_par_condexp(k,i,2) = 1.96*std(loglik_ratio)/sqrt(length(loglik_ratio));

    end
end

figure
hold on
plot(1:length(Rx), Rx, '--','linewidth', 3)
for k=1:length(n_inner_vect)
    errorbar(r_vect, MI_par(k,:,1), MI_par(k,:,2), '-','linewidth', 3)
end
errorbar(r_vect, MI_par_condexp(1,:,1), MI_par_condexp(1,:,2), '-k','linewidth', 3)
set(gca,'YScale','log')
set(gca,'FontSize',20)
xlabel('Reduced parameter dimension, $r$','FontSize',24)
ylabel('$I(X_\perp,Y|X_r)$','FontSize', 24)
legend('Upper bound $\sum_{i > r} \lambda_i$','$\ell = 10$','$\ell = 100$','$\ell = 1000$','Prior Mean, $\ell = 1$')
xlim([min(r_vect),95])
ylim([5e-1,4e2])
set(gca,'LineWidth',2)
hold off
print('-depsc','cd_parameter_MI')

%% Compute KL divergence for projecting data 

% sample from joint density
sol_pr = zeros(obs.n_data, n_samples);
y_pr   = zeros(obs.n_data, n_samples);
for i=1:n_samples
    sol = forward_solve(model, u_pr(:,i));
    sol_pr(:,i) = sol.d;
    y_pr(:,i) = sol.d + obs.std * randn(length(sol.d),1);
end

% evaluate log-likelihood at all samples
misfit = (y_pr - sol_pr);
loglik = -0.5*dot(misfit, Cobs \ misfit).';

% define arrays to store MI
s_vect = 5:5:100;
n_inner_vect = [10,100,1000];
MI_data = zeros(length(n_inner_vect), length(s_vect), 2);
MI_data_condexp = zeros(length(s_vect), 2);

for k=1:length(n_inner_vect)
    
    % set n_inner
    n_inner = n_inner_vect(k);

    % estimate logpiy for each sample y (loop over n_inner for speed)
    logpiycx = zeros(n_samples,n_inner);
    sol_new  = cell(n_inner,1);
    for kk=1:n_inner
        sol_new{kk} = zeros(obs.n_data, n_samples);
        v_pr_new = randn(np, n_samples);
        u_pr_new = matvec_prior_L(prior, v_pr_new) + prior.mean_u;
        for ii=1:n_samples
            sol = forward_solve(model, u_pr_new(:,ii));
            sol_new{kk}(:,ii) = sol.d;
        end
        misfit_kk = y_pr - sol_new{kk};
        logpiycx(:,kk) = (-0.5*dot(misfit_kk, (obs.Cobs\misfit_kk))');
    end
    logpiy = logsumexp(logpiycx.').' - log(n_inner);
    disp(mean(loglik - logpiy))

    % evaluate the KL divergence error for each rank
    for i=1:length(s_vect)
        disp([k,i])

        % define projector
        s = s_vect(i);
        Us = Uy(:,1:s).';
        Cobs_s = Us * obs.Cobs * Us.';

        % evaluate log-likelihood approximation
        Pr_misfit = Us * misfit;
        log_approx_lik = -0.5*dot(Pr_misfit, (Cobs_s\Pr_misfit))';

        % estimate logpitilde_y for each sample y (loop over n_inner for speed)
        logpitildeycx = zeros(n_samples, n_inner);
        for kk=1:n_inner
            Pr_misfit_kk = Us * (y_pr - sol_new{kk});
            logpitildeycx(:,kk) = (-0.5*dot(Pr_misfit_kk, (Cobs_s\Pr_misfit_kk))');
        end
        logpitildey = logsumexp(logpitildeycx.').' - log(n_inner);

        % estimate KL and its standard error
        MI_diff_datasamps = (loglik - logpiy) - (log_approx_lik - logpitildey);
        MI_data(k,i,1) = mean(MI_diff_datasamps);
        MI_data(k,i,2) = 1.96*std(MI_diff_datasamps)/sqrt(length(MI_diff_datasamps));

    end
end

figure
hold on
plot(1:length(Ry), Ry, '--','linewidth', 3)
for k=1:length(n_inner_vect)
    errorbar(s_vect, MI_data(k,:,1), MI_data(k,:,2), '-','linewidth', 3)
end
set(gca,'YScale','log')
set(gca,'FontSize',20)
xlabel('Reduced observation dimension, $s$','FontSize',24)
ylabel('$I(Y_\perp;X|Y_s)$','FontSize', 24)
legend('Upper bound $\sum_{i > s} \lambda_i$','$\ell = 10$','$\ell = 100$','$\ell = 1000$')%,'$m = 1$',)
xlim([min(s_vect),95])
ylim([1e0,2e3])
set(gca,'LineWidth',2)
hold off
print('-depsc','cd_data_MI')

% -- END OF FILE --