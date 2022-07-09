clear; close all; clc
sd = 2; rng(sd)

% setup model
N = 100;
k_obs = 1;
[model, obs, prior] = setup_model(N, k_obs);
obs.std = 0.1;
obs.Cobs = obs.std^2 * speye(obs.n_data);

%% Generate prior samples

n_samples = 200;

% Generate prior samples
np = model.N;
v_pr = randn(np, n_samples);

% transform prior samples to correlated space
u_pr = matvec_prior_L(prior, v_pr) + prior.mean_u;

% sample from joint density
sol_pr = zeros(obs.n_data, n_samples);
y_pr   = zeros(obs.n_data, n_samples);
for i=1:n_samples
    sol = forward_solve(model, u_pr(:,i));
    sol_pr(:,i) = sol.d;
    y_pr(:,i) = sol.d + obs.std * randn(length(sol.d),1);
end

figure
plot(model.tt, [zeros(1,200); y_pr(:,1:200)])
set(gca,'FontSize',20)
xlabel('Time $t$','FontSize',24)
ylabel('$y_t$','FontSize',24)
set(gca,'LineWidth',2)
print('-depsc','sample_observations')

% -- END OF FILE --
