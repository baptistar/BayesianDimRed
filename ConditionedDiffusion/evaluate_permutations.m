clear; close all; clc
sd = 2; rng(sd)

% setup model
N = 100;
k_obs = 1;
[model, obs, prior] = setup_model(N, k_obs);
obs.std = 0.1;
obs.Cobs = obs.std^2 * speye(obs.n_data);

%% Compute projector using prior samples

% Generate prior samples
np = model.N;
n_samples = 1e6;
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
Ux = prior.L * Ux;
Uy = inv(Lobs).' * Uy;

% compute eigenvalue upper bounds
Rx = cumsum(diag(Dx(2:end,2:end)),'reverse');
Ry = cumsum(diag(Dy(2:end,2:end)),'reverse');
%%
% plot entries on diagonal 
figure
hold on
plot(1:length(Dx),diag(THx),'linewidth',3)
plot(1:length(Dy),diag(THy),'linewidth',3)
xlim([1,50])
set(gca,'YScale','log')
set(gca,'FontSize',20)
xlabel('Index i','FontSize',24)
ylabel('Diagonal entries $H_{ii}$','FontSize',24)
legend('Parameter space','Data space','FontSize',20)
set(gca,'LineWidth',2)
hold off
print('-depsc','param_vs_data_diagnostic_matrix_diagonals')
%%
% compare bounds
Rx_perm = cumsum(sort(diag(THx),'descend'),'reverse'); Rx_perm = Rx_perm(2:end);
Ry_perm = cumsum(sort(diag(THy),'descend'),'reverse'); Ry_perm = Ry_perm(2:end);

% plot entries on diagonal 
figure
hold on
plot(1:length(Rx),Rx,'linewidth',3)
plot(1:length(Rx_perm),Rx_perm,'linewidth',3)
xlim([1,50])
set(gca,'YScale','log')
set(gca,'FontSize',20)
xlabel('Reduced parameter dimension, $r$','FontSize',24)
ylabel('Expected KL upper bound','FontSize',24)
legend('Optimal rotation','Optimal permutation','FontSize',20)
set(gca,'LineWidth',2)
hold off
print('-depsc','cd_param_upperbounds')

figure
hold on
plot(1:length(Ry),Ry,'linewidth',3)
plot(1:length(Ry_perm),Ry_perm,'linewidth',3)
xlim([1,50])
set(gca,'YScale','log')
set(gca,'FontSize',20)
xlabel('Reduced observation dimension, $s$','FontSize',24)
ylabel('Expected KL upper bound','FontSize',24)
legend('Optimal rotation','Optimal permutation','FontSize',20)
set(gca,'LineWidth',2)
hold off
print('-depsc','cd_data_upperbounds')

% -- END OF FILE --
