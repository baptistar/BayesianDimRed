clear; close all; clc
sd = 2; rng(sd)

addpath('../tools/')

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

% plot eigenvalues and eigenvectors
figure
hold on
plot(1:length(Dx),diag(Dx),'linewidth',3)
plot(1:length(Dy),diag(Dy),'linewidth',3)
xlim([1,50])
set(gca,'YScale','log')
set(gca,'FontSize',20)
xlabel('Index i','FontSize',24)
ylabel('Eigenvalues, $\lambda_i$','FontSize',24)
legend('Parameter space projector','Data space projector','FontSize',20)
set(gca,'LineWidth',2)
hold off
print('-depsc','param_vs_data_eigvalues')

figure
hold on
plot(1:length(Rx),Rx,'linewidth',3)
plot(1:length(Ry),Ry,'linewidth',3)
xlim([1,50])
set(gca,'YScale','log')
set(gca,'FontSize',20)
xlabel('Index i','FontSize',24)
ylabel('Trailing eigenvalues, $\sum_{i>t}\lambda_i$','FontSize',24)
legend('Parameter space projector','Data space projector','FontSize',20)
set(gca,'LineWidth',2)
hold off
print('-depsc','param_vs_data_eigvalue_bounds')

figure
hold on
plot(1:size(Hx,1),diag(Hx),'linewidth',3)
plot(1:size(Hy,1),diag(Hy),'linewidth',3)
set(gca,'YScale','log')
set(gca,'FontSize',20)
xlabel('Index i','FontSize',24)
ylabel('Diagonal entries, $H_{ii}$','FontSize',24)
legend('Parameter space $H_X$','Observation space $H_Y$','FontSize',20)
set(gca,'LineWidth',2)
hold off
print('-depsc','param_vs_data_diagonals')

% plot eigenvectors
figure()
hold on
for i=1:5
    plot(model.tt(2:end), Ux(:,i),'linewidth',3)
end
hold off
xlim([0,1])
set(gca,'FontSize',20)
xlabel('Time, $t$','FontSize',24)
ylabel('Parameter vectors $u_i$','FontSize',24)
set(gca,'LineWidth',2)
print('-depsc','CMI_param_eigvectors')

figure()
hold on
for i=1:5
    plot(model.tt(2:end), Uy(:,i),'linewidth',3)
end
hold off
xlim([0,1])
set(gca,'FontSize',20)
xlabel('Time, $t$','FontSize',24)
ylabel('Observation vectors $v_i$','FontSize',24)
set(gca,'LineWidth',2)
print('-depsc','CMI_data_eigvectors')

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
loglik = -0.5*dot(misfit, obs.Cobs \ misfit).';

%% Compute other projectors

% compute covariances
C_xx = prior.C;
C_xy = cov([u_pr; y_pr].'); C_xy = C_xy(1:prior.NP,prior.NP+1:end);
C_yy = cov(y_pr.');

% PCA
[Ux_pca,~] = svd(C_xx);
[Uy_pca,~] = svd(C_yy);

 % CCA
[Ux_cca, Uy_cca] = cca_cov(C_xx, C_xy, C_yy);

figure()
hold on
for i=1:5
    plot(model.tt(2:end), Ux_pca(:,i),'linewidth',3)
end
hold off
xlim([0,1])
set(gca,'FontSize',20)
xlabel('Time, $t$','FontSize',24)
ylabel('Parameter vectors $u_i$','FontSize',24)
set(gca,'LineWidth',2)
print('-depsc','PCA_param_eigvectors')

figure()
hold on
for i=1:5
    plot(model.tt(2:end), Uy_pca(:,i),'linewidth',3)
end
hold off
xlim([0,1])
set(gca,'FontSize',20)
xlabel('Time, $t$','FontSize',24)
ylabel('Observation vectors $v_i$','FontSize',24)
set(gca,'LineWidth',2)
print('-depsc','PCA_data_eigvectors')

figure()
hold on
for i=1:5
    plot(model.tt(2:end), Ux_cca(:,i),'linewidth',3)
end
hold off
xlim([0,1])
set(gca,'FontSize',20)
xlabel('Time, $t$','FontSize',24)
ylabel('Parameter vectors $u_i$','FontSize',24)
set(gca,'LineWidth',2)
print('-depsc','CCA_param_eigvectors')

figure()
hold on
for i=1:5
    plot(model.tt(2:end), Uy_cca(:,i),'linewidth',3)
end
hold off
xlim([0,1])
set(gca,'FontSize',20)
xlabel('Time, $t$','FontSize',24)
ylabel('Observation vectors $v_i$','FontSize',24)
set(gca,'LineWidth',2)
print('-depsc','CCA_data_eigvectors')

% -- END OF FILE --