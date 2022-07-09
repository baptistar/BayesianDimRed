clear; close all; clc
sd = 2; rng(sd)
addpath('../tools')

% dimension
d = 50;   

% observation model 
l0     = 500;
alpha  = 1;
tau    = 1e-6;

% prior model
l0t    = 1;
alphat = 2;
taut   = 1e-6;

%% generate model

% define structs
model = struct;
obs = struct;
prior = struct;

% identity forward model
model.G = eye(d);
model.N = d;
obs.n_data = size(model.G,1);

% observation model
U = random_orthogonal_mat(obs.n_data);
Lambda = l0./(1:obs.n_data).^alpha + tau;
model.I = U*diag(Lambda)*U.';
model.Cobs = U*diag(1./Lambda)*U.';
model.Lobs = chol(model.Cobs,'lower');

% prior covariance
V = random_orthogonal_mat(model.N);
Lambdat = l0t./(1:model.N).^alphat + taut;
prior.C = V*diag(Lambdat)*V.';
prior.L = chol(prior.C,'lower');

% prior mean
prior.mean_u = zeros(model.N,1);

% posterior covariance
K = prior.C * model.G.' / (model.G * prior.C * model.G.' + model.Cobs);
model.post_C = prior.C - K * model.G * prior.C;

%% Compute projector using prior samples

% Generate prior samples
np = model.N;
n_samples = 1e5;
v_pr = randn(np, n_samples);

% transform prior samples to correlated space
u_pr = prior.L * v_pr + prior.mean_u;

% estimate the LIS diagnostic matrix
ui = zeros(model.N,n_samples);
Hx = model.G.' * model.I * model.G;
Hy = model.G * prior.C * model.G.';

% apply transformation to Hx, Hy
Hx = prior.L.' * Hx * prior.L;
L_obs = model.Lobs;
Hy = inv(L_obs) * Hy * inv(L_obs).';

% compute eigenvectors of matrices
[Ux,Dx,~] = svd(Hx);
[Uy,Dy,~] = svd(Hy);

% apply inverse transformation to Ux, Uy
Ux = prior.L * Ux;
Uy = inv(L_obs).' * Uy;

% define cumulative sum of eigenvalue tails
Rx = cumsum(diag(Dx(2:end,2:end)),'reverse');
Ry = cumsum(diag(Dy(2:end,2:end)),'reverse');

% compute LSI constant
Gt = inv(L_obs.') * model.G * prior.L;
Sigma_joint = [eye(d), Gt.'; Gt, Gt*Gt.' + eye(obs.n_data)];
LSI_const = svd(Sigma_joint); 
LSI_const = LSI_const(1);

%% Compute KL divergence for projecting parameters and data

% define grid of ranks
r_vect = 1:d;
s_vect = 1:obs.n_data;
[R,S] = meshgrid(r_vect, s_vect);

% define arrays to store KL and MI
MI_joint_CMI            = zeros(length(r_vect),length(s_vect));
MI_joint_upperbound_CMI = zeros(length(r_vect),length(s_vect));
MI_joint_upperbound_trace = zeros(length(r_vect),length(s_vect));

% evaluate the KL divergence error for each rank
for i=1:length(r_vect)
    for j=1:length(s_vect)
        
        % define r and s
        r = r_vect(i);
        s = s_vect(j);

        % define CMI projectors
        Uxr = Ux(:,1:r).' / prior.C;
        Uys = Uy(:,1:s).';
        Prx = (Ux(:,1:r) * Ux(:,1:r).') / prior.C;
        
        % compute exact MI and upper bound for CMI projector
        MI_joint_CMI(i,j) = MI_joint(model, prior, Uxr, Uys);
        MI_joint_upperbound_CMI(i,j) = MI_par(model, prior, Uxr) + MI_data(model, prior, Uys);
        
        % compute trace upper bound for CMI
        U_perp_cmi = prior.L \ Ux(:,r+1:end);
        V_perp_cmi = inv(model.Lobs).' \ Uy(:,s+1:end);
        MI_joint_upperbound_trace(i,j) = ...
            (trace(U_perp_cmi.' * Hx * U_perp_cmi) + trace(V_perp_cmi.' * Hy * V_perp_cmi));

    end
end

%% Find optimal ranks

% set epsilon, alpha_x, alpha_y
epsilon = logspace(1,-0.8,5);
epsilon2 = logspace(1,-0.5,5);
alpha_x = [0.5,1,2];
alpha_y = [2,1,0.5];

r_opt_MI      = zeros(length(epsilon),length(alpha_x));
s_opt_MI      = zeros(length(epsilon),length(alpha_x));
r_opt_MIbound = zeros(length(epsilon),length(alpha_x));
s_opt_MIbound = zeros(length(epsilon),length(alpha_x));
r_opt_Trbound = zeros(length(epsilon),length(alpha_x));
s_opt_Trbound = zeros(length(epsilon),length(alpha_x));

for i=1:length(epsilon)
    for j=1:length(alpha_x)
        
        % set weights
        w_x = alpha_x(j) / (alpha_x(j) + alpha_y(j));
        w_y = 1 - w_x;
        
        % find pareto front for MI
        valid_idx = find(MI_joint_CMI < epsilon(i));
        valid_R = R(valid_idx);
        valid_S = S(valid_idx);
        objective = valid_R * w_x + valid_S * w_y;
        [~, min_idx] = min(objective);
        
        % find optimal entries
        r_opt_MI(i,j) = valid_R(min_idx);
        s_opt_MI(i,j) = valid_S(min_idx);
        
        % find pareto front for MI upper bound
        valid_idx = find(MI_joint_upperbound_CMI < epsilon(i));
        valid_R = R(valid_idx);
        valid_S = S(valid_idx);
        objective = valid_R * w_x + valid_S * w_y;
        [~, min_idx] = min(objective);
        
        % find optimal entries
        r_opt_MIbound(i,j) = valid_R(min_idx);
        s_opt_MIbound(i,j) = valid_S(min_idx);
        
        % find pareto front for trace upper bound
        valid_idx = find(MI_joint_upperbound_trace < epsilon2(i));
        valid_R = R(valid_idx);
        valid_S = S(valid_idx);
        objective = valid_R * w_x + valid_S * w_y;
        [~, min_idx] = min(objective);
        
        % find optimal entries
        r_opt_Trbound(i,j) = valid_R(min_idx);
        s_opt_Trbound(i,j) = valid_S(min_idx);

    end
end

%% Plot results

% plot individual errors
ax_limits = [1e-3,12];
figure
hold on
contourf(R, S, MI_joint_CMI, logspace(0,-3,10));
set(gca,'FontSize',20)
xlabel('Reduced parameter dimension, $r$','FontSize',24)
ylabel('Reduced data dimension, $s$','FontSize',24)
xlim([min(r_vect),max(r_vect)-1])
ylim([min(s_vect),max(s_vect)-1])
set(gca,'ColorScale','log')
set(gca,'CLim',ax_limits);
colorbar('TickLabelInterpreter', 'latex','FontSize',20);
set(gca,'LineWidth', 2);
hold off
print('-depsc','joint_true_objective_CMIproj')
close all

figure
hold on
contourf(R, S, MI_joint_upperbound_CMI, logspace(1,-3,10));
colors = {[0.4940    0.1840    0.5560],[0.8500    0.3250    0.0980],[0    0.4470    0.7410]};
for j=1:length(alpha_x)
    h(j) = plot(r_opt_MIbound(:,j), s_opt_MIbound(:,j), '--s','MarkerSize',10, 'LineWidth', 3, 'Color', colors{j});
end
set(gca,'FontSize',20)
xlabel('Reduced parameter dimension, $r$','FontSize',24)
ylabel('Reduced data dimension, $s$','FontSize',24)
xlim([min(r_vect),max(r_vect)-1])
ylim([min(s_vect),max(s_vect)-1])
set(gca,'ColorScale','log')
set(gca,'CLim',ax_limits)
colorbar('TickLabelInterpreter', 'latex','FontSize',20);
set(gca,'LineWidth', 2);
legend(h,{'$\alpha_X = 0.2$','$\alpha_X = 0.5$','$\alpha_X = 0.8$'},'location','northeast');
hold off
print('-depsc','joint_upperbound_CMIproj')
close all

figure
hold on
contourf(R, S, MI_joint_upperbound_trace, logspace(1,-3,10));
colors = {[0.4940    0.1840    0.5560],[0.8500    0.3250    0.0980],[0    0.4470    0.7410]};
for j=1:length(alpha_x)
    h(j) = plot(r_opt_Trbound(:,j), s_opt_Trbound(:,j), '--s','MarkerSize',10, 'LineWidth', 3, 'Color', colors{j});
end
set(gca,'FontSize',20)
xlabel('Reduced parameter dimension, $r$','FontSize',24)
ylabel('Reduced data dimension, $s$','FontSize',24)
xlim([min(r_vect),max(r_vect)-1])
ylim([min(s_vect),max(s_vect)-1])
set(gca,'ColorScale','log')
set(gca,'CLim',ax_limits)
colorbar('TickLabelInterpreter', 'latex','FontSize',20);
set(gca,'LineWidth', 2);
legend(h,{'$\alpha_X = 0.2$','$\alpha_X = 0.5$','$\alpha_X = 0.8$'},'location','northeast');
hold off
print('-depsc','joint_TraceUpperbound_CMIproj')
close all

ratio = MI_joint_CMI./MI_joint_upperbound_trace;
figure
hold on
contourf(R, S, ratio, linspace(0.1,0.5,10));
set(gca,'FontSize',20)
xlabel('Reduced parameter dimension, $r$','FontSize',24)
ylabel('Reduced data dimension, $s$','FontSize',24)
xlim([min(r_vect),max(r_vect)-1])
ylim([min(s_vect),max(s_vect)-1])
set(gca,'CLim',[0.1,0.5])
colorbar('TickLabelInterpreter', 'latex','FontSize',20);
set(gca,'LineWidth', 2);
hold off
print('-depsc','joint_ratioupperbound_CMIproj')
close all

%% -- Helper Functions --

function MI = MI_par(model, prior, Ur)
    % Find C_y and C_yxr
    C_y = (model.G * prior.C * model.G.' + model.Cobs);
    C_yxr = model.G * prior.C * Ur.';
    % Find covariance of Y conditioned on projected Xr
    C_xr = Ur * prior.C * Ur.';
    C_ycondxr = C_y - C_yxr * (C_xr \ C_yxr.');
    % compute MI
    MI = -0.5*sum(log(svd(model.Cobs))) + 0.5*sum(log(svd(C_ycondxr)));
end

function MI = MI_data(model, prior, Us)
    % Find C_ys and C_xys
    C_xys = prior.C * model.G.' * Us.';
    C_ys = Us * (model.G * prior.C * model.G.' + model.Cobs) * Us.';
    % compute approximate posterior covariance
    post_C_approx = prior.C - C_xys * (C_ys \ C_xys.');
    % compute MI
    MI = -0.5*sum(log(svd(model.post_C))) + 0.5*sum(log(svd(post_C_approx)));
end

function MI = MI_joint(model, prior, Uxr, Uys)
    % Find projected prior covariance and cross-covariance
    C_xr = Uxr * prior.C * Uxr.';
    C_xrys = Uxr * prior.C * model.G.' * Uys.';
    % Find covariance of projected data
    C_ys = Uys * (model.G * prior.C * model.G.' + model.Cobs) * Uys.';
    % compute approximate posterior covariance
    post_C_approx = C_xr - C_xrys * (C_ys \ C_xrys.');
    % compute MI
    MI_XY = 0.5*sum(log(svd(prior.C))) - 0.5*sum(log(svd(model.post_C)));
    MI_XrYs = 0.5*sum(log(svd(C_xr))) - 0.5*sum(log(svd(post_C_approx)));
    MI = MI_XY - MI_XrYs;
end

% -- END OF FILE --