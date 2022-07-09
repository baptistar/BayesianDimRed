clear; close all; clc
sd = 1; rng(sd);

%% Define parameters

dom = [-16,16];
N = 100000;
Nim = 32;

%% Generate samples for gradients

% sample (xoff, yoff)
xoff = range(dom)*rand(N,1) + dom(1);
yoff = range(dom)*rand(N,1) + dom(1);

% set sigma and sample gamma
sigma = 3;
gamma = (5-0.25)*rand(N,1) + 0.25;

% evaluate data and gradients for each sample
data = zeros(N,Nim^2);
grad_loglik = zeros(N,Nim^2,3);

for i=1:N
    % sample and display image
    I_XY = simulate_image(xoff(i), yoff(i), sigma, gamma(i));
    data(i,:) = reshape(I_XY,1,Nim^2);
    % evaluate gradient of log-likelihood
    grad_loglik(i,:,:) = mixed_gradient_image_loglikelihood(I_XY, ...
                    xoff(i), yoff(i), sigma, gamma(i));
end

%% Plot upper bound

% form HY
HY = zeros(Nim^2,Nim^2);
for i=1:N
    grad_loglik_i = squeeze(grad_loglik(i,:,:));
    HY = HY + grad_loglik_i * grad_loglik_i.';
end
HY = HY/N;

% compute eigenvectors and eigenvalues
[U_CMI,S_CMI,~] = svd(HY);

% compute upper bound
UB_CMI = zeros(Nim^2, 1);
for i=1:Nim^2
    UB_CMI(i) = trace(U_CMI(:,(i+1):end).'*HY/N*U_CMI(:,(i+1):end));
end

figure;
hold on
semilogy(UB_CMI, 'LineWidth', 3, 'DisplayName', 'CMI')
set(gca,'YScale','log')
xlim([1,Nim^2])
set(gca,'FontSize',20)
xlabel('Dimension of reduced observations, s','FontSize',24)
ylabel('Trailing eigenvalues, $\sum_{i > s} \lambda_i$','FontSize',24)
xlim([1,400])
set(gca,'LineWidth',2)
hold off
print('-depsc','eigvalues_image')

%% Plot modes 

for i=1:11
    figure
    Ui = reshape(U_CMI(:,i),32,32);
    contourf(Ui)
    colormap('bone')
    axis off
    axis square
    caxis([-0.07,0.07])
    print('-depsc',['eigen_mode' num2str(i) '_CMI'])
    close all
end

%% Plot goal-oriented modes

% for xoff
HY = zeros(Nim^2,Nim^2);
for i=1:N
    grad_loglik_i = squeeze(grad_loglik(i,:,1)).';
    HY = HY + grad_loglik_i * grad_loglik_i.';
end
HY = HY/N;

[U,~,~] = svd(HY);
clim = max(abs(U(:,1:6)),[],'all');
for i=1:6
    figure
    Ui = reshape(U(:,i),32,32);
    contourf(Ui)
    colormap('bone')
    axis off
    axis square
    caxis([-clim,clim])
    print('-depsc',['eigen_mode' num2str(i) '_CMI_xoff'])
end

% for yoff
HY = zeros(Nim^2,Nim^2);
for i=1:N
    grad_loglik_i = squeeze(grad_loglik(i,:,2)).';
    HY = HY + grad_loglik_i * grad_loglik_i.';
end
HY = HY/N;

[U,~,~] = svd(HY);
clim = max(abs(U(:,1:5)),[],'all');
for i=1:6
    figure
    Ui = reshape(U(:,i),32,32);
    contourf(Ui)
    colormap('bone')
    axis off
    axis square
    caxis([-clim,clim])
    print('-depsc',['eigen_mode' num2str(i) '_CMI_yoff'])
end

% for gamma
HY = zeros(Nim^2,Nim^2);
for i=1:N
    grad_loglik_i = squeeze(grad_loglik(i,:,3)).';
    HY = HY + grad_loglik_i * grad_loglik_i.';
end
HY = HY/N;

[U,~,~] = svd(HY);
clim = max(abs(U(:,1:5)),[],'all');
for i=1:6
    figure
    Ui = reshape(U(:,i),32,32);
    contourf(Ui)
    colormap('bone')
    axis off
    axis square
    caxis([-clim,clim])
    print('-depsc',['eigen_mode' num2str(i) '_CMI_gamma'])
end

% -- END OF FILE --