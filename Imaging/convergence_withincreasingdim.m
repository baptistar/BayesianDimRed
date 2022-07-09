clear; close all; clc
sd = 1; rng(sd);

%% Define parameters

dom = [-16,16];
N   = 100000;
Nim = [6,8,10,20,32];
max_eig = 50;

%% form HY for each Nim

% define vector to store eigs
S_CMI = cell(length(Nim),1);
S_PCA = cell(length(Nim),1);

for d=1:length(Nim)
    fprintf('Dim = %d\n', Nim(d))
    
    % sample (xoff, yoff)
    xoff = range(dom)*rand(N,1) + dom(1);
    yoff = range(dom)*rand(N,1) + dom(1);

    % set sigma and sample gamma
    sigma = 3;
    gamma = (5-0.25)*rand(N,1) + 0.25;

    % evaluate data and gradients for each sample
    data = zeros(N,Nim(d)^2,1);
    grad_loglik = zeros(N,Nim(d)^2,3);
    HY = zeros(Nim(d)^2,Nim(d)^2);
    for i=1:N
        % sample and display image
        I_XY = simulate_image(xoff(i), yoff(i), sigma, gamma(i), Nim(d));
        data(i,:,:) = reshape(I_XY,1,Nim(d)^2);
        % evaluate gradient of log-likelihood
        grad_loglik(i,:,:) = mixed_gradient_image_loglikelihood(I_XY, ...
                        xoff(i), yoff(i), sigma, gamma(i), Nim(d));
        % add gradient to log-lik
        HY = HY + squeeze(grad_loglik(i,:,:)) * squeeze(grad_loglik(i,:,:)).';
    end
    HY = HY/N;
        
    % compute eigenvalues of HY
    S_CMI{d} = svds(HY, max_eig);
    
end

%% Plot HY

figure;
hold on
for d=1:length(Nim)
    semilogy(S_CMI{d}/max(S_CMI{d}), 'LineWidth', 3, 'DisplayName', ['$m = ' num2str(Nim(d)^2) '$'])
end
set(gca,'YScale','log')
xlim([1,max_eig])
set(gca,'FontSize',20)
xlabel('Dimension of reduced observations, s','FontSize',24)
ylabel('Normalized eigenvalues, $\lambda_i/\lambda_1$','FontSize',24)
legend('show','FontSize',22,'location','northeast')
set(gca,'LineWidth',2)
hold off
print('-depsc','eigs_vs_dim')

% -- END OF FILE --