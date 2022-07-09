%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Creation of the wrench %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

hmax = 0.08; % mesh size: 0.5=coarse , 0.15=fine , 0.08 = ULTRA fine
precomputeEverything = 1; % you need it to be 1 in order to compute gradients, and 2 for parallel precomputation

wrench = Wrench(hmax,precomputeEverything);


% plot the geometry & the mesh
figure(1)

subplot(2,1,1)
wrench.plotGeometry();

subplot(2,1,2)
wrench.plotMesh();

%% The parameter is the "log of YoungModulus field"
% get a realization from it (Gaussian distrib)...
logE = wrench.logE(); 
% ... and plot it
clf
wrench.plot(logE)

%% Compute and plot the solution
logE = wrench.logE();
wrench.plotSol(logE)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The observation and its gradient %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Vertical displacement of the solution on the left of the wrench
% type = 1;
% Von Mises stress at some point
% type = 2;
% Displacement along the line
type = 3;

logE = wrench.logE();
[y,g,XObs,L] = wrench.evalObs(logE,type);


% disp('Quantity of Interest')
% disp(y)
% wrench.plot(g)
% hold on
% plot(XObs(1),XObs(2),'or')
% hold off

plot(XObs(1,:),y)

%% Norm on the data space (if type=3)

RX = wrench.M + wrench.K;
[~,~,~,L] = wrench.evalObs(wrench.logE(),type);
L = wrench.B'*L;
RYinv = full(L'*(RX\L));
RY = inv(RYinv);

% Then, measure distances in Im(L) using the metric induced by 
%      <.,.> = (.^T)*RY*(.)

%% Matrix H for dimension reduction
K=500;
Hx = zeros(wrench.dimParam);
Hy = zeros(size(XObs,2));
data = zeros(size(XObs,2),K);
xy = zeros(wrench.dimParam,size(XObs,2),K);
for k=1:K
    logE = wrench.logE();
    [y,g] = wrench.evalObs(logE,type);
    data(:,k) = y;
    xy(:,:,k) = logE .* y.';
%     H = H + g'*g;
    Hx = Hx + g'*RY*g;
    Hy = Hy + g*wrench.Sigma*g.';
end
Hx = Hx/K;
Hy = Hy/K;

%% Perform CCA (using low-dimensional X, Y)

% basis for X
[Ux,D] = svd(wrench.Sigma);
D=diag(D);
ind = D<(200*eps*max(D));
Ux(:,ind)=[];

% basis for Y
C_yy = cov(data');
[Uy,D] = svd(C_yy);
D=diag(D);
ind = D<(200*eps*max(D));
Uy(:,ind)=[];

% extract and project covariances
C_xy = zeros(size(Ux,2),size(Uy,2));
for k=1:K
    C_xy = C_xy + (Ux.' * xy(:,:,k) * Uy);  % since x is mean zero
end
C_xy = C_xy/K;
C_xx = Ux.'*wrench.Sigma*Ux;
C_yy = Uy.'*C_yy*Uy;

% compute CCA vectors
[Ux_cca, Uy_cca] = cca_cov(C_xx, C_xy, C_yy);

% compute PCA vectors
[Ux_pca,~] = svd(C_xx);
[Uy_pca,~] = svd(C_yy);

%% Compute the *informed* directions

%[UU,DD] = svd( wrench.Sigma12'*Hx*wrench.Sigma12);
%DD = diag(DD);

% compute basis for wrench.Sigma
[Ux_basis,Dx_basis] = svd(wrench.Sigma);
dim = size(wrench.Sigma12,2);
Ux_basis = Ux_basis(:,1:dim);
Dx_basis = diag(Dx_basis);
Dx_basis = diag(sqrt(Dx_basis(1:dim)));

% define inner matrix and compute SVD
Hx_inner = (Dx_basis.'*Ux_basis.'*Hx*Ux_basis*Dx_basis);
[UU,DD] = svd( Hx_inner);
DD = diag(DD);

% compute upper bound
UB_CMI = zeros(length(DD), 1);
for i=1:length(DD)
    UB_CMI(i) = sum(DD(i+1:end));
end
% compute bound explicitly 
UB_CMI2 = zeros(length(DD),1);
for i=1:length(DD)
    UB_CMI2(i) = trace(UU(:,i+1:end).' * Hx_inner * UU(:,i+1:end));
end
UB_CCA = zeros(size(Ux_cca,2), 1);
for i=1:size(Ux_cca,2)
    % multiply CCA vectors by Cpr^{T/2} = (Dx_basis * Ux_basis.')
    Ui_cca = Dx_basis * Ux_basis.' * Ux * Ux_cca(:,1:i);
    % make CCA vectors orthogonal
    [Uxorth_cca,~] = svd(Ui_cca);
    Uxorth_cca = Uxorth_cca(:,i+1:end);
    UB_CCA(i) = trace(Uxorth_cca.'* Hx_inner * Uxorth_cca);
end
UB_PCA = zeros(size(Ux_pca,2), 1);
for i=1:size(Ux_pca,2)
    % move vectors to original space
    Ui_pca = Ux_basis.' * Ux * Ux_pca(:,1:i);
    % make PCA vectors orthogonal
    [Uxorth_pca,~] = svd(Ui_pca);
    Uxorth_pca = Uxorth_pca(:,i+1:end);
    UB_PCA(i) = trace(Uxorth_pca.' * Hx_inner * Uxorth_pca);
end

figure()
hold on
semilogy(UB_CMI,'LineWidth',3)
%semilogy(UB_CMI2,'--')
semilogy(UB_PCA,'LineWidth',3)
semilogy(UB_CCA,'LineWidth',3)
set(gca,'YScale','log')
set(gca,'FontSize',20)
xlabel('Reduced parameter dimension, $r$','FontSize',24)
ylabel('Trailing eigenvalues, $\sum_{i > r} \lambda_i$','FontSize',24)
set(gca,'LineWidth',2)
legend('CMI','PCA','CCA')
xlim([1,100])
hold off
print('-depsc','wrench_param_eigenvalues')

for i=1:3
    figure()
    %wrench.plot(wrench.Sigma12 * UU(:,i))
    wrench.plot(Ux_basis * Dx_basis * UU(:,i))
    shading interp
    axis off
    axis tight
    xlabel('Displacement line','FontSize',24)
    ylabel(['Mode ' num2str(i)],'FontSize',24)
    set(gca,'LineWidth',2)
    %title(['CMI Mode ' num2str(i)])
    hold off
    print('-depsc',['wrench_param_CMIeigenvector' num2str(i)])
end

for i=1:3
    figure()
    wrench.plot(wrench.Sigma * Ux * Ux_cca(:,i))
    shading interp
    axis off
    axis tight
    xlabel('Displacement line','FontSize',24)
    ylabel(['CCA Mode ' num2str(i)],'FontSize',24)
    set(gca,'LineWidth',2)
    %title(['CCA Mode ' num2str(i)])
    hold off
    print('-depsc',['wrench_param_CCAeigenvector' num2str(i)])
end

for i=1:3
    figure()
    wrench.plot(Ux_basis * Dx_basis.' * Ux_basis.' * Ux * Ux_pca(:,i))
    shading interp
    axis off
    axis tight
    xlabel('Displacement line','FontSize',24)
    ylabel(['CCA Mode ' num2str(i)],'FontSize',24)
    set(gca,'LineWidth',2)
    %title(['PCA Mode ' num2str(i)])
    hold off
    print('-depsc',['wrench_param_PCAeigenvector' num2str(i)])
end

%% Compute the *informative* directions

% A = chol(RYinv);
%A = chol(RY); % chol(Cobs^{-1})
Lobs = chol(RYinv,'lower');  % RYinv = Cobs
A = inv(Lobs); % inverse cholesky factor of Cobs
%UU = A\UU;

% define inner matrix and compute SVD
Hy_inner = (A*Hy*A.');
[UU,DD] = svd(Hy_inner);
DD = diag(DD);

% compute upper bounds
UB_CMI = zeros(length(DD), 1);
for i=1:length(DD)
    UB_CMI(i) = sum(DD(i+1:end));
end
UB_CCA = zeros(size(Uy_cca,2), 1);
for i=1:size(Uy_cca,2)
    % multiply CCA vectors by Cobs^{T/2}
    Ui_cca = Lobs.' * (Uy * Uy_cca(:,1:i));
    % make CCA vectors orthogonal
    [Uyorth_cca,~] = svd(Ui_cca);
    Uyorth_cca = Uyorth_cca(:,i+1:end);
    UB_CCA(i) = trace(Uyorth_cca.' * Hy_inner * Uyorth_cca);
end
UB_PCA = zeros(size(Uy_pca,2), 1);
for i=1:size(Uy_pca,2)
    % move vectors to original space
    Ui_pca = (Uy * Uy_pca(:,1:i));
    % make PCA vectors orthogonal
    [Uyorth_pca,~] = svd(Ui_pca);
    Uyorth_pca = Uyorth_pca(:,i+1:end);
    UB_PCA(i) = trace(Uyorth_pca.' * Hy_inner * Uyorth_pca);
end

figure()
semilogy(UB_CMI,'LineWidth',3)
hold on
semilogy(UB_PCA,'LineWidth',3)
semilogy(UB_CCA,'LineWidth',3)
set(gca,'FontSize',20)
xlabel('Reduced observation dimension, $s$','FontSize',24)
ylabel('Trailing eigenvalues, $\sum_{i > s} \lambda_i$','FontSize',24)
set(gca,'LineWidth',2)
legend('CMI','PCA','CCA')
xlim([1,size(A,1)-1])
hold off
print('-depsc','wrench_data_eigenvalues')

ind = 5;
figure
h=plotDataModes(A\UU(:,1:ind),wrench);
legend([h{:}],arrayfun(@(x) ['Mode ' num2str(x)],1:5,'UniformOutput',0),'Position',[0.2,0.2,0.6,1.1],'FontSize',14,'Orientation','horizontal');
print('-depsc','wrench_data_eigenvectors_CMI')
figure
% multiply CCA modes by Cobs = inv(RY)
h=plotDataModes(RYinv*(Uy*Uy_cca(:,1:ind))/20,wrench);
%legend([h{:}],arrayfun(@(x) ['Mode ' num2str(x)],1:5,'UniformOutput',0),'Position',[0.2,0.2,0.6,1.1],'FontSize',16,'Orientation','horizontal');
print('-depsc','wrench_data_eigenvectors_CCA')
figure
% multiply PCA modes by Cobs = inv(RY)
h=plotDataModes(A\(Uy*Uy_pca(:,1:ind)),wrench);
%legend([h{:}],arrayfun(@(x) ['Mode ' num2str(x)],1:5,'UniformOutput',0),'Position',[0.2,0.2,0.6,1.1],'FontSize',16,'Orientation','horizontal');
print('-depsc','wrench_data_eigenvectors_PCA')

% figure()
% plot(XObs(1,:),UU(:,1:5),'LineWidth',3)
% hold on
% xlim([XObs(1,1), XObs(1,end)])
% set(gca,'FontSize',20)
% xlabel('Displacement line','FontSize',24)
% ylabel('Vectors, $V_i$','FontSize',24)
% set(gca,'LineWidth',2)
% hold off
% legend(arrayfun(@(x) ['Mode ' num2str(x)],1:5,'UniformOutput',0),'location','northwest','FontSize',16)
% print('-depsc','wrench_data_eigenvectors')
% 
% figure()
% plot(XObs(1,:), A\(Uy*Uy_cca(:,1:5)),'LineWidth',3)
% hold on
% xlim([XObs(1,1), XObs(1,end)])
% set(gca,'FontSize',20)
% xlabel('Displacement line','FontSize',24)
% ylabel('Vectors, $V_i$','FontSize',24)
% set(gca,'LineWidth',2)
% hold off
% legend(arrayfun(@(x) ['Mode ' num2str(x)],1:5,'UniformOutput',0),'location','northwest','FontSize',16)
% %print('-depsc','wrench_data_eigenvectors')
% 
% figure()
% plot(XObs(1,:), A\(Uy*Uy_pca(:,1:5)),'LineWidth',3)
% hold on
% xlim([XObs(1,1), XObs(1,end)])
% set(gca,'FontSize',20)
% xlabel('Displacement line','FontSize',24)
% ylabel('Vectors, $V_i$','FontSize',24)
% set(gca,'LineWidth',2)
% hold off
% legend(arrayfun(@(x) ['Mode ' num2str(x)],1:5,'UniformOutput',0),'location','northwest','FontSize',16)

%%

function h=plotDataModes(u,wrench)

nodesObs = wrench.PDEmodel.Mesh.findNodes('region','Edge',14);
nodesObs = wrench.PDEmodel.Mesh.Nodes(:,nodesObs);


h = pdemesh(wrench.PDEmodel,'ElementLabels','off','NodeLabels','off','EdgeColor','k');
h(1).Color = 0.7*[1 1 1];
h(2).Color = 0.7*[1 1 1];
axis equal

% hold on
% plot(nodesObs(1,:),nodesObs(2,:),'r','linewidth',2)
% plot(nodesObs(1,:),nodesObs(2,1)+u,'b','linewidth',2)
% for ind=1:size(nodesObs,2)
%     plot([nodesObs(1,ind),nodesObs(1,ind)],[nodesObs(2,ind),nodesObs(2,ind)+u(ind)],'k')
% end
% hold off

hold on
plot(nodesObs(1,:),nodesObs(2,:),'-k','linewidth',1)
h={};
for ind=1:size(u,2)
    h{ind}=plot(nodesObs(1,:),nodesObs(2,1)+u(:,ind),'linewidth',2);
%     plot([nodesObs(1,ind),nodesObs(1,ind)],[nodesObs(2,ind),nodesObs(2,ind)+u(ind)],'k')
end
hold off

colorbar off
set(gca,'visible','off')

end

