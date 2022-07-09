clear; close all; clc
sd = 1; rng(sd);

%% Creation of the wrench

hmax = 0.08;%0.15; % mesh size: 0.5=coarse , 0.15=fine , 0.08 = ULTRA fine
precomputeEverything = 1; % you need it to be 1 in order to compute gradients, and 2 for parallel precomputation

wrench = Wrench(hmax,precomputeEverything);

% plot the geometry & the mesh
figure(1)

subplot(2,1,1)
wrench.plotGeometry();

subplot(2,1,2)
wrench.plotMesh();

%% Compute and plot the solution for many parameters

nMC = 5;
for i=1:nMC
    % get a realization from it (Gaussian distrib)...
    logE = wrench.logE(); 
    % ... and plot it
    figure
    wrench.plot(logE)
    shading interp
    axis off
    axis tight
    set(gca, 'LooseInset', [0,0,0,0]);
    print('-depsc',['logE_' num2str(i)])
    close all
    figure
    logE = wrench.logE();
    wrench.plotSol(logE)
    shading interp
    axis off
    axis tight
    set(gca, 'LooseInset', [0,0,0,0]);
    print('-depsc',['sol_logE_' num2str(i)])
    close all
end

%% Compute and plot many realizations of the observations

% Displacement along the line
type = 3;

dim_y = 48;
nMC = 50;

data = zeros(nMC,dim_y);
for i=1:nMC
    logE = wrench.logE();
    [data(i,:),~,Xobs,~] = wrench.evalObs(logE,type);
end

%%

figure
hold on
plot(Xobs(1,:),data(1:50,:),'LineWidth',3)
xlim([Xobs(1,1), Xobs(1,end)])
set(gca,'FontSize',20)
xlabel('Line where force is applied','FontSize',24)
ylabel('Vertical displacement','FontSize',24)
set(gca,'LineWidth',2)
hold off