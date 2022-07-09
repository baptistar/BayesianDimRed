clear; close all; clc
sd = 1; rng(sd);

%% Define parameters

dom = [-16,16];
N = 100000;
Nim = 32;

%% Plot samples

Nsamples = 10;
for i=1:Nsamples
    % sample (xoff, yoff)
    xoff = range(dom)*rand(1) + dom(1);
    yoff = range(dom)*rand(1) + dom(1);
    % set sigma and sample gamma
    sigma = 3;
    gamma = (5-0.25)*rand(1) + 0.25;
    % sample image
    I_XY = simulate_image(xoff, yoff, sigma, gamma);
    figure;
    imshow(I_XY, [0,1], 'InitialMagnification', 1000)
    text(3, 15,['$x_1 = ' num2str(xoff,2) '$'],'FontSize',15)
    text(3, 18,['$x_2 = ' num2str(yoff,2) '$'],'FontSize',15)
    text(3, 21,['$ \gamma = ' num2str(gamma,2) '$'],'FontSize',15)
    print('-depsc',['image' num2str(i)])
    close all
end

% -- END OF FILE --