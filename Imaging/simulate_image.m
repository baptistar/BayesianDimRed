function I_XY = simulate_image(xoff, yoff, sigma, gamma, N)
% Simulate 32x32 pixel image given location of blob (xoff, yoff) and 
% the two parameters (sigma, gamma).
if nargin < 5
    N = 32;
end

% define grid
[X,Y] = meshgrid(linspace(-16,16,N));

% evaluate rxy and probability of blob
r_XY = (X - xoff).^2 + (Y - yoff).^2;
p_XY = 0.9 - 0.8*exp(-0.5*(r_XY/sigma.^2).^gamma);

% sample intensities
b = ContinuousBernoulli(p_XY);
I_XY = b.sample(1);

end