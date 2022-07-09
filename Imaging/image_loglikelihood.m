function loglik = image_loglikelihood(I_XY, xoff, yoff, sigma, gamma)
% Evaluate likelihood for 32x32 pixel image I_XY given location of blob
% (xoff, yoff) and the two parameters (sigma, gamma).

% define grid
N = 32;
[X,Y] = meshgrid(linspace(-16,16,N));

% evaluate rxy and probability of blob
r_XY = (X - xoff).^2 + (Y - yoff).^2;
p_XY = 0.9 - 0.8*exp(-0.5*(r_XY/sigma^2).^gamma);

% evaluate log-likelihood 
b = ContinuousBernoulli(p_XY);
loglik = b.log_pdf(I_XY);

% evaluate sum of log-likelihoods for independent pixels
loglik = sum(loglik, 'all')/N^2;
