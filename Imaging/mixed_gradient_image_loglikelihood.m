function grad_loglik = mixed_gradient_image_loglikelihood(I_XY, xoff, yoff, sigma, gamma, N)
% Evaluate gradient of likelihood for 32x32 pixel image I_XY with respect
% to both image and parameters = (xoff, yoff)
if nargin < 6
    N = 32;
end

% define grid
[X,Y] = meshgrid(linspace(-16,16,N));

% evaluate rxy and probability of blob
r_XY = (X - xoff).^2 + (Y - yoff).^2;
p_XY = 0.9 - 0.8*exp(-0.5*(r_XY/sigma^2).^gamma);

% evaluate gradients of p_XY with respect to r_XY
grad_r_p     = -0.8*exp(-0.5*(r_XY/sigma^2).^gamma) * (-0.5/(sigma^2).^gamma) * gamma .* r_XY.^(gamma-1);

% evaluate gradients of r_XY with respect to parameters
grad_xoff_r  = 2*(X - xoff)*(-1);
grad_yoff_r  = 2*(Y - yoff)*(-1);

% evaluate gradient of p_XY with respect to parameters
grad_xoff_p  = grad_r_p .* grad_xoff_r;
grad_yoff_p  = grad_r_p .* grad_yoff_r;

% evaluate gradient of p_XY with respect to gamma
grad_gamma_p = -0.8*exp(-0.5*(r_XY/sigma^2).^gamma) * (-0.5) .* log(r_XY/sigma^2) .* (r_XY/sigma^2).^gamma;

% evaluate gradients of ContBernoulli with respect to Image and p
b = ContinuousBernoulli(p_XY);
grad_p_grad_I_logpdf = b.grad_x_grad_lambda_log_pdf(I_XY);

% evaluate gradient of binopdf with respect to Image and parameters
grad_xoff_grad_I_logpdf  = grad_p_grad_I_logpdf .* grad_xoff_p;
grad_yoff_grad_I_logpdf  = grad_p_grad_I_logpdf .* grad_yoff_p;
grad_gamma_grad_I_logpdf = grad_p_grad_I_logpdf .* grad_gamma_p;

% assemble gradients
grad_loglik = [grad_xoff_grad_I_logpdf(:), grad_yoff_grad_I_logpdf(:), grad_gamma_grad_I_logpdf(:)]/N^2;

end