classdef ContinuousBernoulli
    % Defines continuous Bernoulli distribution
    % with density p(x) = C(lambda) * \lambda^x * (1 - lambda)^(1-x)
    
    
    properties
        lambda % normalizing constant
    end

    methods
        function self = ContinuousBernoulli(lambda)
            self.lambda = lambda;
        end
        % -----------------------------------------------------------------
        function c = norm_const(self)
            c = 2*atanh(1 - 2*self.lambda)./(1-2*self.lambda);
            c(self.lambda == 0.5) = 2;            
        end
        % -----------------------------------------------------------------
        function logpi = log_pdf(self, x)
            logc = log(self.norm_const());
            logpi = logc + x.*log(self.lambda) + (1-x).*log(1-self.lambda);
        end
        % -----------------------------------------------------------------
        function grad2_logpi = grad_x_grad_lambda_log_pdf(self, ~)
            grad2_logpi = 1./self.lambda + 1./(1-self.lambda);
        end
        % -----------------------------------------------------------------
        function finv = inverse_cdf(self, u)
            a = self.lambda;
            if isscalar(self.lambda) && self.lambda == 0.5
                finv = u;
            else
                finv = -log(-(a - 1 + (1-2*a).*u)./(1-a))./(log(1-a) - log(a));
                finv(self.lambda == 0.5) = u;
            end
        end
        % -----------------------------------------------------------------
        function x = sample(self, N)
            u = rand(N,1);
            x = self.inverse_cdf(u);
        end
        % -----------------------------------------------------------------
    end
    
end

