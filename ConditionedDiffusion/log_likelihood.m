function llkd = log_likelihood(model, obs, u)
% Evaluates log-likelihood
% 
% Inputs:  model - struct for forward model
%          obs - struct for observation model
%          v - parameter samples in whitened space
% Outputs: ll - value of log-likelihood

% minus log likelihood for a given observable model output 
% gradient w.r.t. observable model output
% Fisher information matrix at observable model output

% define array to store log-likelihoods
n = size(u,2);
llkd = zeros(n,1);

% evaluate forward model
for i=1:n

    sol = forward_solve(model, u(:,i));

    switch obs.like
        case {'normal'}
            misfit = (sol.d - obs.data)./obs.std;
            llkd(i)  = -0.5*sum(reshape(misfit, obs.n_data, []).^2, 1);
        case {'poisson'}
            d = reshape(sol.d, obs.n_data, []);
            llkd(i) = obs.data(:)'*log(d) - sum(d, 1);
        otherwise
            error('likelihood not implemented')
    end
    
end

end