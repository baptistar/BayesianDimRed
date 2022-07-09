function [W_x, W_y] = cca_cov(C_xx, C_xy, C_yy)
    n_par = size(C_xx,1);
    n_obs = size(C_yy,1);

    % compute CCA vectors
    if n_par <= n_obs
        % compute W_x
        [rhosq, W_x] = gen_eig(C_xy*(C_yy\C_xy'), C_xx);
        % compute W_y
        inv_rho = diag(1./sqrt(diag(rhosq)));
        W_y = C_yy\(C_xy'*W_x)*inv_rho;
    else
        % compute W_y
        [rhosq, W_y] = gen_eig(C_xy'*(C_xx\C_xy), C_yy);
        % compute W_x
        inv_rho = diag(1./sqrt(diag(rhosq)));
        W_x = C_xx\(C_xy*W_y)*inv_rho;
    end

end

function [D,W] = gen_eig(A, B)
% Compute the generalized eigenvalues and eigenvectors of the matrix
% pencil (A,B). That is, (w_i,\delta_i) s.t. A*w_i = B*w_i*\delta_i
% where A and B are SPD.

% compute cholesky decomposition of B
S_B = chol(B, 'lower');
Sinv_B = inv(S_B);

% compute generalized eigenvalues
if nargout == 1
    D = svd(Sinv_B*A*Sinv_B.',0);%eig(Sinv_B*A*Sinv_B.');
else
    [W,D] = svd(Sinv_B*A*Sinv_B.',0);%eig(Sinv_B*A*Sinv_B.');
    W = Sinv_B.'*W;
end

end