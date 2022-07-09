function Q = random_orthogonal_mat(d)

% sample matrix 
[Q,R] = qr(randn(d));
Q = Q*diag(sign(diag(R)));

end
