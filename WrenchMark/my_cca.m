function [U,V,D]=my_cca(C_xx,C_yy,C_yx,tol)

%Cm12_xx = inv(chol(C_xx));
Cm12_xx = truncated_invSQRT(C_xx,tol);
%Cm12_yy = inv(chol(C_yy));
Cm12_yy = truncated_invSQRT(C_yy,tol);

C_yx = Cm12_yy'*C_yx*Cm12_xx;

[V,D,U] = svd(C_yx);
D = diag(D);

V = Cm12_yy * V;
U = Cm12_xx * U;

end

function Cm12 = truncated_invSQRT(C,tol)

[U,D] = svd(C);
D = diag(D);
ind = find(D./D(1) > tol);
U = U(:,ind);
D = D(ind);

Cm12 = U*diag(1./sqrt(D));
%Cm12 = U*diag(sqrt(D));

end