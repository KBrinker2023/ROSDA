function Theta0 = Theta_0(n,W,X,Py,Q)
    I = eye(n);
    Pwx = W * X * inv(X' * W^2 * X) * (W*X)';
    %smallest right singular vectors
    [~,~,V] = svd((I - Pwx)*W*Py);
    Theta0 = V(:,end-Q+1:end);
end