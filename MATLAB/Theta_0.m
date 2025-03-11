function Theta0 = Theta_0(n,W,X,Py,Q)
    I = eye(n);
    Pwx = X * inv(X' * W * X) * X' * W;
    %smallest right singular vectors
    [~,~,V] = svd((I - Pwx)*W*Py);
    Theta0 = V(:,end-Q+1:end);
end