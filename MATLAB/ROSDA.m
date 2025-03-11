%OSDA Algorithm Function as presented by Huang & Zhang 2020
%Establish Matrices and Parameters
tolerance = 1e-14;
max_iteration = 10000;
RSSold = 2;
RSS = 10;
iter = 0;
K = 2; %Number of classes

%Data
% Synthetic X Matrix
X = randi([0 10],100, 40); %Replace with Image Matrix
X = normalize(X);

%Synthetic Y Matrix - Replace with Target Variable
num_samples = size(X,1);
num_classes = K; % Classes: 0, 1 
Y_labels = randi([0, 1], num_samples, 1); %Column vector of class labels

% Convert to dummy variable (one-hot encoding)
Y = zeros(num_samples, num_classes); % Initialize matrix
for i = 1:num_samples
    Y(i, Y_labels(i) + 1) = 1; % "+1" because MATLAB indexing starts from 1
end

n = size(X,1);
Q = K-1;
betaj = ones(size(X,2),Q);
Thetaj = ones(K,Q);
D = (1/n)*transpose(Y)*Y;
%ROSDA Loop
while (abs(RSSold - RSS)/RSS > tolerance && iter < max_iteration)
    for i = 1:n
        z(i) = norm(Y(i, :)*Thetaj-X(i, :)*betaj,"fro")^2;%[1]
    end
    for i = 1:n
        w(i) = Psi_dx(z(i));
    end
    W = diag(w);%[2]
    RSSold = RSS;
    betaj = inv(transpose(X)*W*X)*transpose(X)*W*Y*Thetaj;
    [Py,~,~] = svd(Y);
    I = eye(n);
    Theta0 = Theta_0(n,W,X,Py,Q);
    Thetaj = (1/sqrt(n))*inv(D)*transpose(Y)*Py*Theta0;
    for i = 1:n % Recalculate z using theta(j) and beta(j)
        z(i) = norm(Y(i, :) * Thetaj - X(i, :) * betaj, "fro")^2;
    end
    RSS = (1/n) * sum(Psi(z(i)));
    iter=iter+1;
end

% Predict classes by finding which Thetak is closest to X_iB
XB = X * betaj; 
predicted_classes = zeros(n, K);
for i = 1:n
    [~, index] = min(abs(Thetaj - XB(i, :)));
    predicted_classes(i, index) = 1; 
end

%References:
%1. https://www.mathworks.com/help/matlab/ref/norm.html
%2. %https://www.mathworks.com/help/matlab/ref/diag.html
%3. https://www.mathworks.com/help/matlab/ref/double.svd.html#d126e1761848
%4. https://www.mathworks.com/help/matlab/matlab_prog/matlab-operators-and-special-characters.html
% 5. https://www.mathworks.com/matlabcentral/answers/152301-find-closest-value-in-array