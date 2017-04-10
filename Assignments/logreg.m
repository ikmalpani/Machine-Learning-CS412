% printing option
more off;

% read files
D_tr = csvread('spambase-train.csv'); 
D_ts = csvread('spambase-test.csv');  

% construct x and y for training and testing
X_tr = D_tr(:, 1:end-1);
y_tr = D_tr(:, end);
X_ts = D_ts(:, 1:end-1);
y_ts = D_ts(:, end);

% number of training / testing samples
n_tr = size(D_tr, 1);
n_ts = size(D_ts, 1);

% add 1 as a feature
X_tr = [ones(n_tr, 1) X_tr];
X_ts = [ones(n_ts, 1) X_ts];

% perform gradient descent :: logistic regression
n_vars = size(X_tr, 2);              % number of variables
lr = 1e-3;                           % learning rate
w = zeros(n_vars, 1);                % initialize parameter w
tolerance = 1e-2;                    % tolerance for stopping criteria

iter = 0;                            % iteration counter
max_iter = 1000;                     % maximum iteration
while true
    iter = iter + 1;                 % start iteration

    % calculate gradient
    grad = zeros(n_vars, 1);         % initialize gradient
    for j=1:n_vars
        % grad(j) = ....             % compute the gradient with respect to w_j here
    end

    % take step
    % w_new = w + .....              % take a step using the learning rate
    
    printf('iter = %d, mean abs gradient = %0.3f\n', iter, mean(abs(grad)));
    fflush(stdout);

    % stopping criteria and perform update if not stopping
    if mean(abs(grad)) < tolerance
        w = w_new;
        break;
    else
        w = w_new;
    end

    if iter >= max_iter 
        break;
    end
end

% use w for prediction
pred = zeros(n_ts, 1);               % initialize prediction vector
for i=1:n_ts
    % pred(i) = .....                % compute your prediction
end

% calculate testing accuracy
% ...

% repeat the similar prediction procedure to get training accuracy
% ...
