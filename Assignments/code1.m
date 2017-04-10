% printing option
more off;

% read files
D_tr = csvread('spambasetrain.csv'); 
D_ts = csvread('spambasetest.csv');  

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
    grad = X_tr.'*(y_tr - (exp(X_tr*w))./(1+ exp(X_tr*w)));

    % take step
    w_new = w + (lr*grad);              % take a step using the learning rate
    
    %printf('iter = %d, mean abs gradient = %0.3f\n', iter, mean(abs(grad)));
    fflush(stdout);

    %stopping criteria and perform update if not stopping
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
pred_train = zeros(n_tr, 1);               % initialize prediction vector
pred_test = zeros(n_tr, 1);               

pred_train = (exp(X_tr*w)./(1+exp(X_tr*w))) >= 0.5;
pred_test = (exp(X_ts*w)./(1+exp(X_ts*w))) >= 0.5;

for i=1:y_tr
  accuracy_train = y_tr == pred_train;
  acctrain = sum(accuracy_train)/size(accuracy_train,1);
  
end
for i=1:y_ts
  accuracy_test = y_ts == pred_test;
  acctest = sum(accuracy_test)/size(accuracy_test,1);
end

printf('acctest = %f, acctrain = %f \n',acctest,acctrain);