pkg load statistics

% setup data
D = csvread('iris.csv');
X_train = D(:, 1:2);
y_train = D(:, end); 

% setup meshgrid
[x1, x2] = meshgrid(2:0.01:5, 0:0.01:3);
grid_size = size(x1);
X12 = [x1(:) x2(:)]; % 301*301

% compute 1NN decision 
n_X12 = size(X12, 1); % 90601
decision = zeros(n_X12, 1);
for i=1:n_X12    
    point = X12(i, :);
    
    % compute euclidan distance from the point to all training data
    dist = pdist2(X_train, point);
    
    % sort the distance, get the index
    [~, idx_sorted] = sort(dist);
    
    % find the class of the nearest neighbour
    pred = y_train(idx_sorted(1));
    
    decision(i) = pred;
end

% plot decisions in the grid
figure;
decisionmap = reshape(decision, grid_size);
imagesc(2:0.01:5, 0:0.01:3, decisionmap);
set(gca,'ydir','normal');

% colormap for the classes
% class 1 = light red, 2 = light green, 3 = light blue
cmap = [1 0.8 0.8; 0.8 1 0.8; 0.8 0.8 1];
colormap(cmap);

% satter plot data
hold on;
scatter(X_train(y_train == 1, 1), X_train(y_train == 1, 2), 10, 'r');
scatter(X_train(y_train == 2, 1), X_train(y_train == 2, 2), 10, 'g');
scatter(X_train(y_train == 3, 1), X_train(y_train == 3, 2), 10, 'b');
hold off;