function [classifications] = ClassifyByTree(tree, attributes, instance)
% Store the actual classification
actual = instance(1, length(instance));

% Recursion with 3 cases

% Case 1: Current node is labeled 'Gesture'
% Return the classification as 1
if (strcmp(tree.value, 'Gesture'));
    classifications = [1, actual];
    return
end

% Case 2: Current node is labeled 'FreeGesture'
% Return the classification as 0
if (strcmp(tree.value, 'FreeGesture'));
    classifications = [0, actual];
    return
end

% Case 3: Current node is labeled an attribute
% Follow correct branch by looking up index in attributes, and recur
index = find(ismember(attributes,tree.value)==1);
if (instance(1, index)); % attribute is true for this instance
    % Recur down the right side
    classifications = ClassifyByTree(tree.right, attributes, instance); 
else
    % Recur down the left side
    classifications = ClassifyByTree(tree.left, attributes, instance);
end

return
end