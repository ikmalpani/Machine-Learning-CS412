function [tree] = ID3(examples, attributes, activeAttributes)
% ID3   Runs the ID3 algorithm on the matrix of examples and attributes
% args:
%       examples            - matrix of 1s and 0s for trues and falses, the
%                             last value in each row being the value of the
%                             classifying attribute
%       attributes          - cell array of attribute strings (no CLASS)
%       activeAttributes    - vector of 1s and 0s, 1 if corresponding attr.
%                             active (no CLASS)
% return:
%       tree                - the root node of a decision tree
% tree struct:
%       value               - will be the string for the splitting
%                             attribute, or 'Gesture' or 'FreeGesture' for leaf
%       left                - left pointer to another tree node (left means
%                             the splitting attribute was false)
%       right               - right pointer to another tree node (right
%                             means the splitting attribute was true)

if (isempty(examples));
    error('No data to train');
end

numberAttributes = length(activeAttributes);
numberExamples = length(examples(:,1));

% Create the tree node
tree = struct('value', 'null', 'left', 'null', 'right', 'null');

% If last value of all rows in examples is 1, return tree labeled 'Gesture'
lastColumnSum = sum(examples(:, numberAttributes + 1));
if (lastColumnSum == numberExamples);
    tree.value = 'Gesture';
    return
end
% If last value of all rows in examples is 0, return tree labeled 'FreeGesture'
if (lastColumnSum == 0);
    tree.value = 'FreeGesture';
    return
end
% If activeAttributes is empty, then return tree with label as most common
% value
if (sum(activeAttributes) == 0);
    if (lastColumnSum >= numberExamples / 2);
        tree.value = 'Gesture';
    else
        tree.value = 'FreeGesture';
    end
    return
end

% Find the current entropy
p1 = lastColumnSum / numberExamples;
if (p1 == 0);
    p1_eq = 0;
else
    p1_eq = -1*p1*log2(p1);
end
p0 = (numberExamples - lastColumnSum) / numberExamples;
if (p0 == 0);
    p0_eq = 0;
else
    p0_eq = -1*p0*log2(p0);
end
currentEntropy = p1_eq + p0_eq;

% Find the attribute that maximizes information gain
gains = -1*ones(1,numberAttributes); 

% Loop through active attributes updating gains
for i=1:numberAttributes;
    r = floor(1+((numberExamples - 1)*rand(1)));
    val = examples(r,i);
    if (activeAttributes(i)) % this one is still active, update its gain
        s0 = 0; s0_and_true = 0;
        s1 = 0; s1_and_true = 0;
        for j=1:numberExamples;
            if (examples(j,i) >= val); % this instance has splitting attr. Gesture %// TO DO
                s1 = s1 + 1;
                if (examples(j, numberAttributes + 1)); %target attr is Gesture
                    s1_and_true = s1_and_true + 1;
                end
            else
                s0 = s0 + 1;
                if (examples(j, numberAttributes + 1)); %target attr is Gesture
                    s0_and_true = s0_and_true + 1;
                end
            end
        end
        
        % Entropy for S(v=1)
        if (~s1);
            p1 = 0;
        else
            p1 = (s1_and_true / s1); 
        end
        if (p1 == 0);
            p1_eq = 0;
        else
            p1_eq = -1*(p1)*log2(p1);
        end
        if (~s1);
            p0 = 0;
        else
            p0 = ((s1 - s1_and_true) / s1);
        end
        if (p0 == 0);
            p0_eq = 0;
        else
            p0_eq = -1*(p0)*log2(p0);
        end
        entropy_s1 = p1_eq + p0_eq;

        % Entropy for S(v=0)
        if (~s0);
            p1 = 0;
        else
            p1 = (s0_and_true / s0); 
        end
        if (p1 == 0);
            p1_eq = 0;
        else
            p1_eq = -1*(p1)*log2(p1);
        end
        if (~s0);
            p0 = 0;
        else
            p0 = ((s0 - s0_and_true) / s0);
        end
        if (p0 == 0);
            p0_eq = 0;
        else
            p0_eq = -1*(p0)*log2(p0);
        end
        entropy_s0 = p1_eq + p0_eq;
        
        gains(i) = currentEntropy - ((s1/numberExamples)*entropy_s1) - ((s0/numberExamples)*entropy_s0);
    end
end

% Pick the attribute with maximum information gain
[~, bestAttribute] = max(gains);
% Set tree.value to bestAttribute's relevant string
tree.value = attributes{bestAttribute};
% Remove splitting attribute from activeAttributes
activeAttributes(bestAttribute) = 0;

% Initialize and create the new example matrices
examples_0 = []; examples_0_index = 1;
examples_1 = []; examples_1_index = 1;
for i=1:numberExamples;
    if (examples(i, bestAttribute));
        examples_1(examples_1_index, :) = examples(i, :); 
        examples_1_index = examples_1_index + 1;
    else
        examples_0(examples_0_index, :) = examples(i, :);
        examples_0_index = examples_0_index + 1;
    end
end

% For value = 0 or FreeGesture, corresponds to left branch
% If examples_0 is empty, add leaf node to the left with relevant label
if (isempty(examples_0));
    leaf = struct('value', 'null', 'left', 'null', 'right', 'null');
    if (lastColumnSum >= numberExamples / 2); 
        leaf.value = 'Gesture';
    else
        leaf.value = 'FreeGesture';
    end
    tree.left = leaf;
else
    tree.left = ID3(examples_0, attributes, activeAttributes);
end
% For value = 1 or Gesture, corresponds to right branch
% If examples_1 is empty, add leaf node to the right with relevant label
if (isempty(examples_1));
    leaf = struct('value', 'null', 'left', 'null', 'right', 'null');
    if (lastColumnSum >= numberExamples / 2);
        leaf.value = 'Gesture';
    else
        leaf.value = 'FreeGesture';
    end
    tree.right = leaf;
else
    tree.right = ID3(examples_1, attributes, activeAttributes);
end
return
end