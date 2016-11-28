function [] = PrintTree(tree, parent)
% Prints the tree structure (preorder traversal)

% Print current node
if (strcmp(tree.value, 'Gesture'));
    fprintf('parent: %s\t Gesture\n', parent);
    return
elseif (strcmp(tree.value, 'FreeGesture'));
    fprintf('parent: %s\t   FreeGesture\n', parent);
    return
else
    % Current node an attribute splitter
    fprintf('parent: %s\t   Attribute: %s\t   FreeGestureChild:%s\t Gesture   Child:%s\n', ...
        parent, tree.value, tree.left.value, tree.right.value);
end

% Recur the left subtree
PrintTree(tree.left, tree.value);

% Recur the right subtree
PrintTree(tree.right, tree.value);

end