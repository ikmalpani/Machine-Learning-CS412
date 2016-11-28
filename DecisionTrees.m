pkg load statistics;

%inputFileName = 'small_test_1';
inputFileName = 'test_data4';
numberOfTrials = 20;

% Read the specified input file
fid = fopen(inputFileName);
dataInput = textscan(fid, '%s');
fclose(fid);

i = 1;
%no of attributes
while (~strcmp(dataInput{1}{i}, 'CurrentLabel'));
    i = i + 1;
end
disp("Number of attributes");
disp(i);  

%Store list of attributes in cell array
attributes = cell(1,i);
for j=1:i;
    attributes{j} = dataInput{1}{j};
end
%disp(attributes);  %list of attributes

%NOTE: The classification will be the last attribute in the data rows below
numAttributes = i;
numInstances = (length(dataInput{1}) - numAttributes) / numAttributes;
disp("Number of sample records");
disp(numInstances);  % no of records in the input file

%Store the data into matrix
data = [];
i = i + 1;

disp('-------------------------------------------------------');
for j=1:numInstances
    for k=1:numAttributes
        data(j,k) = strread(dataInput{1}{i}, "%n");
        i = i + 1;
    end    
end

%disp(data);

trainingSetSize = floor(numInstances * 0.8);
% Iterating for multiple trials
for i=1:numberOfTrials;
    fprintf('TRIAL NUMBER: %d\n\n', i);
    
    % Split data into training and testing sets randomly
    % Use randsample to get a vector of row numbers for the training set
    rows = sort(randsample(numInstances, trainingSetSize));
    % Initialize two new matrices, training set and test set
    trainingSet = zeros(trainingSetSize, numAttributes);
    testingSetSize = (numInstances - trainingSetSize);
    testingSet = zeros(testingSetSize, numAttributes);
    % Loop through data matrix, copying relevant rows to each matrix
    training_index = 1;
    testing_index = 1;
    for data_index=1:numInstances;
        if (rows(training_index) == data_index);
            trainingSet(training_index, :) = data(data_index, :);
            if (training_index < trainingSetSize);
                training_index = training_index + 1;
            end
        else
            testingSet(testing_index, :) = data(data_index, :);
            if (testing_index < testingSetSize);
                testing_index = testing_index + 1;
            end
        end
    end
    
    % Construct a decision tree on the training set using the ID3 algorithm
    activeAttributes = ones(1, length(attributes) - 1);
    new_attributes = attributes(1:length(attributes)-1);
    tree = ID3(trainingSet, attributes, activeAttributes);
    
    %Print the tree (Note: It is huge and it gets printed for every trial)
    %fprintf('DECISION TREE STRUCTURE:\n');
    %PrintTree(tree, 'root');
    
    %Classify the training data using the decision tree 
    % The second column is for actual classification, first for calculated
    ID3_Classifications = zeros(testingSetSize,2);
    ID3_numCorrect = 0; 
    for k=1:testingSetSize; %over the testing set
        % Call a recursive function to follow the tree nodes and classify
        ID3_Classifications(k,:) = ...
            ClassifyByTree(tree, new_attributes, testingSet(k,:));
        
        if (ID3_Classifications(k,1) == ID3_Classifications(k, 2)); %correct
            ID3_numCorrect = ID3_numCorrect + 1;
        end           
    end
    
    % No of correct  classifications
    if (testingSetSize);
        ID3_Percentage = round(100 * ID3_numCorrect / testingSetSize);
    else
        ID3_Percentage = 0;
    end
    ID3_Percentages(i) = ID3_Percentage;
    
    fprintf('\tPercentage of test cases correctly classified by ID3 decision tree = %d\n' ...
        , ID3_Percentage);
end
 
 meanID3 = round(mean(ID3_Percentages));
 
 %Print the accuracy of the classification on test data
 fprintf('Filename = %s\n', inputFileName);
 fprintf('Number of trials = %d\n', numberOfTrials);
 fprintf('Training set size for each trial = %d\n', trainingSetSize);
 fprintf('Testing set size for each trial = %d\n', testingSetSize);
 fprintf('Testing Accuracy of decision tree over all trials = %d\n', meanID3);