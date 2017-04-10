clear all;
clc;

fid = fopen('SMSSpamCollection');            % read file
data = fread(fid);
fclose(fid);
lcase = abs('a'):abs('z');
ucase = abs('A'):abs('Z');
caseDiff = abs('a') - abs('A');
caps = ismember(data,ucase);
data(caps) = data(caps)+caseDiff;     % convert to lowercase
data(data == 9) = abs(' ');          % convert tabs to spaces
validSet = [9 10 abs(' ') lcase];         
data = data(ismember(data,validSet)); % remove non-space, non-tab, non-(a-z) characters
data = char(data);                    % convert from vector to characters

words = strsplit(data');             % split into words

% Array of alpha
exp=[-5 -4 -3 -2 -1 0];
k=numel(exp);
for ind=1:k
    alpha_arr(ind)=2^exp(ind);
end

% alpha=0.1;

% split into examples
count = 0;
examples = {};

for (i=1:length(words))
   if (strcmp(words{i}, 'spam') || strcmp(words{i}, 'ham'))
       count = count+1;
       examples(count).spam = strcmp(words{i}, 'spam');
       examples(count).words = [];
   else
       examples(count).words{length(examples(count).words)+1} = words{i};
   end
end

%split into training and test
random_order = randperm(length(examples));
train_examples = examples(random_order(1:floor(length(examples)*.8)));
test_examples = examples(random_order(floor(length(examples)*.8)+1:end));

% count occurences for spam and ham
for ind=1:k
    alpha=alpha_arr(ind);
spamcounts = javaObject('java.util.HashMap');
numspamwords = 0;
hamcounts = javaObject('java.util.HashMap');
numhamwords = 0;



for (i=1:length(train_examples))
    for (j=1:length(train_examples(i).words))
        word = train_examples(i).words{j};
        if (train_examples(i).spam == 1)
            numspamwords = numspamwords+1;
            current_count = spamcounts.get(word);
            if (isempty(current_count))
                spamcounts.put(word, 1+alpha);    % initialize by including pseudo-count prior
            else
                spamcounts.put(word, current_count+1);  % increment
            end
        else
            numhamwords = numhamwords+1;
            current_count = hamcounts.get(word);
            if (isempty(current_count))
                hamcounts.put(word, 1+alpha);    % initialize by including pseudo-count prior
            else
                hamcounts.put(word, current_count+1);  % increment
            end
        end
    end    
end

spamcounts.get('free')/(numspamwords+alpha*20000);   % probability of word 'free' given spam
hamcounts.get('free')/(numhamwords+alpha*20000);   % probability of word 'free' given ham

no_spam=0;
% Probability of Spam/Ham
for i=1:length(train_examples)
if train_examples(i).spam
no_spam=no_spam+1;
end
end

p_spam=no_spam/length(train_examples);
p_ham=1-p_spam;

% Predict the Train Examples
cnt_spam=ones(length(train_examples),1);
cnt_ham=ones(length(train_examples),1);
for (i=1:length(train_examples))
    for (j=1:length(train_examples(i).words))
        word = train_examples(i).words{j};
            c=spamcounts.get(word);
            if (isempty(c))
                c=1+alpha;
            end
            cnt_spam(i)=cnt_spam(i)*c;
      
 
            c_ham=hamcounts.get(word);
            if (isempty(c_ham))
                c_ham=1+alpha;
            end
            cnt_ham(i)=cnt_ham(i)*c_ham;
    end
    
    cnt_spam(i)=cnt_spam(i)/((numspamwords+alpha*20000)^length(train_examples(i).words));
    cnt_ham(i)=cnt_ham(i)/((numhamwords+alpha*20000)^length(train_examples(i).words));
end

% Compute Posterior
train_spam_prob=cnt_spam.*p_spam;
train_ham_prob=cnt_ham.*p_ham;

% Perform argmax with spam being the positive class
pred_train=train_spam_prob>=train_ham_prob;

% Calcualate True Postive, True Negative, False Positive, False Negative
for i=1:length(train_examples)
tp_train_vec(i)=(pred_train(i)==1) && (train_examples(i).spam==1);
tn_train_vec(i)=(pred_train(i)==0) && (train_examples(i).spam==0);
fn_train_vec(i)=(pred_train(i)==0) && (train_examples(i).spam==1);
fp_train_vec(i)=(pred_train(i)==1) && (train_examples(i).spam==0);
end

tp_train=sum(tp_train_vec);
tn_train=sum(tn_train_vec);
fn_train=sum(fn_train_vec);
fp_train=sum(fp_train_vec);

% Calculate Precision,Recall, Accuracy, and F-Score
precision_train=tp_train/(tp_train+fp_train);
recall_train=tp_train/(tp_train+fn_train);
f_score_train(ind)=(2*precision_train*recall_train)/(precision_train+recall_train);
accuracy_train(ind)=(tp_train+tn_train)/length(train_examples);

% Perform the same for Test Examples

cnt_spam_test=ones(length(test_examples),1);
cnt_ham_test=ones(length(test_examples),1);
for (i=1:length(test_examples))
    for (j=1:length(test_examples(i).words))
        word = test_examples(i).words{j};
            c=spamcounts.get(word);
            if (isempty(c))
                c=1+alpha;
            end
            cnt_spam_test(i)=cnt_spam_test(i)*c;
      
 
            c_ham=hamcounts.get(word);
            if (isempty(c_ham))
                c_ham=1+alpha;
            end
            cnt_ham_test(i)=cnt_ham_test(i)*c_ham;
    end
    
    cnt_spam_test(i)=cnt_spam_test(i)/((numspamwords+alpha*20000)^length(test_examples(i).words));
    cnt_ham_test(i)=cnt_ham_test(i)/((numhamwords+alpha*20000)^length(test_examples(i).words));
end

test_spam_prob=cnt_spam_test.*p_spam;
test_ham_prob=cnt_ham_test.*p_ham;

pred=test_spam_prob>=test_ham_prob;

for i=1:length(test_examples)
tp_vec(i)=(pred(i)==1) && (test_examples(i).spam==1);
tn_vec(i)=(pred(i)==0) && (test_examples(i).spam==0);
fn_vec(i)=(pred(i)==0) && (test_examples(i).spam==1);
fp_vec(i)=(pred(i)==1) && (test_examples(i).spam==0);
end

tp=sum(tp_vec);
tn=sum(tn_vec);
fn=sum(fn_vec);
fp=sum(fp_vec);

precision=tp/(tp+fp);
recall=tp/(tp+fn);
f_score(ind)=(2*precision*recall)/(precision+recall);
accuracy(ind)=(tp+tn)/length(test_examples);

end


% Plot Accuracy/F-Score of test and train data with different Alpha
plot(exp, accuracy,'g');
hold on
plot(exp, accuracy_train,'r');
legend('Test Accuracy', 'Train Accuracy');
title('Plot of Accuracy of Train and Test Examples with i');
xlabel('i');
ylabel('Accuracy');

figure;

plot(exp, f_score,'g');
hold on
plot(exp, f_score_train,'r');
legend('Test F-Score', 'Train F-Score');
title('Plot of F-Score of Train and Test Examples with i');
xlabel('i');
ylabel('F-Score');