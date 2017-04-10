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
alpha=0.1;

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

% Probability of Spam/Ham
no_spam=0;
for i=1:length(train_examples)
if train_examples(i).spam
no_spam=no_spam+1;
end
end

p_spam=no_spam/length(train_examples);
p_ham=1-p_spam;

% Predict for Test Examples

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

% Posterior
test_spam_prob=cnt_spam_test.*p_spam;
test_ham_prob=cnt_ham_test.*p_ham;

% Compute argmax with spam being the posituve class
pred=test_spam_prob>=test_ham_prob;

% Confusion Matrix
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

% Metric Calculation

precision=tp/(tp+fp);
recall=tp/(tp+fn);
f_score=(2*precision*recall)/(precision+recall);
accuracy=(tp+tn)/length(test_examples);
