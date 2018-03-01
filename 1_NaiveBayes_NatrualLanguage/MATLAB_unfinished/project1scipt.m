% Comp 135: Intro to ML
% Project 1 % 02/28
% Jiacheng Qu 1234031
% 
% 
% We have three datasets collected from three domains: 
% imdb.com, amazon.com, yelp.com. 
% Each dataset consists of sentences with sentiment labels 
% (1 for positive and 0 for negative) extracted from . 
% These form 3 datasets for the assignment.
% 
% Each dataset is given in a single text file, with each line as an instance. 
% Each line is a list of space separated words, which essentially a sentence, 
% followed by a tab character, and then followed by the label. 
% Here is a snippet from the yelp dataset:
%       Crust is not good. 0 
%       Best burger ever had! 1
% 
% You are suggested to remove 
%       all the punctuation, 
%       numeric values, N
%       convert upper case to lower case 
% for each example so that the same word will treated in the same way in the data.

% load the data, skip the numbers and punctuation
% remove some words in the data.
% histogram 

clear all
cd sentiment_labelled_sentences
files=dir(fullfile(pwd,'*labelled.txt')); % read properly labelled file
for i=1:length(files)
   fid(i) = fopen(files(i).name);
end
cd ..

% Training and testing
keySet = {'a'};     valueSet = [0]; % initiate the Hash-Map
trainPosWord = containers.Map(keySet,valueSet);
trainNegWord = containers.Map(keySet,valueSet);
infqNum = 1; % if count < infqNum then remove from the dict
rmTense = 'y';
[trainPosWord, trainNegWord] = trainDict( fid(1), trainPosWord, trainNegWord, infqNum, rmTense);
[trainPosWord, trainNegWord] = trainDict( fid(2), trainPosWord, trainNegWord, infqNum, rmTense);

infqNum = 2; % if count < infqNum then remove from the dict
[unionMap] = unionWords(trainPosWord, trainNegWord, infqNum);
[trainPosWord, trainNegWord] = trainedWords(unionMap, trainPosWord, trainNegWord);
%% 

%  trainResults = mergePosNegWord(trainPosWord, trainNegWord);
[trainPosWord, trainNegWord] = testDict( fid(3), trainPosWord, trainNegWord, infqNum);



% [trainPosWord, trainNegWord] = trainDict( fid(3), trainPosWord, trainNegWord);

for i=1:length(files)
   fclose(fid(i));
end
% clean up the data space
clear i keySet valueSet ans fid filename files;
































