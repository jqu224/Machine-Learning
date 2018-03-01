function [trainPosWord, trainNegWord] = trainDict(fileID, dataMapPos, dataMapNeg)
% Comp 135: Intro to ML
% project 1
% Jiacheng Qu % 02/28
% fileID = fopen(filename);
% label = zeros(1,1000);
% keySet = {'a'};     valueSet = [0]; % initiate the Hash-Map
% dataMapPos = containers.Map(keySet,valueSet);
% dataMapNeg = containers.Map(keySet,valueSet);
% input the data
i = 1; % start from the first sentence
a = 1; % indicator for positive review
b = 1; % indicator for negative review
while ~feof(fileID)
    % read by lines
    clear line newStr tempKeys keysPos valNew keysNeg debuteKeys tempMap dupKeys sizeNewData dupVal dupValarr sizeNewData;
    line = fgetl(fileID);
    
    % Erase Punctuation and numebr from Text
    newStr = replace(line,"/"," "); newStr = replace(newStr,"'","");
    newStr = regexprep(newStr, '[-_=@-$&{}()[]:_=+|*\.,''!?]',' '); 
    % erasePunctuation(newStr); % works for MATLAB 2017b 
    % split string into cell array and convert to lower case
    tempKeys = lower(strsplit(newStr)); 
    sizeOfTemp = size(tempKeys);
    sizeTemp =sizeOfTemp(2) - 1;  % length of the array
    
    
    % store each word of current sentence to the hash-map
%     tf = isKey(dataMap,keySet);
    if line(end) == '1' % positive review % label(i) = 1;
        keysPos = keys(dataMapPos);
        % find the new data only appears in tempKeys
        % set the value to 1        
        % where tempKeys is the current cell array 
        debuteKeys = setdiff(tempKeys, keysPos); 
        if(~isempty(debuteKeys)) % if there is debuteKeys
            % add debuteKeys to the dataMapPos
            val = ones(1, length(debuteKeys));
            tempMap = containers.Map(debuteKeys, val);
            dataMapPos = [dataMapPos; tempMap]; % merge
        end
        
        % find the existing data appears in both dataSets
        % take out the values and +1        
        % where tempKeys is the current cell array 
        dupKeys = intersect(tempKeys, keysPos);
        if(~isempty(dupKeys)) % if there is dupKeys
            % add 1 to the existing word in the dataMapPos
            dupVal = values(dataMapPos,dupKeys); % cell
            dupValarr = cell2mat(dupVal)+1; % cell to array and +1
            sizeNewData = ones(1,length(dupValarr));
            valNew = mat2cell(dupValarr, [1], sizeNewData); % array to cell
            tempMap = containers.Map(dupKeys, valNew);
            dataMapPos = [dataMapPos; tempMap]; % merge
        end
        
        a = a+1;
    elseif line(end) == '0' % negative review label(i) = 0;
        keysNeg = keys(dataMapNeg);
        % find the new data only appears in tempKeys
        % set the value to 1        
        % where tempKeys is the current cell array 
        debuteKeys = setdiff(tempKeys, keysNeg); 
        if(~isempty(debuteKeys)) % if there is debuteKeys
            % add debuteKeys to the dataMapNeg
            val = ones(1, length(debuteKeys));
            tempMap = containers.Map(debuteKeys, val);
            dataMapNeg = [dataMapNeg; tempMap]; % merge
        end
        
        % find the existing data appears in both dataSets
        % take out the values and +1        
        % where tempKeys is the current cell array 
        dupKeys = intersect(tempKeys, keysNeg);
        if(~isempty(dupKeys)) % if there is dupKeys
            % add 1 to the existing word in the dataMapNeg
            dupVal = values(dataMapNeg,dupKeys); % cell
            dupValarr = cell2mat(dupVal)+1; % cell to array and +1
            sizeNewData = ones(1,length(dupValarr));
            valNew = mat2cell(dupValarr, [1], sizeNewData); % array to cell
            tempMap = containers.Map(dupKeys, valNew);
            dataMapNeg = [dataMapNeg; tempMap]; % merge
        end
%         C{b,2} = strsplit(newStr);
        b = b+1;
    end
    
    i = i+1;
end
%% remove infreq and freq words
posVal = values(dataMapPos);
pvalArray = cell2mat(posVal);
pinfreqvalIndex = find(pvalArray<3);
pinfreqKeys = keys(dataMapPos);
pinfreqKeys = pinfreqKeys(pinfreqvalIndex);
remove(dataMapPos, pinfreqKeys);
remove(dataMapPos, {''});remove(dataMapPos, 'a');

negVal = values(dataMapNeg);
nvalArray = cell2mat(negVal);
ninfreqvalIndex = find(nvalArray<=3);
ninfreqKeys = keys(dataMapNeg);
ninfreqKeys = ninfreqKeys(ninfreqvalIndex);
remove(dataMapNeg, ninfreqKeys);
remove(dataMapNeg, ''); remove(dataMapNeg, 'a');

trainPosWord = dataMapPos;
trainNegWord = dataMapNeg;

% %% plot the histogram
% figure(1)
% bar( cell2mat( values(dataMapPos) ) )
% set(gca,'XTick',[1:length(keys(dataMapPos))])
% set(gca,'xticklabel', keys(dataMapPos))
% 
% figure(2)
% bar( cell2mat( values(dataMapNeg) ) )
% set(gca,'XTick',[1:length(keys(dataMapNeg))])
% set(gca,'xticklabel', keys(dataMapNeg))

end














