function [guessedLabel] = testDict(fileID, dataMapPos, dataMapNeg)
    % Comp 135: Intro to ML
    % project 1
    % Jiacheng Qu % 02/28`
    % input the data
    i = 1; % start from the first sentence
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


        % find each word of current sentence to the hash-map
        
        % positive review % label(i) = 1;
           [probPos] = probGivenFeature(dataMapPos, tempKeys);
        % negative review label(i) = 0;
           [probNeg] = probGivenFeature(dataMapNeg, tempKeys);
        i = i+1;
    end

end


function dataMapPos = probGivenFeature(dataMapPos, tempKeys)
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
end










