function [trainPosWord, trainNegWord] = trainDict(fileID, dataMapPos, dataMapNeg, num, Tense)
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
    while ~feof(fileID)
        % read by lines
        clear line newStr tempKeys keysPos valNew keysNeg debuteKeys tempMap dupKeys sizeNewData dupVal dupValarr sizeNewData;
        line = fgetl(fileID);

        % Erase Punctuation and numebr from Text
        newStr = remove_Punctuation_Tense(line, Tense);
        
        % erasePunctuation(newStr); % works for MATLAB 2017b 
        % split string into cell array and convert to lower case
        tempKeys = strsplit(newStr); 
        sizeOfTemp = size(tempKeys);
        sizeTemp =sizeOfTemp(2) - 1;  % length of the array

        % store each word of current sentence to the hash-map
        if line(end) == '1' % positive review % label(i) = 1;
           [dataMapPos] = debuteNdup(dataMapPos, tempKeys);
        elseif line(end) == '0' % negative review label(i) = 0;
           [dataMapNeg] = debuteNdup(dataMapNeg, tempKeys);
        end
        i = i+1;
    end


    % remove infreq and freq words
    if num > 1 % skip if numInfreq !> 1; there is nothing to remove
        [dataMapPos] = rmInfreqNfreq(dataMapPos, num);
        [dataMapNeg] = rmInfreqNfreq(dataMapNeg, num);
    end
    trainPosWord = dataMapPos;
    trainNegWord = dataMapNeg;

    %% plot the histogram
    figure(1)
    bar( cell2mat( values(dataMapPos) ) )
    set(gca,'XTick',[1:length(keys(dataMapPos))])
    set(gca,'xticklabel', keys(dataMapPos))
    figure(2)
    bar( cell2mat( values(dataMapNeg) ) )
    set(gca,'XTick',[1:length(keys(dataMapNeg))])
    set(gca,'xticklabel', keys(dataMapNeg))
end


function dataMapPos = debuteNdup(dataMapPos, tempKeys)
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

function [dataMapNeg] = rmInfreqNfreq(dataMapNeg, num)
    posVal = values(dataMapNeg);
    pvalArray = cell2mat(posVal);
    pinfreqKeys = keys(dataMapNeg);
    pinfreqKeys = pinfreqKeys(pvalArray < num);
    remove(dataMapNeg, pinfreqKeys);
    remove(dataMapNeg, {''});remove(dataMapNeg, 'a');
end



function  [newStr] = remove_Punctuation_Tense(line, Tense)
    newStr = replace(line,"/"," "); newStr = replace(newStr,"'","");
    newStr = regexprep(newStr, '[-_=@-$&{}()[]:_=+|*\.,''!?]',' '); 
    newStr = lower(newStr);
    if Tense == 'y'
        newStr = regexprep(newStr, '(\w+)ied','y');
        newStr = regexprep(newStr, '(\w+)ies','y'); 
        newStr = regexprep(newStr, '(\w+)ly',''); 
        newStr = regexprep(newStr, '(\w+)ed','');
        newStr = regexprep(newStr, '(\w+)ing','');
        newStr = regexprep(newStr, '(\w+)e','');
        newStr = regexprep(newStr, '(\w+)es','');
    end
end




