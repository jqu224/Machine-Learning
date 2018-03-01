function [trainedPosWord, trainedNegWord] = trainedWords(unionMap, dataMapPos, dataMapNeg)
% for trainPosWord and trainNegWord, delete word that does show up in unionMap 
    keyUnion = keys(unionMap);
    keysPos = keys(dataMapPos);
    keysNeg = keys(dataMapNeg);

    % find the new data only appears in keysNeg/tempKeys
    % remove from the dataMapPos/dataMapNeg
    % where keysNeg/tempKeys is the current cell array 
    clear debuteKeys;
    debuteKeys = setdiff(keysPos, keyUnion); 
    if(~isempty(debuteKeys)) % if there is debuteKeys in keysNeg
        remove(dataMapPos, debuteKeys);
    end
    clear debuteKeys;
    debuteKeys = setdiff(keysNeg, keyUnion); 
    if(~isempty(debuteKeys)) % if there is debuteKeys in keysNeg
        remove(dataMapNeg, debuteKeys);
    end
    trainedPosWord = dataMapPos;
    trainedNegWord = dataMapNeg;

end