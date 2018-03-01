function [unionMap] = unionWords(dataMapPos, dataMapNeg, num)
    keysPos = keys(dataMapPos);
    keysNeg = keys(dataMapNeg);
    % find the new data only appears in tempKeys
    % set the value to 1        
    % where tempKeys is the current cell array 
    unionKeys = union(keysPos, keysNeg); 
    
    unionMap = debuteNdup(dataMapPos, dataMapNeg);
    
    unionMap = rmInfreqNfreq(unionMap, num);
    
end

function unionMap = debuteNdup(dataMapPos, dataMapNeg)

    keysPos = keys(dataMapPos);
    keysNeg = keys(dataMapNeg);

    % find the data only appears in dataMapNeg
    % merge into dataMapPos
    % where tempKeys is the current cell array 
    debuteKeys = setdiff(keysNeg, keysPos); 
    if(~isempty(debuteKeys)) % if there is debuteKeys
        % add debuteKeys to the dataMapPos
        tempVal = values(dataMapNeg, debuteKeys);
        tempMap = containers.Map(debuteKeys, tempVal);
        dataMapPos = [dataMapPos; tempMap]; % merge
    end
    
    % find the existing data appears in both dataSets
    % take out the values and add them up        
    % where tempKeys is the current cell array 
    dupKeys = intersect(keysNeg, keysPos);
    if(~isempty(dupKeys)) % if there is dupKeys
        % find two values in each side        
        dupVal = values(dataMapPos,dupKeys); % cell
        dupValArrPos = cell2mat(dupVal); % cell to array 
        dupVal2 = values(dataMapNeg,dupKeys); % cell
        dupValArrNeg = cell2mat(dupVal2); % cell to array

        % add up two values
        dupValArrPos = dupValArrPos + dupValArrNeg; 
        sizeNewData = ones(1,length(dupValArrPos));
        valNew = mat2cell(dupValArrPos, [1], sizeNewData); % array back to cell
        tempMap = containers.Map(dupKeys, valNew);
        dataMapPos = [dataMapPos; tempMap]; % merge
    end
    unionMap = dataMapPos;
    
end


function [dataMapNeg] = rmInfreqNfreq(dataMapNeg, num)
    posVal = values(dataMapNeg);
    pvalArray = cell2mat(posVal);
    pinfreqKeys = keys(dataMapNeg);
    pinfreqKeys = pinfreqKeys(pvalArray < num);
    remove(dataMapNeg, pinfreqKeys);
    remove(dataMapNeg, {''});remove(dataMapNeg, 'a');
end