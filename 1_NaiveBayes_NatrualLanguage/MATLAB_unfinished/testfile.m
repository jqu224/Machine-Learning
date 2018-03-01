a1 = {'a','b','d','c'};
valueSet = [0,2,10,20];
dataMapPos = containers.Map(a1,valueSet);

b = {'z','w','c','d','the'};
% valueSet = [0,2,33,2,13];
% dataMapNeg = containers.Map(b,valueSet);
tic
scs = isKey(dataMapPos, b);
toc

tic
for n = 1:5
    scs = isKey(dataMapPos, b{n});
end
toc
% 
% if(isKey(dataMapPos, b))
%     
% end
%% 
% 
% tic
% dataMapPos = debuteNdup(dataMapPos,b);
% toc
% 
% function dataMapPos = debuteNdup(dataMapPos, tempKeys)
%     keysPos = keys(dataMapPos);
%     % find the new data only appears in tempKeys
%     % set the value to 1        
%     % where tempKeys is the current cell array 
%     debuteKeys = setdiff(tempKeys, keysPos); 
%     if(~isempty(debuteKeys)) % if there is debuteKeys
%         % add debuteKeys to the dataMapPos
%         val = ones(1, length(debuteKeys));
%         tempMap = containers.Map(debuteKeys, val);
%         dataMapPos = [dataMapPos; tempMap]; % merge
%     end
% 
%     % find the existing data appears in both dataSets
%     % take out the values and +1        
%     % where tempKeys is the current cell array 
%     dupKeys = intersect(tempKeys, keysPos);
%     if(~isempty(dupKeys)) % if there is dupKeys
%         % add 1 to the existing word in the dataMapPos
%         dupVal = values(dataMapPos,dupKeys); % cell
%         dupValarr = cell2mat(dupVal)+1; % cell to array and +1
%         sizeNewData = ones(1,length(dupValarr));
%         valNew = mat2cell(dupValarr, [1], sizeNewData); % array to cell
%         tempMap = containers.Map(dupKeys, valNew);
%         dataMapPos = [dataMapPos; tempMap]; % merge
%     end
% end
%% 
keysPos = keys(dataMapPos);
keysNeg = keys(dataMapNeg);

% find the new data only appears in tempKeys
% set the value to 1        
% where tempKeys is the current cell array 
debuteKeys = setdiff(keysNeg, keysPos); 
if(~isempty(debuteKeys)) % if there is debuteKeys
    % add debuteKeys to the dataMapPos
    val = ones(1, length(debuteKeys));
    tempVal = values(dataMapNeg, debuteKeys);
    tempMap = containers.Map(debuteKeys, tempVal);
    dataMapPos = [dataMapPos; tempMap]; % merge
end

% 
% tempVal = keys(dataMapPos)
% tempVal = values(dataMapPos)

% find the existing data appears in both dataSets
% take out the values and +1        
% where tempKeys is the current cell array 
dupKeys = intersect(keysNeg, keysPos);
if(~isempty(dupKeys)) % if there is dupKeys
    % add 1 to the existing word in the dataMapPos
    dupVal = values(dataMapPos,dupKeys); % cell
    dupValArr1 = cell2mat(dupVal); % cell to array 
    dupVal2 = values(dataMapNeg,dupKeys); % cell
    dupValArr2 = cell2mat(dupVal2); % cell to array

    % add up two values
    dupValArr1 = dupValArr1 + dupValArr2; 
    sizeNewData = ones(1,length(dupValArr1));
    valNew = mat2cell(dupValArr1, [1], sizeNewData); % array to cell
    tempMap = containers.Map(dupKeys, valNew);
    dataMapPos = [dataMapPos; tempMap]; % merge
end
% 
% tempVal = keys(dataMapPos)
% tempVal = values(dataMapPos)