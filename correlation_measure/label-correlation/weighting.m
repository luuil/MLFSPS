function W = weighting(data, L, type)
%Weight for each label in the label set L
%
% inputs:
%  Data: the dataset
%  L:    a set of index of the label set in dataset Data
%  type: weighting strategies
%         'NCA': No Correlation Asignment
%         'LCA': Large Correlation Asignment
%         'SCA': Small Correlation Asignment (default)
% outputs:
%  W:    the weight vector

if nargin < 3
    type = 'SCA';
end

ln = length(L);
lc = [];

switch type
    case 'NCA'
        lc = ones(1, ln);
    case 'LCA'
        lc = label_correlation(data, L);
    case 'SCA'
        lc = label_correlation(data, L);
        lc = 1 - lc; %invert, i.e. small asignment
    otherwise
        fprintf('%s', 'No such weighting strategy.');
end

W = [];
if ~isempty(lc)
    W  = zeros(1, ln); %weight vector
    for i = 1 : ln
        W(i) = lc(i) / sum(lc);
    end
end
end