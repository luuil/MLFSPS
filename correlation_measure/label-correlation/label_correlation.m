function LC = label_correlation(Data, L)
%Label Correlation
% inputs:
%  Data: the dataset
%  L:    the index vector of the labelset in dataset 'Data'
% outputs:
%  LC:   the correlation vector

ln = length(L);
LC = zeros(1, ln);

for i = 1 : ln
    l_i   = Data(:, L(i)); %label L(i)
    for j = 1 : ln
        if j ~= i
            l_j   = Data(:, L(j)); %label L(j)
            LC(i) = LC(i) + mutualinfo(l_i, l_j);
        end
    end
    LC(i) = LC(i) / (ln - 1);
end
end