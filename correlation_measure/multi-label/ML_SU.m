function [score] = ML_SU(data,f,L,W)
% data: full data set
% f:    current feature index
% L:    label set indexes
% W:    label weight vector
firstVector = data(:,f);
score = 0;
for i=1:length(L)
    secondVector = data(:,L(i));
    score = score + W(i) * SU(firstVector,secondVector);
end