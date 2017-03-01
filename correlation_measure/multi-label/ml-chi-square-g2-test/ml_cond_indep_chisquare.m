function [CI, dep, p_value] = ml_cond_indep_chisquare(data, f, L, S, weight, test, alpha, node_sizes)
% COND_INDEP_CHISQUARE Test:
%  if X indep Y given Z using either chisquare test or likelihood ratio test G2

% the feature values of X must be from 1 to dom(X), for example, if feature X have
% two values, in data, the two values are denoted as [1,2].

%
% Inputs:
%       data : a N * M matrix, with all features including the class attribute
%       f    : the index of feature f in Data matrix
%       L    : the index set of label set L in Data matrix
%       S    : the indexes of variables in set S
%       weight: weight vector (default: 'SCA' strategy)
%       alpha: the significance level (default: 0.05)
%       test : 'chi2' for Pearson's chi2 test
%		       'g2'   for G2 likelihood ratio test (default)
%       node_sizes: node sizes (default: max(Data'))
%
% Outputs:
%       CI     : test result (1=conditional independency, 0 = not conditional independency)
%       dep    : flag
%       p_value: p value
%

if nargin < 6, test = 'g2';            end
if nargin < 7, alpha = 0.05;           end
if nargin < 8, node_sizes = max(data); end

% initialization
ln          = length(L);
vec_CI      = zeros(1, ln);
vec_dep     = zeros(1, ln);
vec_p_value = zeros(1, ln);

for i = 1 : ln
    [vec_CI(i), vec_dep(i), vec_p_value(i)] = my_cond_indep_chisquare(data, f, L(i), S, test, alpha, node_sizes);
end

% Weighting each label
vec_CI      = vec_CI      .* weight;
vec_dep     = vec_dep     .* weight;
vec_p_value = vec_p_value .* weight;

% CI
CI = ( sum(vec_CI)>=0.5 );

% dep
if 2 * length( vec_dep( isnan(vec_dep) ) ) >= length(vec_dep)
    dep = NaN;
else
    dep = mode(vec_dep); % Most frequent value in sample
end

% p_value
p_value = mode(vec_p_value);

end