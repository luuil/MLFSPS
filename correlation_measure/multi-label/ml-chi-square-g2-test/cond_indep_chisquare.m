function [CI, dep, alpha2] = cond_indep_chisquare(Data, X, Y, S, test, alpha, ns)
% COND_INDEP_CHISQUARE Test:
%  if X indep Y given Z using either chisquare test or likelihood ratio test G2

% the feature values of X must be from 1 to dom(X), for example, if feature X have
% two values, in data, the two values are denoted as [1,2].

%
% [CI Chi2 Prob_Chi2] = cond_indep_chisquare(X, Y, S, Data, test, alpha, node_sizes)
%
% Input :
%       Data:  the data matrix, NbVar columns * N rows
%       X:     the index of variable X in Data matrix
%       Y:     the index of variable Y in Data matrix
%       S:     the indexes of variables in set S
%       alpha: the significance level (default: 0.05)
%       test:  'chi2' for Pearson's chi2 test
%		       'g2'   for G2 likelihood ratio test (default)
%       ns:    node_sizes (default: max(Data'))
%
% Output :
%       CI: test result (1 = conditional independency, -1 = not conditional independency)
%       Chi2: chi2 value (-1 if not enough data to perform the test --> CI=1)
%
%
% V1.4 : 24 july 2003 (Ph. Leray - philippe.leray@univ-nantes.fr)
%
%
% Things to do :
% - do not use 'find' in nij computation (when S = empty set)
% - find a better way than 'warning off/on' in tmpij, tmpijk computation
%

if nargin < 5, test = 'g2';    end
if nargin < 6, alpha = 0.05;   end
if nargin < 7, ns = max(Data); end

%N     = size(Data,2);
N      = size(Data,1);
qi     = ns(S);
tmp    = [1 cumprod(qi(1 : end-1))];
qs     = 1 + (qi - 1) * tmp';

dep    = -1.0;
alpha2 = 1;

if isempty(qs),
    nij = zeros(ns(X), ns(Y));
    df  = prod(ns([X Y]) - 1) * prod(ns(S));
else
    
    %Commented by Mingyi
    % nijk=zeros(ns(X),ns(Y),qs);
    % tijk=zeros(ns(X),ns(Y),qs);
    %Commention ends
    
    % Added by Mingyi
    nijk = zeros(ns(X), ns(Y), 1);
    tijk = zeros(ns(X), ns(Y), 1);
    % Addition ends
    
    df = prod(ns([X Y]) - 1) * qs;
end


% Added by YuKui
if(df <= 0)
    df = 1;
end
% Addition ends

%if (N<10*df)
if (N < 5 * df)
    
    % Not enough data to perform the test
    Chi2 = -1;
    CI   = 1;
    fprintf('Not enough data to perform the test: \t\t\t: INDPCY :\tCHI2=%8.2f \t\n',Chi2);
    
elseif isempty(S)
    
    for i = 1 : ns(X)
        for j = 1 : ns(Y)
            nij(i, j) = length(find(((Data(:, X)) == i) & (Data(:,Y) == j)));
        end
    end
    
    %restr=find(sum(nij,1)==0);        %Commented by Liu
    restr = find(sum(nij, 1) == 0, 1); %Add by Liu
    if ~isempty(restr)
        nij = nij(:, find(sum(nij,1)));
    end
    
    tij = sum(nij,2) * sum(nij,1) / N ; % a number
    
    switch test
        case 'chi2'
            tmpij = nij - tij;
            [xi, yj] = find(tij < 10);
            
            for i = 1 : length(xi)
                tmpij(xi(i), yj(i))= abs(tmpij(xi(i), yj(i))) - 0.5;
            end
            
            warning off;
            tmp = (tmpij.^2) ./ tij;
            warning on;
            
            tmp(tmp == Inf) = 0;
            
        case 'g2'
            warning off;
            tmp = nij ./ tij;
            warning on;
            
            tmp(tmp == Inf | tmp == 0) = 1;
            tmp(tmp ~= tmp)            = 1;
            tmp                        = 2 * nij .* log(tmp);
            
        otherwise
            error(['unrecognized test ' test]);
    end
    
    Chi2   = sum(sum(tmp));
    alpha2 = 1 - chisquared_prob(Chi2, df);
    %CI    = (alpha2 >= alpha); %Commented by Liu
    
    %Add by Liu, for weighting asignment
    if alpha2 >= alpha
        CI = 1;
    else
        CI = -1;
    end
    %Addition ends
    
    %Added by YuKui
    statistic = Chi2;
    if(alpha2 >= alpha)
        dep = (-2.0) - statistic / df;
    else
        dep = 2.0 + statistic / df;
    end
    %Addition ends
    
else
    SizeofSSi = 1;
    for exemple = 1 : N,
        
        i  = Data(exemple, X);
        j  = Data(exemple, Y);
        Si = Data(exemple, S) - 1;
        
        %Added by Mingyi
        if exemple == 1
            SSi(SizeofSSi, :)     = Si;
            nijk(i, j, SizeofSSi) = 1;
        else
            flag = 0;
            for iii = 1 : SizeofSSi
                if isequal(SSi(iii,:), Si)
                    nijk(i, j, iii) = nijk(i, j, iii) + 1;
                    flag            = 1;
                end
            end
            if flag == 0
                SizeofSSi             = SizeofSSi + 1;
                SSi(SizeofSSi, :)     = Si;
                nijk(i, j, SizeofSSi) = 1;
            end
        end
        %Addition ends
        
        %Commented by Mingyi
        %         k=1+Si*tmp';
        %         nijk(i,j,k)=nijk(i,j,k)+1;
        %Commention ends
    end
    
    nik = sum(nijk, 2);
    njk = sum(nijk, 1);
    N2  = sum(njk);
    
    %for k=1:qs,          %Commented by Mingyi
    for k = 1 : SizeofSSi %Added by Mingyi
        if N2(:, :, k) == 0
            tijk(:, :, k) = 0;
        else
            tijk(:, :, k) = nik(:, :, k) * njk(:, :, k) / N2(:, :, k);
        end
    end
    
    switch test
        case 'chi2',
            tmpijk   = nijk - tijk;
            
            [xi, yj] = find(tijk < 10);
            for i=1:length(xi),
                tmpijk(xi(i), yj(i)) = abs(tmpijk(xi(i), yj(i))) - 0.5;
            end
            
            warning off;
            tmp = (tmpijk.^2) ./ tijk;
            warning on;
            
            tmp(tmp == Inf)=0;
            
        case 'g2',
            warning off;
            tmp = nijk ./ tijk;
            warning on;
            
            tmp(tmp == Inf | tmp == 0) = 1;
            tmp(tmp ~= tmp)            = 1;
            tmp                        = 2 * nijk .* log(tmp);
            
        otherwise,
            error(['unrecognized test ' test]);
    end
    
    Chi2   = sum(sum(sum(tmp)));
    alpha2 = 1 - ml_chisquared_prob(Chi2, df);
    %CI    = (alpha2 >= alpha); %Commented by Liu
    
    %Add by Liu, for weighting assignment
    if alpha2 >= alpha
        CI = 1;
    else
        CI = -1;
    end
    %Addition ends
    
    %Added by YuKui
    statistic = Chi2;
    if(alpha2 >= alpha)
        dep = (-2.0) - statistic / df;
    else
        dep = 2.0 + statistic / df;
    end
    %Addition ends
    
end
%fprintf('\t\t\t: INDPCY :\tCHI2=%8.2f \t\n',Chi2);

clear tijk
clear nijk
clear nij
clear tij
clear tmpijk