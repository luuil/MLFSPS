function [CI, dep1]= ml_optimal_compter_dep_2(bcf, f, L, max_k, discrete, weight, alpha, test, data)
%for a discrete data set, discrete=1, otherwise, discrete=0 for a continue data set

% Add by Liu
% Inputs:
%       bcf     : the indexes of the best candidate features in Data matrix
%       f       : the index of feature f in Data matrix
%       L       : the indexes of label set L in Data matrix
%       max_k   : the most number of features to be select
%       discrete: 1,      for discrete data set;
%                 0,      for continue data set;
%       weight: weight vector (default: 'SCA' strategy)
%       alpha   : the significance level (default: 0.05)
%       test    : 'chi2' for Pearson's chi2 test
%		          'g2'   for G2 likelihood ratio test (default)
%       data    : a N * M matrix, with all features including the class attribute
%
%  Adopted for Online Multi-Label Feature Selection(OMLFS) by Liu(humooo@outlook.com)
%  2016-04-29
% Addition ends

%if new feature X is not redundant, that is, we cannot
%remove X from X, then we check redundency for each feature.
%When we test redundency for a feature, we only consider
%its candidate Markov blanlets containg the new feautre X to redunce the
%number of tests. for example, now BCF=[2,3,4,5], If feature 6 is added
%into BCF, BCF=[2,3,4,5,6]. When testing feature 5, we only consider the
%following subsets: [6],[2,6],[3,6],[4,6],[2,3,6],[2,4,6],[3,4,6],if
%max_k=3.

%dep1         = 0; %Commented by Liu
x             = 0;
n_pc          = length(bcf);
code          = bcf;
N             = size(data,1);
max_cond_size = max_k;
CI            = 0;
%p            = 1; %Commented by Liu

if max_cond_size > n_pc
    max_cond_size = n_pc;
end

%cond     =[]; %Commented by Liu
cond_size = 1;

while cond_size <= max_cond_size
    
    cond_index = zeros(1,cond_size);
    for i=1:cond_size
        cond_index(i)=i;
    end
    
    stop = 0;
    while stop == 0
        
        %Commented by Liu
        %cond = [];
        %for i = 1 : cond_size
        %  if i == cond_size
        %     cond_index(i)=n_pc;
        %     cond = [cond code(cond_index(i))];
        %  else
        %     cond = [cond code(cond_index(i))];
        %  end
        %end
        
        %Add by Liu
        cond = zeros(1, cond_size);
        for i = 1 : cond_size
            if i == cond_size
                cond_index(i) = n_pc;
            end
            cond(i) = code(cond_index(i));
        end
        %Addition ends
        
        if discrete == 1
            %[CI, dep, alpha2] = my_cond_indep_chisquare(data, var, target, cond, test, alpha); %Commented by Liu
            [CI, dep] = ml_cond_indep_chisquare(data, f, L, cond, weight, test, alpha);          %Add by Liu
            x = dep;
        else
            %[CI, r, p]= my_cond_indep_fisher_z(data, var, target, cond, N, alpha); %Commented by Liu
            [CI, r]= ml_cond_indep_fisher_z(data, f, L, cond, N, alpha);     %Add by Liu
            x = r;
        end
        
        if(CI == 1 || isnan(x))
            stop      = 1;
            cond_size = max_cond_size + 1;
        end
        
        if(stop == 0)
            [cond_index, stop] = optimal_next_cond_index(n_pc, cond_size, cond_index);
        end
    end
    cond_size = cond_size + 1;
end
dep1 = x;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Change:
%       'cond_index'  to 'next_index'; 
%       'cond_index1' to 'cur_index';
%by Liu
function [next_index, stop] = optimal_next_cond_index(n_pc, cond_size, cur_index)

stop = 1;
i    = cond_size;

while i >= 1
    if(cur_index(i) < n_pc + i - cond_size)
        if i == cond_size
            cur_index(i) = n_pc + i-cond_size;
        else
            cur_index(i) = cur_index(i) + 1;
        end
        
        if i < cond_size
            for j = i+1 : cond_size
                if j == cond_size
                    cur_index(j) = n_pc;
                else
                    cur_index(j) = cur_index(j-1) + 1;
                end
            end
        end
        stop = 0;
        i    = -1;
    end
    i = i-1;
end
next_index = cur_index;
end
