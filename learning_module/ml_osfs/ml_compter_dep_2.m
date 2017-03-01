function [CI, dep1, p_value] = ml_compter_dep_2(bcf, f, L, max_k, discrete, weight, alpha, test, data)
% function [CI,dep1,p_value] = compter_dep_2(bcf, var, target, max_k, discrete, alpha, test, data) % Commented by Liu

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
%   Adopted for Online Multi-Label Feature Selection(OMLFS) by Liu, humooo@outlook.com
%   2016-04-29
% Addition ends

% Original contents(Commented by Liu)
%   test     = 'chi2', for Pearson's chi2 test;
%              'g2',   for G2 likelihood ratio test (default)
% Contents ends

%dep1=0;
x             = 0;
n_bcf         = length(bcf);
code          = bcf;
N             = size(data, 1);
max_cond_size = max_k;
CI            = 0;
p_value       = 1;

if(max_cond_size > n_bcf)
    max_cond_size = n_bcf;
end

%cond     = []; % Commented by Liu
cond_size = 1;

while cond_size <= max_cond_size
    
    cond_index = zeros(1, cond_size);
    
    for i = 1 : cond_size
        cond_index(i) = i;
    end
    
    stop=0;
    
    while stop == 0
        
        %Commented by Liu
        % cond = [];
        % for i = 1 : cond_size
        %     cond = [cond code(cond_index(i))];
        % end
        
        % Add by Liu, pre-allocating for speed
        cond = zeros(1, cond_size);
        for i = 1 : cond_size
            cond(i) = code(cond_index(i));
        end
        % Addition ends
        
        if discrete == 1
            ns = max(data);
            
            %[CI, dep, p_value] = my_cond_indep_chisquare(data, var, target, cond, test, alpha, ns); %Commented by Liu
            %x = dep;
            [CI, x, p_value] = ml_cond_indep_chisquare(data, f, L, cond, weight, test, alpha, ns);    %Add by Liu
            
        else
            %[CI, r, p_value] = my_cond_indep_fisher_z(data, var, target, cond, N, alpha); %Commented by Liu
            %x = r;
            [CI, x, p_value] = ml_cond_indep_fisher_z(data, f, L, cond, N, alpha);  %Add by Liu
        end
        
        if(CI==1 || isnan(x))
            stop = 1;
            cond_size = max_cond_size + 1;
        end
        
        if(stop == 0)
            [cond_index, stop] = next_cond_index(n_bcf, cond_size, cond_index);
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
function [next_index, stop] = next_cond_index(n_bcf, cond_size, cur_index)

stop = 1;
i    = cond_size;

while i >= 1
    if (cur_index(i) < n_bcf + i - cond_size)
        cur_index(i) = cur_index(i) + 1;
        if i < cond_size
            for j = i + 1 : cond_size
                cur_index(j) = cur_index(j-1) + 1;
            end
        end
        stop = 0;
        i = -1;
    end
    i = i - 1;
end
next_index = cur_index;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
