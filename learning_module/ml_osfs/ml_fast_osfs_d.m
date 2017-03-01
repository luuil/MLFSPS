function [selected_features, time] = ml_fast_osfs_d(data, L, alpha, test, weight)
% function [selected_features,time] = fast_osfs_d(data1, class_index, alpha, test) % Commented by Liu

% for disccrete data

% Add by Liu
%  Inputs:
%   data : a N * M matrix, with all features including the class attribute
%   L    : the indexes of the label set
%   alpha: significant level( 0.01 or 0.05 )
%   test : 'chi2', for Pearson's chi2 test;
%          'g2'  , for G2 likelihood ratio test;
%          (for discrete data)
%
%  Adopted by Liu(humooo@outlook.com)
%  2016-04-29
% Addition ends


% Original contents(Commented by Liu)
%  important note:
%   for discrete dataset: the feature values of X must be from 1 to max_value(X), for example,feature X has to take consecutive integer values starting from 1,
%   that is, 1..max_value(X). For example, if max_value(X)=3, this means that feature X takes values {1,2,3}.

% input parameter:
%   data1:  data with all features including the class attribute
%   target: the index of the class attribute ( we assume the class attribute is the last colomn of a data set)
%   alpha:  significant level( 0.01 or 0.05 )

%  for discrete data: 
%   test = 'chi2', for Pearson's chi2 test;
%          'g2',   for G2 likelihood ratio test
%  for example: 
%   The UCI dataset wdbc with 569 instances and 31 features (the index of the  class attribute is 31).
%     [selected_features1,time]=fast_osfs_d(wdbc,31,0.01,'g2')
%   if the feature values of X must be from 0 to max_value(X)-1,then
%     [selected_features1,time]=fast_osfs_d(wdbc+1,31,0.01,'g2')

% output:
%   selected_features1: the selected features
%   time: running time

%  please refer to the following papers for the details and cite them:
%  Wu, Xindong, Kui Yu, Wei Ding, Hao Wang, and Xingquan Zhu. "Online feature selection with streaming features." Pattern Analysis and Machine Intelligence, IEEE Transactions on 35, no. 5 (2013): 1178-1192.
% Contents ends



start              = tic;
col                = size(data, 2);
ns                 = max(data);
selected_features  = [];
selected_features1 = [];
%b                 = []; %Commented by Liu

time_start         = cputime;
w                  = weighting(data, L, weight); %weight vector
fprintf('Label weight calculation costs: %d s\n', cputime - time_start);


for i = 1 : col-length(L) %the last length(L) features is the class attribute, i.e., the target)
    
    
    % for very sparse data
    n1 = sum(data(:, i));
    if n1 == 0
        continue;
    end
    
    
    stop=0;
    
    %CI=1; % Commented by Liu
    %[CI] = my_cond_indep_chisquare(data1,i, class_index, [], test, alpha, ns);  % Commented by Liu
    [CI] = ml_cond_indep_chisquare(data, i, L, [], w, test, alpha, ns);   % Add by Liu
    
    if CI == 0
        stop = 1;
    end
    
    if stop
        
        if ~isempty(selected_features)
            %[CI]=compter_dep_2(selected_features,i,class_index, 3, 1, alpha, test,data); % Commented by Liu
            [CI] = ml_compter_dep_2(selected_features, i, L, 3, 1, w, alpha, test, data); % Add by Liu
        end
        
        if CI == 0
            
            selected_features  = [selected_features, i];
            selected_features1 = selected_features;
            p2                 = length(selected_features);
            
            for j = 1 : p2
                
                b = setdiff(selected_features1, selected_features(j), 'stable');
                if ~isempty(b)
                    
                    %[CI]=optimal_compter_dep_2(b,selected_features(j),class_index,3, 1, alpha, test,data);         % Commented by Liu
                    [CI] = ml_optimal_compter_dep_2(b, selected_features(j), L, 3, 1, w, alpha, test, data); % Add by Liu
                    
                    if CI == 1
                        selected_features1 = b;
                    end %if
                end
            end %for
        end %if
    end %if
    selected_features = selected_features1;
    
    disp([num2str(i), ': ', num2str(selected_features)]);% Add by Liu
    
end %for

time = toc(start);

end
