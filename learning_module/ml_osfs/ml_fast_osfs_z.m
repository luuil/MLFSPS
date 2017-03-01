function [selected_features,time] = ml_fast_osfs_z(data, L, alpha)
% function [selected_features,time]= fast_osfs_z(data1,class_index,alpha) % Commented by Liu

%for continouous data

%%%%Add by Liu
%  Inputs:
%   data : a N * M matrix, with all features including the class attribute
%   L    : the indexes of the label set
%   alpha: significant level( 0.01 or 0.05 )
%
% Adopted for Online Multi-Label Feature Selection(OMLFS) by Liu, humooo@outlook.com
% 2016-04-29
%%%%Addition ends

%%%%Original contents(Commented by Liu)
%input parameter:
% data1:  data with all features including the class attribute.
% the class attribute in data matrix has to take consecutive integer values starting from 0 for classification.
% target: the index of the class attribute (we assume the class attribute is the last colomn of data1)
% alpha:  significant level( 0.01 or 0.05 )
%for example: The UCI dataset wdbc with 569 instances and 31 features (the index of the class attribute is 31).
% [selected_features1,time]=fast_osfs_z(wdbc,31,0.01)

%output:
% selected_features1: the selected features
% time:               running time

%please refer to the following papers for the details and cite them:
%Wu, Xindong, Kui Yu, Wei Ding, Hao Wang, and Xingquan Zhu. "Online feature selection with streaming features." Pattern Analysis and Machine Intelligence, IEEE Transactions on 35, no. 5 (2013): 1178-1192.
%%%%Contents ends

start              = tic;

[n, p]             = size(data);
selected_features  = [];
selected_features1 = [];
% b                = []; % Commented by Liu

for i = 1 : p-1 %the last feature is the class attribute, i.e., the target)

    %for very sparse data
    n1 = sum(data(:,i));
    if n1 == 0
        continue;
    end

    stop = 0;
    
    % CI=1;                                                                  % Commented by Liu
    % [CI,dep] = my_cond_indep_fisher_z(data1,i,class_index,[],n,alpha);     % Commented by Liu
    [CI, dep] = ml_cond_indep_fisher_z(data, i, L, [], n, alpha);    % Add by Liu
    
    if CI==1 || isnan(dep)
        continue;
    end
    
    if CI == 0
        stop = 1;
    end
    
    if stop
        
        if ~isempty(selected_features)
            % [CI,dep]= compter_dep_2(selected_features,i,class_index,3, 0, alpha, 'z',data1);         % Commented by Liu
            [CI, dep] = ml_compter_dep_2(selected_features, i, L, 3, 0, alpha, 'z', data);    % Add by Liu
        end
        
        if CI==0 && ~isnan(dep)
            
            selected_features  = [selected_features, i]; %adding i to the set of selected_features
            selected_features1 = selected_features;
            p2                 = length(selected_features);
            
            for j = 1 : p2
                
                b = setdiff(selected_features1, selected_features(j), 'stable');
                
                if ~isempty(b)
                    % [CI,dep]=optimal_compter_dep_2(b,selected_features(j),class_index,3, 0, alpha, 'z',data1);      % Commented by Liu
                    [CI, dep] = ml_optimal_compter_dep_2(b,selected_features(j), L, 3, 0, alpha, 'z', data); % Add by Liu
                    
                    if CI==1 || isnan(dep)
                        selected_features1=b;
                    end
                end
                
            end
        end
    end
    selected_features = selected_features1;
end

time = toc(start);



