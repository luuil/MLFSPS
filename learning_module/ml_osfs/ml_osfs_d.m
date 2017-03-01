function [selected_features, time] = ml_osfs_d(data, L, alpha, test, weight)
% for disccrete data

start              = tic;
col                = size(data, 2);
ns                 = max(data);
selected_features  =[];
selected_features1 =[];
% b=[]; %commented by Liu

time_start         = cputime;
w                  = weighting(data, L, weight); %weight vector
fprintf('Label weight calculation costs: %d s\n', cputime - time_start);

for i = 1 : col-length(L)
    
    
    %for very sparse data
    n1=sum(data(:,i));
    if n1==0
        continue;
    end
    
    stop=0;
    %CI=1;
    
    [CI] = ml_cond_indep_chisquare(data, i, L, [], w, test, alpha, ns);
    
    if CI==0
        stop=1;
        selected_features = [selected_features,i];
    end
    
    if stop
        
        p2=length(selected_features);
        selected_features1=selected_features;
        
        for j=1:p2
            
            b=setdiff(selected_features1, selected_features(j),'stable');
            
            if ~isempty(b)
                [CI] = ml_compter_dep_2(b,selected_features(j), L, 3, 1, w, alpha, test, data);
                
                if CI==1
                    selected_features1=b;
                end
            end
        end
    end
    selected_features = selected_features1;
    
    disp([num2str(i), ': ', num2str(selected_features)]); % Add by Liu
end

time=toc(start);