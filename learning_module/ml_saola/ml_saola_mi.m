function [current_feature,time] = ml_saola_mi(data,L,weight,threshold)

%Performs the SAOLA algorithm using mutual information measure by Yu 2014.
%data: columns denote features (attributes), while rows represent data
%instances. if data is the sparse format, please using full(data)
%the last column of a data set is the class attribute
%
%input
% data:      full data set
% L:         label set indexes
% weight:    label weight strategy
% threshold: dependency threshold, default 0

%output
% current_feature: selected features
% time:            running time

if nargin<4
    threshold = 0;
end

start=tic;

% numFeatures = size(data,2);
%
% class_a=numFeatures;%the index of the class attribute
%
% current_feature=[];
%
% dep=sparse(1,numFeatures-1);

% Add by Liu
W = weighting(data, L, weight); %weight vector

current_feature=[];

col = size(data,2);

dep=sparse(1,col-length(L));
% Addition ends

for i = 1:col-length(L)
    
    %for very sparse data
    n1=sum(data(:,i));
    if n1==0
        continue;
    end
    % [dep(i)] = SU(data(:,i),data(:,class_a));
    
    % Add by Liu
    dep(i) = ML_SU(data,i,L,W);
    % Addition ends
    
    if dep(i) <= threshold
        continue;
    end
    
    current_feature=[current_feature, i];
    
    %current_feature1=setdiff(current_feature,i,'stable');
    current_feature1=current_feature(~sum(bsxfun(@eq,current_feature',i),2));
    
    if ~isempty(current_feature1)
        
        p=length(current_feature1);
        
        for j=1:p
            
            [dep_ij] = SU(data(:,i),data(:,current_feature1(j)));
            
            if dep_ij<=threshold
                continue;
            end
            
            max_dep=dep_ij;
            max_feature=current_feature1(j);
            
            if dep(max_feature)>dep(i) && max_dep>dep(i)
                
                %current_feature=setdiff(current_feature,i, 'stable');
                current_feature=current_feature(~sum(bsxfun(@eq,current_feature',i),2));
                break;
            end
            
            if dep(i)>dep(max_feature) && max_dep>dep(max_feature)
                
                %current_feature=setdiff(current_feature,max_feature, 'stable');
                current_feature=current_feature(~sum(bsxfun(@eq,current_feature',max_feature),2));
            end
        end
    end
    disp([num2str(i), ': ', num2str(current_feature)]);% Add by Liu
end

time=toc(start);
