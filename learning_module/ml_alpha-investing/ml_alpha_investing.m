% Below is the main streamwise feature selection (SFS) code. It uses two helper functions, Linear_Regression and Prediction_Error

% the main function, Alpha_Investing

function [selected_features,time] = ml_alpha_investing(data, L, weight)

start=tic;

% Add by Liu
WV = weighting(data, L, weight); %weight vector
selected_features = setdiff(1:size(data,2), L, 'stable'); % feature indexes
F = data(:,selected_features); % Feature set
Y = data(:,L); % Label set
% Addition ends

% configure parameters (I never change these)
wealth = 0.5;
delta_alpha = 0.5;

% n observations; p features
[n,p] = size(F);

% initially add constant term into the model
model = [1, zeros(1,p-1)];
error = Prediction_Error(F(:,model==1), Y, Linear_Regression(F(:,model==1), Y, WV));

for i=2:p
    alpha = wealth/(2*i);
    
    %compute p_value
    %method one: derive delta(loglikelihood) from L2 error
    model(i) = 1;
    error_new = Prediction_Error(F(:,model==1), Y, Linear_Regression(F(:,model==1), Y, WV));
    sigma2 = error/n;
    p_value = exp((error_new-error)/(2*sigma2));
    
    %method two: derive delta(loglikelihood) from t-statistic
    %model(i) = 1;
    %w = Linear_Regression(X(:,model==1), y);
    %sigma2 = Prediction_Error(X(:,model==1), y, w)/n;
    %EX = mean(X(:,model==1));
    %w_new_std = w(end)/sqrt(sigma2/(sum(sum((X(:,model==1)-ones(n,1)*EX).^2, 2))));
    %p_value = 2*(1-normcdf(abs(w_new_std), 0, 1));
    
    if p_value < alpha %feature i is accepted
        model(i) = 1;
        error = error_new;
        wealth = wealth + delta_alpha - alpha;
    else %feature i is discarded
        model(i) = 0;
        wealth = wealth - alpha;
    end
    disp([num2str(i) '.']);
end

% train final model
w = zeros(p,1);
w(model==1,1) = Linear_Regression(F(:,model==1), Y, WV);
selected_features = find(model);

time=toc(start);
end


% Linear_Regression
function [w] = Linear_Regression(F, L, WV)
% this is not the most efficient way to find w!
% F: row*i, L: row*1 => w: i*1
L_n = size(L,2);
w   = zeros(size(F,2),1);
for i = 1:L_n
    w_i = (F'*F)\F'*L(:,i); %inv(X'*X)*X'*y;
    w_i = w_i * WV(i); %weighting
    w   = w + w_i;
end
end

% Prediction_Error
function [error] = Prediction_Error(F, L, w)
% X: row*i, w: i*1 => yhat: row*1
yhat = F*w;
L_n = size(L,2);
error = 0;
for i = 1:L_n
    error_i = sum((L(:,i)-yhat).^2); %error = sum((y-yhat).^2);
    error   = error + error_i;
end
error = error/L_n;
end
