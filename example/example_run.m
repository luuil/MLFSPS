weight = 'SCA';
data   = enron;
col    = size(data, 2);
L      = col - labels_N + 1 : col;

[selected_features, time] = ml_alpha_investing(data, L, weight);