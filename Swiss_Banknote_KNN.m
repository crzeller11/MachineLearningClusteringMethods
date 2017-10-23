function [ rate_training_error_features, rate_training_error_general, rate_test_error ] = Swiss_Banknote_KNN( swissbanknote )

%initializations 
real_data = swissbanknote(1:100, :);
fake_data = swissbanknote(101:200, :);

% Find the number of 10% of the rows for real and fake data
num_real_rows = size(real_data, 1); 
num_fake_rows = size(fake_data, 1);
n_r = round(num_real_rows * 0.1); 
n_f = round(num_fake_rows * 0.1); 

% make a new array for the 90% of data
training_data = vertcat(real_data, fake_data);

% make_array for test data indices
test_indices = zeros(n_r + n_f, 1);

labels = zeros(num_real_rows + num_fake_rows, 1);
for i = 1:num_real_rows + num_fake_rows
    if i <= 100
        labels(i) = 1;
    else
        labels(i) = 0;
    end
end

% make a new array for test data
test_data = zeros(n_r + n_f, 6); 
test_labels = zeros(n_r + n_f, 1);
for i = 1:n_r 
    index = randi(num_real_rows); 
    test_indices(i, :) = index; 
    test_data(i, :) = real_data(index, :);
    test_labels(i) = 1;
end
for i = n_r + 1:n_f + n_r
    index = randi(num_fake_rows);
    test_indices(i, :) = index; 
    test_data(i, :) = fake_data(index, :); 
    test_labels(i) = 0;
end


% Training Data Array
training_data(test_indices, :) = [];
labels(test_indices, :) = [];
knn_training_data_col_56 = horzcat(training_data(:, 5), training_data(:, 6));

% pass all 6 columns as training data, then pass the real and fake column
% K Nearest Neighbor
modelKNN_56 = fitcknn(knn_training_data_col_56, labels, 'NumNeighbors', 4, 'Distance', 'cosine');
modelKNN = fitcknn(training_data, labels,'NumNeighbors', 4, 'Distance', 'cosine');

% Check your error on training set
prediction_labels_training_56 = predict(modelKNN_56, knn_training_data_col_56);
prediction_labels = predict(modelKNN, training_data);
rate_training_error_features = sum(abs(prediction_labels_training_56 - labels)) / size(training_data,1);
rate_training_error_general = sum(abs(prediction_labels - labels)) / size(training_data,1);

% Plotting K Nearest Neighbor Clustering
title('KNN: Swiss Banknote, Columns 5 and 6, Training Data');
hold on;
plot(knn_training_data_col_56(prediction_labels_training_56==0,1), knn_training_data_col_56(prediction_labels_training_56==0,2), 'r.','MarkerSize', 20);
hold on;
plot(knn_training_data_col_56(prediction_labels_training_56==1,1), knn_training_data_col_56(prediction_labels_training_56==1,2), 'b.', 'MarkerSize', 20);
hold on;
plot(knn_training_data_col_56(labels==0, 1), knn_training_data_col_56(labels==0, 2), 'r.', 'MarkerSize', 10);
hold on;
plot(knn_training_data_col_56(labels==1, 1), knn_training_data_col_56(labels==1, 2), 'b.', 'MarkerSize', 10);
hold on;
legend('Real Cluster', 'Fake Cluster', 'Real Data','Fake Data');

                            %%% TEST DATA %%%
                            
% Check your error on training set
test_data_56 = horzcat(test_data(:,5), test_data(:,6));
prediction_labels_56_test = predict(modelKNN_56, test_data_56);
rate_test_error = sum(abs(prediction_labels_56_test - test_labels)) / size(training_data,1);

% Plotting Test Data Error
figure();
title('KNN: Swiss Banknote, Columns 5 and 6, Test Data');
hold on;
plot(test_data_56(prediction_labels_56_test==0, 1), test_data_56(prediction_labels_56_test==0,2),'m.','MarkerSize', 20);
hold on;
plot(test_data_56(prediction_labels_56_test==1, 1), test_data_56(prediction_labels_56_test==1,2),'c.','MarkerSize', 20);
hold on;
plot(test_data_56(test_labels==0, 1), test_data_56(test_labels==0, 2), 'm.', 'MarkerSize', 10);
hold on;
plot(test_data_56(test_labels==1, 1), test_data_56(test_labels==1, 2), 'c.', 'MarkerSize', 10);
hold on;
legend('Real Cluster','Fake Cluster','Real Test Data','Fake Test Data');


end
