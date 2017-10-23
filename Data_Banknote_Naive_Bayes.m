function [ rate_training_error_features, rate_training_error_general, rate_test_error ] = Data_Banknote_Naive_Bayes( filename )

% initializations
wavCurr = csvread(filename);

% separate real and fake data
real_data = wavCurr(1:762, :);  
fake_data = wavCurr(763:1372, :);
num_real_rows = size(real_data, 1);
num_fake_rows = size(fake_data, 1);

% Real Data Banknote
n_r = round(num_real_rows * 0.1); 
n_f = round(num_fake_rows * 0.1);

training_data = wavCurr;

% initialize arrays to hold test indices
test_indices = zeros(n_r + n_f, 1);

% make a new array for test data
test_data = zeros(n_r + n_f, 5); 
for i = 1:n_r 
    index = randi(num_real_rows); 
    test_indices(i, :) = index; 
    test_data(i, :) = real_data(index, :); 
end
for i = n_r + 1:n_f + n_r
    index = randi(num_fake_rows);
    test_indices(i, :) = index; 
    test_data(i, :) = fake_data(index, :); 
end

% Make an array of Training Data (the other 90% of data)
training_data(test_indices, :) = [];

nb_training_data = training_data(:, 1:4);
nb_training_data_col_12 = horzcat(training_data(:, 1), training_data(:, 2));
labels = training_data(:, 5);

% pass all 6 columns as training data, then pass the real and fake column
% K Nearest Neighbor
model_naive_bayes_12 = fitcnb(nb_training_data_col_12, labels);
model_naive_bayes = fitcnb(nb_training_data, labels);

% Checking error on training set
prediction_labels_12 = predict(model_naive_bayes_12, nb_training_data_col_12);
prediction_labels = predict(model_naive_bayes, nb_training_data);
rate_training_error_features = sum(abs(prediction_labels_12 - labels)) / size(training_data,1);
rate_training_error_general = sum(abs(prediction_labels - labels)) / size(training_data,1);

% Plotting K Nearest Neighbor Clustering
title('Naive Bayes: Data Banknote Authent, Columns 1 and 2, Training Data');
hold on;
plot(nb_training_data_col_12(prediction_labels_12==0,1), nb_training_data_col_12(prediction_labels_12==0,2), 'r.','MarkerSize', 20);
hold on;
plot(nb_training_data_col_12(prediction_labels_12==1,1), nb_training_data_col_12(prediction_labels_12==1,2), 'b.', 'MarkerSize', 20);
hold on;
plot(real_data(:, 1), real_data(:, 2), 'r.', 'MarkerSize', 10);
hold on;
plot(fake_data(:, 1), fake_data(:, 2), 'b.', 'MarkerSize', 10);
hold on;
legend('Real Cluster', 'Fake Cluster', 'Real Data','Fake Data');

                                %%% TEST DATA %%%

% Check your error on training set
test_labels = test_data(:, 5);
test_data_12 = horzcat(test_data(:,1), test_data(:,2));
prediction_labels_12_test = predict(model_naive_bayes_12, test_data_12);
rate_test_error = sum(abs(prediction_labels_12_test - test_labels)) / size(test_data,1);

% Plotting Test Data Error
figure();
title('Naive Bayes: Data Banknote Authent, Columns 1 and 2, Test Data');
hold on;
plot(test_data_12(prediction_labels_12_test==0, 1), test_data_12(prediction_labels_12_test==0,2),'m.','MarkerSize', 20);
hold on;
plot(test_data_12(prediction_labels_12_test==1, 1), test_data_12(prediction_labels_12_test==1,2),'c.','MarkerSize', 20);
hold on;
plot(test_data_12(test_labels==0, 1), test_data_12(test_labels==0, 2), 'm.', 'MarkerSize', 10);
hold on;
plot(test_data_12(test_labels==1, 1), test_data_12(test_labels==1, 2), 'c.', 'MarkerSize', 10);
hold on;
legend('Real Cluster','Fake Cluster','Real Test Data','Fake Test Data');

end
