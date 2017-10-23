function [ rate_training_error_features, rate_training_error_general, rate_test_error ] = Swiss_Banknote_Naive_Bayes( swissbanknote )

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

% make a new array for test data
test_data = zeros(n_r + n_f, 6); 
test_labels = zeros(n_r+n_f,1);
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

training_labels = zeros(num_real_rows + num_fake_rows, 1);
for i = 1:num_real_rows + num_fake_rows
    if i <= 100
        training_labels(i) = 1;
    else
        training_labels(i) = 0;
    end
end

% Training Data Array
training_data(test_indices, :) = [];
training_labels(test_indices, :) = [];
nb_training_data_col_56 = horzcat(training_data(:, 5), training_data(:, 6));

% K Nearest Neighbor
model_naive_bayes_56 = fitcnb(nb_training_data_col_56, training_labels);
model_naive_bayes = fitcnb(training_data, training_labels);

% Check error
prediction_labels_56 = predict(model_naive_bayes_56, nb_training_data_col_56);
prediction_labels = predict(model_naive_bayes, training_data);
rate_training_error_features = sum(abs(prediction_labels_56 - training_labels)) / size(training_data,1); 
rate_training_error_general = sum(abs(prediction_labels - training_labels)) / size(training_data,1);


% Plotting NB Clustering
title('Naive Bayes: Swiss Banknote, Columns 5 and 6, Training Data');
hold on;
plot(nb_training_data_col_56(prediction_labels_56==0,1), nb_training_data_col_56(prediction_labels_56==0,2), 'r.','MarkerSize', 20);
hold on;
plot(nb_training_data_col_56(prediction_labels_56==1,1), nb_training_data_col_56(prediction_labels_56==1,2), 'b.', 'MarkerSize', 20);
hold on;
plot(nb_training_data_col_56(training_labels==0, 1), nb_training_data_col_56(training_labels==0, 2), 'r.', 'MarkerSize', 10);
hold on;
plot(nb_training_data_col_56(training_labels==1, 1), nb_training_data_col_56(training_labels==1, 2), 'b.', 'MarkerSize', 10);
hold on;
legend('Real Cluster', 'Fake Cluster', 'Real Data','Fake Data');

                                %%% TEST DATA %%%

% Check your error on training set
test_data_56 = horzcat(test_data(:,5), test_data(:,6));
prediction_labels_56_test = predict(model_naive_bayes_56, test_data_56);
rate_test_error = sum(abs(prediction_labels_56_test - test_labels)) / size(test_data,1);

% Plotting Test Data Error
figure();
title('Naive Bayes: Swiss Banknote, Columns 5 and 6, Test Data');
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