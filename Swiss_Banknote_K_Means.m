function [ cluster_assignments, rate_of_error_training, rate_of_error_test ] = Swiss_Banknote_K_Means( swissbanknote )

% NOTE: YOU MUST IMPORT "swissbanknote" AS A VARIABLE BEFORE RUNNING THIS
% FUNCTION. FOR INSTRUCTIONS ON HOW TO DO THIS, PROCEED TO THE README.

%initializations 
real_data = swissbanknote(1:100, :);
fake_data = swissbanknote(101:200, :);

% Find the number of 10% of the rows for real and fake data
num_real_rows = size(real_data, 1); 
num_fake_rows = size(fake_data, 1);
n_r = round(num_real_rows * 0.1); 
n_f = round(num_fake_rows * 0.1); 

% make array for training data
training_data = vertcat(real_data, fake_data);

% make array for test data indices and test data
test_indices = zeros(n_r + n_f, 1);
test_data = zeros(n_r + n_f, 6); 

% randomly select test data indices
test_labels = zeros(n_r + n_f,1);
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

% isolating columns 5 and 6
training_data_col_56 = horzcat(training_data(:, 5), training_data(:, 6));
length_training_data = size(training_data_col_56, 1);

% Run K-Means Algorithm
[idx,cluster_assignments] = kmeans(training_data_col_56, 2, 'Distance', 'Cityblock', 'Replicates', 5);
isSwapOk = 0;
for i = 1:size(idx,1)
    if (idx(i) == 1)
        if (i > 100) 
            isSwapOk = isSwapOk + 1;
        end
    end
end

if (isSwapOk > 50) 
    temp = cluster_assignments(1, :);
    cluster_assignments(1, :) = cluster_assignments(2, :);
    cluster_assignments(2, :) = temp;
    for i = 1:size(idx,1)
        if idx(i) == 1
            idx(i) = 2;
        else
            idx(i) = 1;
        end
    end
end

% compute euclidean distance between training data and clusters
centDist = pdist2(training_data_col_56, cluster_assignments);


% Plotting K Means Clustering
% ISSUE: this assumes that the first index is the real index, and that the
% second index is the fake index
title('K Means: Swiss Banknote, Columns 5 and 6, Training Data');
hold on;
plot(training_data_col_56(idx==1,1), training_data_col_56(idx==1,2), 'r.','MarkerSize', 20);
hold on;
plot(training_data_col_56(idx==2,1), training_data_col_56(idx==2,2), 'b.', 'MarkerSize', 20)
hold on;

% Plotting Original Training Data On Top of K Means
plot(real_data(:, 5), real_data(:, 6), 'r.', 'MarkerSize', 10);
hold on;
plot(fake_data(:, 5), fake_data(:, 6), 'b.', 'MarkerSize', 10);
hold on;
plot(cluster_assignments(:,1),cluster_assignments(:,2),'kx',....
    'MarkerSize', 15, 'LineWidth', 3);
hold on;
legend('Real Cluster', 'Fake Cluster', 'Real Data', 'Fake Data', 'Centroids');

% Find Error in Training Data Clustering
min(centDist); 
instances_of_error_training = zeros(50, 1);
whichCenter = zeros(length_training_data, 1);
for i = 1:length_training_data
    if centDist(i, 1) > centDist(i, 2) 
        whichCenter(i) = 1;
        if i > length_training_data
            instances_of_error_training(i) = 1; 
        end 
    else
        whichCenter(i) = 2;
        if i <= (length_training_data / 2)
            instances_of_error_training(i) = 1;
        end 
    end
end


% find instances of error and append to new array
error_data = zeros(100, 6);
for i = 1:50
    if instances_of_error_training(i) == 1
        error_data(i, :) = training_data(i);
    end
end

% count instances of error
number_of_errors = 0;
for i = 1:50
    if instances_of_error_training(i) == 1
        number_of_errors = number_of_errors + 1;
    end
end

% compute rate of error in training data
rate_of_error_training = number_of_errors / length_training_data;

                    %%% ASSIGNING LABELS TO TEST SET %%%


% Test data Euclidean distance to Centroids
test_data_col_56 = horzcat(test_data(:,5), test_data(:,6));
length_test_data = size(test_data_col_56, 1);

test_centDist = pdist2(test_data_col_56, cluster_assignments);

% Find Error in Test Data
min(centDist); 
instances_of_error_test = zeros(50, 1);
whichCenter_test = zeros(length_test_data, 1);
for i = 1:length_test_data
    if test_centDist(i, 1) > test_centDist(i, 2) 
        whichCenter_test(i) = 1;
        if i > length_test_data
            instances_of_error_test(i) = 1; 
        end 
    else
        whichCenter_test(i) = 2;
        if i <= (length_test_data / 2)
            instances_of_error_test(i) = 1;
        end 
    end
end


% Plotting K Means Clustering on Test Data
figure();
title('K Means: Swiss Banknote, Columns 5 and 6, Test Error');
hold on;
plot(test_data(whichCenter_test==1, 5), test_data(whichCenter_test==1, 6), 'c.', 'MarkerSize', 20);
hold on;
plot(test_data(whichCenter_test==2, 5), test_data(whichCenter_test==2, 6),  'm.', 'MarkerSize', 20);
hold on; 
plot(test_data(test_labels==0, 5), test_data(test_labels==0, 6), 'c.', 'MarkerSize', 10);
hold on;
plot(test_data(test_labels==1, 5), test_data(test_labels==1, 6), 'm.', 'MarkerSize', 10);
hold on;
plot(cluster_assignments(:,1),cluster_assignments(:,2),'kx',....
    'MarkerSize', 15, 'LineWidth', 3);
hold on;
legend('Real Cluster','Fake Cluster','Real Data', 'Fake Data', 'Centroids');

% find instances of error and append to new array
error_data_test = zeros(100, 4);
for i = 1:50
    if instances_of_error_test(i) == 1
        error_data_test(i, :) = test_data(i);
    end
end

% count instances of error
number_of_errors_test = 0;
for i = 1:50
    if instances_of_error_test(i) == 1
        number_of_errors_test = number_of_errors_test + 1;
    end
end

% compute rate of error for test data
rate_of_error_test = number_of_errors_test / (n_r + n_f);

end

