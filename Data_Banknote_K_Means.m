function [ cluster_assignments, rate_of_error_training, rate_of_error_test ] = Data_Banknote_K_Means( filename )

% initializations
wavCurr = csvread(filename);
wavFeats = wavCurr(:, 1:4); 
real_data = wavFeats(1:762, :);
fake_data = wavFeats(763:1372, :);
num_real_rows = size(real_data, 1);
num_fake_rows = size(fake_data, 1);

% calculating num rows in real test set and fake test set
n_r = round(num_real_rows * 0.1); 
n_f = round(num_fake_rows * 0.1);

% initialize training data
training_data = wavFeats;

% initialize array for test indices and test data
test_indices = zeros(n_r + n_f, 1);
test_data = zeros(n_r + n_f, 4); 
training_truth_values = zeros(size(training_data,1),1);

% find random test data indices 
for i = 1:n_r 
    index = randi(num_real_rows); 
    test_indices(i, :) = index; 
    test_data(i, :) = real_data(index, :); 
    training_truth_values(i) = 0; % real data is 0
end
for i = n_r + 1:n_f + n_r
    index = randi(num_fake_rows);
    test_indices(i, :) = index; 
    test_data(i, :) = fake_data(index, :); 
    training_truth_values(i) = 1; % fake data is 1
end

% exclude test data from training data set
training_data(test_indices, :) = [];
training_truth_values(test_indices) = [];
length_training_data_banknote = size(training_data, 1);

% isolating columns 1 and 2 (features)
training_data_col_12 = horzcat(training_data(:, 1), training_data(:, 2));

% Calculating K-Means 
[idx, cluster_assignments] = kmeans(training_data_col_12, 2, 'Distance', 'Cityblock', 'Replicates', 5);
isSwapOk = 0;
for i = 1:size(idx,1)
    if (idx(i) == 1)
        if (training_truth_values(i) == 1) % basically, the first index should be real shit
            % the training data should THEN have a value of 0 
            isSwapOk = isSwapOk + 1;
        end
    end
    if (idx(i) == 2)
        if training_truth_values(i) == 0 
            isSwapOk = isSwapOk + 1;
        end
    end
end

if (isSwapOk > 600) 
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

% Test data Euclidean distance to Centroids
centDist = pdist2(training_data_col_12, cluster_assignments);

% Plotting K Means Clustering
title('K Means: Data Banknote Authent, Columns 1 and 2');
hold on;
plot(training_data_col_12(idx==1,1), training_data_col_12(idx==1,2), 'r.','MarkerSize', 20);
hold on;
plot(training_data_col_12(idx==2,1), training_data_col_12(idx==2,2), 'b.', 'MarkerSize', 20)
hold on;

% Plotting True/Fake Training Data
plot(real_data(:, 1), real_data(:, 2), 'r.', 'MarkerSize', 10);
hold on;
plot(fake_data(:, 1), fake_data(:, 2), 'b.', 'MarkerSize', 10);
hold on;
plot(cluster_assignments(:,1),cluster_assignments(:,2),'kx',....
    'MarkerSize', 15, 'LineWidth', 3);
hold on;

legend('Real Cluster','Fake Cluster','Real Data','Fake Data','Centroids');

% Find Error in Training Data
min(centDist); 
instances_of_error_training = zeros(50, 1);
whichCenter = zeros(length_training_data_banknote, 1);
for i = 1:length_training_data_banknote
    if centDist(i, 1) > centDist(i, 2) 
        whichCenter(i) = 1;
        if i > length_training_data_banknote
            instances_of_error_training(i) = 1; 
        end 
    else
        whichCenter(i) = 2;
        if i <= (length_training_data_banknote / 2)
            instances_of_error_training(i) = 1;
        end 
    end
end


% find instances of error and append to new array
error_data = zeros(100, 4);
for i = 1:50
    if instances_of_error_training(i) == 1
        error_data(i, :) = training_data(i);
    end
end

number_of_errors = 0;
for i = 1:50
    if instances_of_error_training(i) == 1
        number_of_errors = number_of_errors + 1;
    end
end

rate_of_error_training = number_of_errors / length_training_data_banknote;

                    %%% ASSIGNING LABELS TO TEST SET %%%
                    
% Test data Euclidean distance to Centroids
test_data_col_12 = horzcat(test_data(:,1), test_data(:,2));
length_test_data_banknote = size(test_data_col_12, 1);

test_centDist = pdist2(test_data_col_12, cluster_assignments);

% Find Error in Training Data
min(centDist); 
instances_of_error_test = zeros(50, 1);
whichCenter_test = zeros(length_test_data_banknote, 1);
for i = 1:length_test_data_banknote
    if test_centDist(i, 1) > test_centDist(i, 2) 
        whichCenter_test(i) = 1;
        if i > length_test_data_banknote
            instances_of_error_test(i) = 1; 
        end 
    else
        whichCenter_test(i) = 2;
        if i <= (length_test_data_banknote / 2)
            instances_of_error_test(i) = 1;
        end 
    end
end


% Plotting K Means Clustering on Test Data
figure();
title('K Means: Data Banknote, Columns 1 and 2, Test Error');
hold on;
plot(test_data(whichCenter_test==1, 1), test_data(whichCenter_test==1, 2), 'c.', 'MarkerSize', 20);
hold on;
plot(test_data(whichCenter_test==2, 1), test_data(whichCenter_test==2, 2),  'm.', 'MarkerSize', 20);
hold on; 
plot(test_data(1:n_r, 1), test_data(1:n_r, 2), 'c.', 'MarkerSize', 10);
hold on;
plot(test_data(n_r + 1:n_r + n_f, 1), test_data(n_r + 1:n_r + n_f, 2), 'm.', 'MarkerSize', 10);
hold on;
plot(cluster_assignments(:,1),cluster_assignments(:,2),'kx',....
    'MarkerSize', 15, 'LineWidth', 3);
hold on;
legend('Real Cluster','Fake Cluster','Real Test Data','Fake Test Data','Centroids');

error_data_test = zeros(100, 4);
for i = 1:50
    if instances_of_error_test(i) == 1
        error_data_test(i, :) = test_data(i);
    end
end

number_of_errors_test = 0;
for i = 1:50
    if instances_of_error_test(i) == 1
        number_of_errors_test = number_of_errors_test + 1;
    end
end

rate_of_error_test = number_of_errors_test / (n_r + n_f);
end
