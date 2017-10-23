% NOTE: 
% If you have not already imported the "swissbanknote" variable, this
% script will not run. Directions for how to import "swissbanknote" are
% included in the README. 

filename = 'data_banknote_authent.csv';

% K MEANS
[ kmeans_databanknote_centroid_assignments, kmeans_data_banknote_rate_of_error_training, kmeans_databanknote_rate_of_error_test ] = Data_Banknote_K_Means( filename );
figure();
[ kmeans_swiss_centroid_assignments, kmeans_swiss_rate_of_error_training, kmeans_swiss_rate_of_error_test ] = Swiss_Banknote_K_Means( swissbanknote );

% K NEAREST NEIGHBOR
figure();
[ knn_databanknote_training_error_features, knn_databanknote_training_error_general, knn_databanknote_test_error ] = Data_Banknote_KNN( filename );
figure();
[ knn_swiss_training_error_features, knn_swiss_training_error_general, knn_swiss_test_error ] = Swiss_Banknote_KNN( swissbanknote );

% NAIVE BAYES
figure();
[ nb_databanknote_training_error_features, nb_databanknote_training_error_general, nb_databanknote_test_error ] = Data_Banknote_Naive_Bayes( filename );
figure();
[ nb_swiss_training_error_features, nb_swiss_training_error_general, nb_swiss_test_error ] = Swiss_Banknote_Naive_Bayes( swissbanknote );


