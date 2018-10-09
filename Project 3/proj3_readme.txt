#### CS 519 Project 3 Xiaonan Zhu

The data set I used is subject1_ideal.log in REALDISP Activity Recognition Dataset (https://archive.ics.uci.edu/ml/datasets/
REALDISP+Activity+Recognition+Dataset) Dr. Cao provided. 


# Arguments:

# python, file name, data, Perceptron, num of iteration, eta
python project3.py digits Perceptron 5 0.1
python project3.py digits Perceptron 20 0.1
python project3.py digits Perceptron 5 0.01
python project3.py digits Perceptron 40 0.01

# python, file name, path of data, Perceptron, num of iteration, eta, initial column of X, end column of X, column of Y
python project3.py C:/Users/xiaon/Desktop/subject1_ideal.log Perceptron 5 0.1 2 118 119
python project3.py C:/Users/xiaon/Desktop/subject1_ideal.log Perceptron 20 0.1 2 118 119
python project3.py C:/Users/xiaon/Desktop/subject1_ideal.log Perceptron 5 0.01 2 118 119
python project3.py C:/Users/xiaon/Desktop/subject1_ideal.log Perceptron 40 0.01 2 118 119




# python, file name, data, SVM, gamma, C
python project3.py digits SVM 0.8 0.5 
python project3.py digits SVM 0.8 1
python project3.py digits SVM 0.8 5
python project3.py digits SVM 0.2 0.5
python project3.py digits SVM 0.2 1
python project3.py digits SVM 0.2 5
python project3.py digits SVM 0.02 0.5
python project3.py digits SVM 0.02 1
python project3.py digits SVM 0.02 5

# python, file name, path of data, SVM, gamma, C, initial column of X, end column of X, column of Y
python project3.py C:/Users/xiaon/Desktop/subject1_ideal.log SVM 0.02 5 2 118 119




# python, file name, data, DecisionTree, max_depth, min_samples_leaf
python project3.py digits DecisionTree 4 1 
python project3.py digits DecisionTree 4 5
python project3.py digits DecisionTree 40 1
python project3.py digits DecisionTree 40 5

# python, file name, path of data, DecisionTree, max_depth, min_samples_leaf, initial column of X, end column of X, column of Y
python project3.py C:/Users/xiaon/Desktop/subject1_ideal.log DecisionTree 4 1 2 118 119
python project3.py C:/Users/xiaon/Desktop/subject1_ideal.log DecisionTree 4 5 2 118 119
python project3.py C:/Users/xiaon/Desktop/subject1_ideal.log DecisionTree 40 1 2 118 119
python project3.py C:/Users/xiaon/Desktop/subject1_ideal.log DecisionTree 40 5 2 118 119




# python, file name, data, KNN, n_neighbors, p
python project3.py digits KNN 1 2
python project3.py digits KNN 5 2
python project3.py digits KNN 10 2

# python, file name, path of data, KNN, n_neighbors, p, initial column of X, end column of X, column of Y
python project3.py C:/Users/xiaon/Desktop/subject1_ideal.log KNN 5 2 2 118 119





# python, file name, data, Log Regression, C 
python project3.py digits Log 0.01
python project3.py digits Log 1
python project3.py digits Log 100

# python, file name, path of data, Log Regression, C, *(fake argument), initial column of X, end column of X, column of Y
python project3.py C:/Users/xiaon/Desktop/subject1_ideal.log Log 0.01 * 2 118 119
python project3.py C:/Users/xiaon/Desktop/subject1_ideal.log Log 1 * 2 118 119
python project3.py C:/Users/xiaon/Desktop/subject1_ideal.log Log 100 * 2 118 119









