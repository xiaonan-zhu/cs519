#### CS 519 Project 1 Xiaonan Zhu


# Arguments:
# python, file name, method, data path, initial column of X, end column of X, column of Y, class labeled as 1, whether separation is space, eta, num of iteration


python project1.py Perceptron C:/Users/xiaon/Desktop/iris.csv 0 3 4 Iris-setosa 0 0.01 50

python project1.py Adaline C:/Users/xiaon/Desktop/iris.csv 0 3 4 Iris-setosa 0 0.001 50

python project1.py SGD C:/Users/xiaon/Desktop/iris.csv 0 3 4 Iris-setosa 0 0.01 50



python project1.py Perceptron C:/Users/xiaon/Desktop/ecoli.csv 1 7 8 cp 1 0.001 50

python project1.py Adaline C:/Users/xiaon/Desktop/ecoli.csv 1 7 8 cp 1 0.001 50

python project1.py SGD C:/Users/xiaon/Desktop/ecoli.csv 1 7 8 cp 1 0.001 50



#One-vs-All
# Arguments:
# python, file name, method, data path, initial column of X, end column of X, column of Y, One-vs-All, whether separation is space, eta, num of iteration


python project1.py Perceptron C:/Users/xiaon/Desktop/iris.csv 0 3 4 OVA 0 0.001 1000

python project1.py Adaline C:/Users/xiaon/Desktop/iris.csv 0 3 4 OVA 0 0.001 1000

python project1.py SGD C:/Users/xiaon/Desktop/iris.csv 0 3 4 OVA 0 0.001 1000



python project1.py Perceptron C:/Users/xiaon/Desktop/ecoli.csv 1 7 8 OVA 1 0.001 1000

python project1.py Adaline C:/Users/xiaon/Desktop/ecoli.csv 1 7 8 OVA 1 0.001 1000

python project1.py SGD C:/Users/xiaon/Desktop/ecoli.csv 1 7 8 OVA 1 0.001 1000