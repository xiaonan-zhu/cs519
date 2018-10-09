#### CS 519 Project 3 Xiaonan Zhu

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import time
start_time = time.time()

import sys

def main():

    if sys.argv[1]=="digits":

        from sklearn import datasets

        df=datasets.load_digits()
        X=df["data"]
        y=df["target"]
        #print(X)


    elif sys.argv[1][-4:]==".log":

        df= pd.read_csv(sys.argv[1], header=None, sep='\t')
        X = df.iloc[:, list(range(int(sys.argv[5]), int(sys.argv[6])+1))].values
        y = df.iloc[:, int(sys.argv[7])].values

    else:
        #print("INVALID ARGUMENTS")
        print("Wrong data")



    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler()
    sc.fit(X)
    X_std = sc.transform(X)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, stratify=y, random_state=1)


    if sys.argv[2] =="Perceptron":

        from sklearn.linear_model import Perceptron
        ppn= Perceptron(max_iter=int(sys.argv[3]), eta0=float(sys.argv[4]), random_state=1)
        ppn.fit(X_train, y_train)

        y_pred=ppn.predict(X_test)
        error = (y_test != y_pred).sum()
        accuracy=(1-error/y_test.shape[0])*100
        print('Misclassified samples: {}.'.format(error))
        print("The accuracy is {:1.2f}%.".format(accuracy))
        print("The running time is {:1.4f} seconds.".format(time.time() - start_time))

    elif sys.argv[2] =="SVM":

        from sklearn.svm import SVC
        svm=SVC(kernel="rbf", random_state=1, gamma=float(sys.argv[3]), C=float(sys.argv[4]))
        svm.fit(X_train, y_train)

        y_pred=svm.predict(X_test)
        error = (y_test != y_pred).sum()
        accuracy=(1-error/y_test.shape[0])*100
        print('Misclassified samples: {}.'.format(error))
        print("The accuracy is {:1.2f}%.".format(accuracy))
        print("The running time is {:1.4f} seconds.".format(time.time() - start_time))

    elif sys.argv[2]=="DecisionTree":

        from sklearn. tree import DecisionTreeClassifier
        tree=DecisionTreeClassifier(criterion="gini", max_depth=int(sys.argv[3]), min_samples_leaf=int(sys.argv[4]), random_state=1)
        tree.fit(X_train, y_train)
           
        y_pred=tree.predict(X_test)
        error = (y_test != y_pred).sum()
        accuracy=(1-error/y_test.shape[0])*100
        print('Misclassified samples: {}.'.format(error))
        print("The accuracy is {:1.2f}%.".format(accuracy))
        print("The running time is {:1.4f} seconds.".format(time.time() - start_time))

    elif sys.argv[2]=="KNN":

        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier (n_neighbors=int(sys.argv[3]) , p=int(sys.argv[4]) , metric="minkowski")
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        error = (y_test != y_pred).sum()
        accuracy=(1-error/y_test.shape[0])*100
        print('Misclassified samples: {}.'.format(error))
        print("The accuracy is {:1.2f}%.".format(accuracy))
        print("The running time is {:1.4f} seconds.".format(time.time() - start_time))


    elif sys.argv[2]=="Log":

        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=float(sys.argv[3]) , random_state =1)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        error = (y_test != y_pred).sum()
        accuracy=(1-error/y_test.shape[0])*100
        print('Misclassified samples: {}.'.format(error))
        print("The accuracy is {:1.2f}%.".format(accuracy))
        print("The running time is {:1.4f} seconds.".format(time.time() - start_time))

    else:
        print("INVALID ARGUMENTS")












if __name__ == '__main__':
    main()