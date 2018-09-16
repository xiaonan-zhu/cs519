#### CS 519 Project 1 Xiaonan Zhu


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Perceptron(object):
    
    def __init__(self, eta=0.1, n_iter=10, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0, scale=0.01, size=1+X.shape[1])
        
        self.num_errors = []
        
        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target-self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.num_errors.append(errors)
        
        return self
    
    def predict(self, X):
        z=np.dot(X, self.w_[1:])+ self.w_[0]
        return np.where(z >=0, 1, -1)






class Adaline(object):
    
    def __init__(self, eta = 0.1, n_iter = 10, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0, scale = 0.01, size = 1+X.shape[1])
        
        self.cost_ = []
        
        for i in range(self.n_iter):
            output = np.dot(X, self.w_[1:]) + self.w_[0]
            error = y-output
            self.w_[1:] += self.eta * X.T.dot(error)
            self.w_[0] += self.eta * error.sum()
            cost = (error**2).sum()/2.0
            self.cost_.append(cost)
        
        return self
    
    def predict(self, X):
        z=np.dot(X, self.w_[1:])+ self.w_[0]
        return np.where(z >=0, 1, -1)



class SGD(object):
    
    def __init__(self, eta = 0.1, n_iter = 10, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]
    
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0, scale = 0.01, size = 1+X.shape[1])

        self.cost_=[]
        for i in range(self.n_iter):
            X, y = self.shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self.update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self
    
    def update_weights(self, xi, target):
        output = np.dot(xi, self.w_[1:])+self.w_[0]
        error = (target-output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5*(error**2)
        return cost
    
    def predict(self, X):
        z=np.dot(X, self.w_[1:])+ self.w_[0]
        return np.where(z >=0, 1, -1)


import sys

def main():

    if len(sys.argv) == 10 and sys.argv[6]!="OVA":
        df= pd.read_csv(sys.argv[2], header=None, delim_whitespace=bool(int(sys.argv[7])))
        X = df.iloc[:, list(range(int(sys.argv[3]), int(sys.argv[4])+1))].values
        y = df.iloc[:, int(sys.argv[5])].values
        y = np.where(y == sys.argv[6], 1, -1)

        from sklearn.preprocessing import StandardScaler
        sc=StandardScaler()
        sc.fit(X)
        X_std = sc.transform(X)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, stratify=y, random_state=1)

        if sys.argv[1] =="Perceptron":
            ppn = Perceptron(eta=float(sys.argv[8]), n_iter=int(sys.argv[9]))
            ppn.fit(X_train, y_train)
            plt.plot(range(1, len(ppn.num_errors) + 1), ppn.num_errors, marker='o')
            plt.title('Perceptron - Learning rate is {}, Iteration is {}'. format(float(sys.argv[8]), int(sys.argv[9])))
            plt.xlabel('Epochs')
            plt.ylabel('Number of errors')
            plt.show()

            y_pred = ppn.predict(X_test)
            error=(y_test != y_pred).sum()
            accuracy=(1-error/y_test.shape[0])*100
            print("Learning rate is {}, Iteration is {}". format(float(sys.argv[8]), int(sys.argv[9])))
            print("The error of each iteration is {}.".format(['%.2f' % elem for elem in ppn.num_errors]))
            print("The accuracy is {:1.2f}%.".format(accuracy))

        elif sys.argv[1] =="Adaline":

            ada = Adaline(eta=float(sys.argv[8]), n_iter=int(sys.argv[9]))
            ada.fit(X_train, y_train)
            plt.plot(range(1, len(ada.cost_) + 1),ada.cost_, marker='o')
            plt.title('Adaline - - Learning rate is {}, Iteration is {}'. format(float(sys.argv[8]), int(sys.argv[9])))
            plt.xlabel('Epochs')
            plt.ylabel('Sum-squared-error')
            plt.show()

            y_pred = ada.predict(X_test)
            error=(y_test != y_pred).sum()
            accuracy=(1-error/y_test.shape[0])*100
            print("Learning rate is {}, Iteration is {}". format(float(sys.argv[8]), int(sys.argv[9])))
            print("The sum of squared error of each iteration is {}.".format(['%.2f' % elem for elem in ada.cost_]))
            print("The accuracy is {:1.2f}%.".format(accuracy))                

        elif sys.argv[1] =="SGD":

            sgd = SGD(eta=float(sys.argv[8]), n_iter=int(sys.argv[9]))
            sgd.fit(X_train, y_train)
            plt.plot(range(1, len(sgd.cost_) + 1),sgd.cost_, marker='o')
            plt.title('SGD - - Learning rate is {}, Iteration is {}'. format(float(sys.argv[8]), int(sys.argv[9])))
            plt.xlabel('Epochs')
            plt.ylabel('Average Cost')
            plt.show()

            y_pred = sgd.predict(X_test)
            error=(y_test != y_pred).sum()
            accuracy=(1-error/y_test.shape[0])*100
            print("Learning rate is {}, Iteration is {}". format(float(sys.argv[8]), int(sys.argv[9])))
            print("The sum of squared error of each iteration is {}.".format(['%.2f' % elem for elem in sgd.cost_]))
            print("The accuracy is {:1.2f}%.".format(accuracy))

        else:

            print("INVALID ARGUMENTS")




    elif len(sys.argv) == 10 and sys.argv[6]=="OVA":

        df= pd.read_csv(sys.argv[2], header=None, delim_whitespace=bool(int(sys.argv[7])))
        X = df.iloc[:, list(range(int(sys.argv[3]), int(sys.argv[4])+1))].values
        y = df.iloc[:, int(sys.argv[5])].values

        from sklearn.preprocessing import StandardScaler
        sc=StandardScaler()
        sc.fit(X)
        X_std = sc.transform(X)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, stratify=y, random_state=1)


        labels=list(set(df[int(sys.argv[5])]))
        labels.sort()

        label_test=[]

        for i in range(len(y_test)):
            for j in range(len(labels)):
                if y_test[i]== labels[j]:
                    label_test.append(j)

        counter=list(map(lambda _: [0]*len(labels), range(len(y_test))))

        for i in range(len(labels)):
            y_train_num = np.where(y_train == labels[i], 1, -1)

            if sys.argv[1] =="Perceptron":

                ppn = Perceptron(eta=float(sys.argv[8]), n_iter=int(sys.argv[9]))
                ppn.fit(X_train, y_train_num)
                y_pred = ppn.predict(X_test)

            elif sys.argv[1] =="Adaline":

                ada = Adaline(eta=float(sys.argv[8]), n_iter=int(sys.argv[9]))
                ada.fit(X_train, y_train_num)
                y_pred = ada.predict(X_test)

            elif sys.argv[1] =="SGD":
                sgd = SGD(eta=float(sys.argv[8]), n_iter=int(sys.argv[9]))
                sgd.fit(X_train, y_train_num)
                y_pred = sgd.predict(X_test)

            else:
                print("INVALID ARGUMENTS")



    
            for j in range(len(y_pred)):

                update1=[0]*len(labels)
                update2=[1]*len(labels)

                if y_pred[j]==1:
                    update1[i]=1
                    counter[j]=list(np.sum([counter[j], update1], axis=0))
                else:
                    update2[i]=0
                    counter[j]=list(np.sum([counter[j], update2], axis=0))

        label_pred=[]

        for k in range(len(y_pred)):
            label_pred.append(counter[k].index(max(counter[k])))

        error=(np.array(label_test) != np.array(label_pred)).sum()
        accuracy=(1-error/y_test.shape[0])*100
        print("The accuracy is {:1.2f}%.".format(accuracy))






    else:
        print("INVALID ARGUMENTS")








if __name__ == '__main__':
    main()