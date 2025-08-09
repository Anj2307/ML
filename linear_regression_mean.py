import numpy as np
import matplotlib.pyplot as plt
class SimpleLinearRegression:
    def __init__(self):
        self.m=None
        self.b=None
    def fit(self,X,Y):
        X=np.array(X)
        Y=np.array(Y)
        x_mean=np.mean(X)
        y_mean=np.mean(Y)
        num=np.sum((X-x_mean)*(Y-y_mean))
        dino=np.sum((X-x_mean)**2)
        self.m=num/dino
        self.b=y_mean-self.m*x_mean
    def predict(self,X):
        X=np.array(X)
        return self.m*X+self.b
    def get_para(self):
        return self.m,self.b
    
# examples
X=[1,2,3,4,5]
Y=[1,2,2,1.3,3.75]

model=SimpleLinearRegression()
model.fit(X,Y)
print(model.predict(X))
print(model.get_para())
y_pred=model.predict(X)

#visualization
plt.scatter(X,Y,color="blue",label="actual data")
plt.plot(X,y_pred,color="red",label="predicted line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()