# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.
   
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: HARISH RAGAVENDRA S
RegisterNumber:  212222230045
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("ex1.txt",header =None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Prediction")

def computeCost(X,y,theta):

  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2
  j=1/(2*m)* np.sum(square_err)
  return j

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]

  for i in range (num_iters):
    predictions=X.dot(theta)
    error = np.dot(X.transpose(),(predictions-y))
    descent=alpha*1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))

  return theta,J_history  

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1" )

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Grading Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Prediction")

def predict (x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000,we predict a profit a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population =70,000,we predict a profit a profit of $"+str(round(predict2,0)))
```

## Output:
#### profit prediction graph:
![Screenshot from 2023-10-31 10-02-46](https://github.com/harish-ragavendra-25/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/114852180/a2b3cd78-9d33-499f-826d-212330b878ee)
#### cost compute value:
![Screenshot from 2023-10-31 10-03-45](https://github.com/harish-ragavendra-25/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/114852180/c661cf70-630d-47f3-9894-59c7e1100c39)
#### h(x) value:
![Screenshot from 2023-10-31 10-04-15](https://github.com/harish-ragavendra-25/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/114852180/f516bf8f-05be-4cf2-825e-6854bf562352)
#### cost function using gradient descent graph:
![Screenshot from 2023-10-31 10-04-32](https://github.com/harish-ragavendra-25/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/114852180/c4f7f581-c814-4048-bd9d-e24023986c69)
#### profit prediction graph:
![Screenshot from 2023-10-31 10-05-04](https://github.com/harish-ragavendra-25/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/114852180/2a4d0ef6-1638-488e-99f8-53a2bbcd9560)
#### profit of the population 35,000:
![Screenshot from 2023-10-31 10-05-32](https://github.com/harish-ragavendra-25/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/114852180/03a16265-6d62-40ae-8c2b-e49706600440)
#### profit of the poulation 70,000:
![Screenshot from 2023-10-31 10-06-20](https://github.com/harish-ragavendra-25/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/114852180/31ca62e5-7211-4afa-8451-bc81776842de)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
