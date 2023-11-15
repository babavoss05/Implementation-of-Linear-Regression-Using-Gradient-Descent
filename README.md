![image](https://github.com/babavoss05/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/103019882/6221dc76-4771-4a70-8a3a-399aef8e2b14)# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Use the standard libraries such as numpy, pandas, matplotlib.pyplot in python for the Gradient Descent.
Step 2. Upload the dataset conditions and check for any null value in the values provided using the .isnull() function.
Step3. Declare the default values such as n, m, c, L for the implementation of linear regression using gradient descent.
Step 4. Calculate the loss using Mean Square Error formula and declare the variables y_pred, dm, dc to find the value of m.
Step 5. Predict the value of y and also print the values of m and c.
Step 6. Plot the accquired graph with respect to hours and scores using the scatter plot function.



## Program:
Program to implement the linear regression using gradient descent.
Developed by: Gokul
RegisterNumber:  212221220013
```py
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  """
  Take in a numpy array X,y,theta and generate the cost function in a linear regression model

  """
  m=len(y)  
  h=X.dot(theta)
  square_err=(h - y)**2
  return 1/(2*m) * np.sum(square_err)

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  """
   Take in numpy array X,y and theta and update theta by taking number with learning rate of alpha

  return theta and the list of the cost of theta during each iteration
  """
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions -y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))

```

## Output:

![image](https://github.com/babavoss05/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/103019882/753073f8-d0a4-4561-bd90-257b00163d45)
### Profit Prediction graph:
![image](https://github.com/babavoss05/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/103019882/30e2579d-992e-4597-9d10-564e70c388ad)
### Compute Cost Value:
![image](https://github.com/babavoss05/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/103019882/b600e0c2-04c3-4ec1-bb60-ed78c614d29a)
### h(x) Value:
![image](https://github.com/babavoss05/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/103019882/f94d90a3-44b0-4932-8d18-fef6de1bfdd7)
### Cost function using Gradient Descent Graph:
![image](https://github.com/babavoss05/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/103019882/310fe9cb-2bbc-4fec-88cf-6fb726af9ed5)
### Profit Prediction Graph:
![image](https://github.com/babavoss05/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/103019882/60b3381f-97d1-43f8-ae5e-07e4705211ee)
### Profit for the Population 35,000:
![image](https://github.com/babavoss05/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/103019882/3781f7e5-f20d-4227-9eb7-2ab375b5313b)
### Profit for the Population 70,000:
![image](https://github.com/babavoss05/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/103019882/df258169-7789-4ee3-918a-e071c2612e45)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
