from torch import nn

print("Hello Github")
print("My first AI project")



from sklearn.linear_model import LinearRegression
import numpy as np

x=np.array([[2,3],[4,5],[6,7],[8,9]])
y=np.array([20,40,60,80])

model = LinearRegression()
model.fit(x,y)
prediction = model.predict([[10,11]])
print("prediction=",prediction)



import pandas as pd

data={"Name":["Mamun","Rahim","Karim","Jamal"],"Marks":[80,45,70,30]}

df=pd.DataFrame(data)
print(df)

print("Average=",df["Marks"].mean())
print("Highest=",df["Marks"].max())
print("Lowest=",df["Marks"].min())

passed=df[df["Marks"]>=50]
print(passed)


import pandas as pd
import matplotlib.pyplot as plt
data={"Name":["Mamun","Rahim","Karim","Jamal"],"Marks":[90,59,70,43]}
df=pd.DataFrame(data)
plt.bar(df["Name"],df["Marks"])
plt.title("Student Marks")
plt.xlabel("Students")
plt.ylabel("Maerks")

plt.show()

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
#House size (square feet)
x=np.array([[1000],[1500],[2000],[2500],[3000]])
#House price
y=np.array([100000,150000,200000,250000,300000])
model=LinearRegression()
model.fit(x,y)
plt.scatter(x,y)
plt.title("House Price")
plt.ylabel("Price")
plt.xlabel("Size")
price=model.predict([[3500]])
print("Predicted Price=",price[0])
plt.show()


import torch
x=torch.tensor([1,2,3,4,5])
print(x)

import torch
a=torch.tensor([1,2,3])
b=torch.tensor([4,5,6])
print(a+b)

import torch
import torch.nn as nn

#Neural Network তৈরি
model=nn.Sequential(nn.Linear(1,10),nn.ReLU(),nn.Linear(10,1))
print(model)


import torch
x=torch.tensor([[5.0]])
print(model(x))


import torch
import torch.nn as nn
import torch.optim as optim

#Training Data
x=torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y=torch.tensor([[2.0],[4.0],[6.0],[8.0]])

#Model
model = nn.Sequential(nn.Linear(1,10),nn.ReLU(),nn.Linear(10,1))

#Loss Function
loss_fn = nn.MSELoss()

#Optimizer
optimizer = optim.Adam(model.parameters(),lr=0.01)

#Training
for epoch in range(500):
    prediction = model(x)
    loss = loss_fn(prediction,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training Complete!")

#Test
test = torch.tensor([[5.0]])
print("Prediction:",model(test).item())


