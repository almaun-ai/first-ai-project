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