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