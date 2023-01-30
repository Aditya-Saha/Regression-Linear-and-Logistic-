import pandas as pd
x=[]
y=[]

for i in range(1,50):
   x.append(i)
   y.append((i*i*i-1))

df=pd.DataFrame()

df["X"]=x
df["Y"]=y

#print(df)

df.to_csv("Lin_Reg_Data_1.csv", header=True, index=False)#storing the two columns in the csv file

print("File succesfully saved....")

df1= pd.read_csv("Lin_Reg_Data_1.csv")


import numpy as np
from sklearn.linear_model import LogisticRegression

x=np.array(df1['X']).reshape(-1,1)
y= np.array(df1['Y'])

model=LogisticRegression(solver='liblinear', max_iter=100, multi_class='ovr')

model.fit(x,y)

print(f"The coefficient of determination : {model.score(x,y)}")

print(f"intercept= {model.intercept_}")

print(f"slope= {model.coef_}")

from sklearn.metrics import mean_squared_error

y_pred=model.predict(x)
y_true=y
mse=mean_squared_error(y_true,y_pred)#calculating mean square error

print("Mean squared error is: ",mse) 

#predicting value for new model
x_new=np.array([51,52,53,54,55]).reshape(-1,1)
y_new=model.predict(x_new)
print(y_new)
