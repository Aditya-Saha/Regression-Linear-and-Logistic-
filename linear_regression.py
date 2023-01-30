import pandas as pd
x=[]
y=[]

for i in range(1,101):
   x.append(i)
   y.append((2*i-5))

df=pd.DataFrame()

df["X"]=x
df["Y"]=y

#print(df)

df.to_csv("Lin_Reg_Data_1.csv",header=True,index=False)#storing the two columns in the csv file

print("File succesfully saved....")

df1= pd.read_csv("Lin_Reg_Data_1.csv")

import numpy as np
from sklearn.linear_model import LinearRegression

# the input (regressor,x) and output (resposonse, y)should be array or similar objects
#calling reshape on x since it should be 2-d or more precisely, it must have one column and as many rows as necessary. 

x=np.array(df['X']).reshape(-1,1)
y=np.array(df['Y'])
#print(x)

model=LinearRegression()

model.fit(x,y)#performing linear regression on the given data

r_sq=model.score(x,y) #finding coefficient of determination R SQAURE

print("The coeeficient of determination is ",r_sq)


print(f"intercept= {model.intercept_}")

print(f"slope= {model.coef_}")


from sklearn.metrics import mean_squared_error

y_pred=model.predict(x)
#print(y_pred)

from sklearn.metrics import r2_score

print(f"Thr r2_score is : {r2_score(y,y_pred)}")#another way of finding the coefficient of determination

y_true=y
mse=mean_squared_error(y_true,y_pred)#calculating mean square error

print(mse) 

#use the trained model to predict the value of y for x={101,102,103,104,105}
x_new=np.array([101,102,103,104,105]).reshape(-1,1)

y_new=model.predict(x_new)

print("The predicted output for the new dataset is:",y_new)














