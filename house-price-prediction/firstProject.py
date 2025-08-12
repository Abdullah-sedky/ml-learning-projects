import numpy as np # for math stuff, DS and ML, simulations, Image and signal processing, faster (written in c)
import pandas as pd #for plotting
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
myDataRead=pd.read_csv("files/data.csv")
myDataRead.head()
print(myDataRead.describe())
sns.pairplot(myDataRead) #to show relationships as graphs and predict if linear regression will work before using model, if points are in straight line, then maybe linear regression will be good
plt.show()
print(myDataRead.isnull().sum()) # Check for missing values. isnull creates similar table of bools, and sum returns sum of trues (number of missing items/Nan)

# Correlation matrix to understand relationships between each 2 features
#correlation is always between 1>=0>=-1
# if 1>=correlation>0 correlation is positive, both features increase together
# if 0>correlation>=-1 correlation is negative, both features decrease together
#if correlation=0, no relationship
#Strong: |r| ≥ 0.7, Moderate: 0.4 ≤ |r| < 0.7, weak: 0.1 ≤ |r| < 0.4, none: |r| < 0.1, |r| for correlation
correlation_matrix = myDataRead.select_dtypes(include=['number']).corr() #to only include numeric values
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm') #annot=True->writes value of correlation on each square. coolwarm for colors, blue for negative, red for positive
plt.title("Correlation Matrix")
plt.show()

x = myDataRead[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition']] #inputs/features
y = myDataRead['price'] #output

#Splitting the dataset into training and testing sets, testing 0.2 or 20%, so training is 80%. random state is constant number so same rows are tested everytime
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=111)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#applying model
model=LinearRegression()
model.fit(x_train, y_train)
#test
y_pred = model.predict(x_test)
print(str(model.score(x_test, y_test)))
print(model.coef_)
print(model.intercept_)

#plot to check how far off I was, it's better when most points are close to 0
residuals = y_test - y_pred
plt.scatter(y_test, residuals) #a way of visualization
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Actual Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


