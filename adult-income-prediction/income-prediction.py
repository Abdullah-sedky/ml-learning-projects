import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import log_loss, roc_auc_score, confusion_matrix, f1_score

myFile=pd.read_csv("data/adult_income.csv")
print(myFile.describe())
myFile.head()
sns.pairplot(myFile)
plt.show()
correlation_matrix=myFile.select_dtypes(include=['number']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("correlation map")
plt.show()

unique_edu = myFile['education'].unique()  #to print all unique categories in education column
print(unique_edu)

#to replace categoric using label encoding after arranging them
education_order = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-acdm','Assoc-voc','Bachelors', 'Masters', 'Prof-school', 'Doctorate']
ordinal=OrdinalEncoder(categories=[education_order])
myFile['education']=ordinal.fit_transform(myFile[['education']])
print(myFile.head())