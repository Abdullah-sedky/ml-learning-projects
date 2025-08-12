import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, confusion_matrix, f1_score
from matplotlib import pyplot as plt
#binary classification using logistic regression
heart_file=pd.read_csv("data/heart-2.csv")
heart_file.describe()
heart_file.head()

#correlations
correlation_matrix=heart_file.select_dtypes(include=['number']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("correlation map")
plt.show()



#dealing with empty values methods:
 #delete rows (if data is big and few rows wont affect accuracy)
 #delete columns (if most data in column are missing (>30%-50%))
 #imputation (using mean, mode or median)
 #knn or regression
 #domain knowledge
print(heart_file.isnull().sum())
heart_file.dropna(inplace=True)
(heart_file.isnull().sum())


all_columns=heart_file.columns.tolist()
input_columns=[col for col in all_columns if col!='target'] #include all columns except target
x = heart_file[input_columns].values #inputs/features
y = heart_file['target'].values #output

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
print(model.coef_)
print(model.intercept_)
#odds: ratio of (something happening : something not happening) = p/1-p
#prob is calculated using log(odds)
y_predic_proba=model.predict_proba(x_test) #probabilities of both 0 and 1
print(y_predic_proba)
y_predic=model.predict(x_test) #predictions of x tested, to compare with actual y tested
print(y_predic)

#Evaluation metrics between y_test and y_predicted:
acc=(y_test==y_predic).mean()
print("accuracy: " ,acc)

print("Log loss: ",log_loss(y_test, y_predic_proba)) #parameters from rule

print("ROC-AUC: ",roc_auc_score(y_test,y_predic_proba)) #???

print("confusion matrix: ",confusion_matrix(y_test, y_predic))

print("F1: ", f1_score(y_test, y_predic))