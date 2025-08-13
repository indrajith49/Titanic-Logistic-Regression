# Titanic-Logistic-Regression
hellooo today i just finished making a ML model [Titanic] (altho absolutely noob), but heyy its my first project hehehehe. Till i have learned - Basics of logistic regression   - Handling missing data - Encoding categorical variables - Train/test split and model evaluation - Using Python libraries like Pandas, NumPy, Seaborn, and Scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import seaborn as sns

df = pd.read_csv("C:/Users/indra/Downloads/titanic.csv")
print(df.info())
print(df.describe())



df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis = 1, inplace = True)

df['Sex'] = df['Sex'].map({'male': 0, 'female' : 1 })

df['Embarked'] = df['Embarked'].map({'S' : 0, 'C':1, 'Q':2 })

X = df.drop('Survived', axis = 1)
Y = df['Survived']

X_train,X_test, y_train, y_test = train_test_split(X,Y,test_size= 0.2, random_state = 42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix : ")
print(confusion_matrix(y_test, y_pred))
print("Classification Report: ")
print(classification_report(y_test, y_pred))
