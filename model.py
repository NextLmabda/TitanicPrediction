import numpy as np
import pandas as pd

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('train.csv')

df.drop(['Cabin', 'Embarked', 'Ticket', 'Name', 'PassengerId'], axis = 1, inplace=True)
df.fillna(df['Age'].mean(), inplace = True)
df = pd.get_dummies(df)

df.to_csv('final_df')

y = df[['Survived']]
X = df.drop('Survived', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state = 1000, stratify= y)

model = LogisticRegression()
model.fit(X_train, y_train)

print(X_test.loc[0])

pred = model.predict(X_test)

print(accuracy_score(pred, y_test))

pickle.dump(model, open('model.pkl', 'wb'))

