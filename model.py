# This file contains the model which we deploy to a flask app

import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df_iris = pd.read_csv("iris.csv")
print(df_iris["Class"].unique())

x = df_iris[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
y = df_iris["Class"]

# Split train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=40)

# Scale features
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

classifier = RandomForestClassifier()

# Fit the model to the training data
classifier.fit(x_train, y_train)

# Dump to pickle file
pickle.dump(classifier, open("model.pkl", "wb"))
