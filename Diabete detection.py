# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as pr
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as mt
import tensorflow as tf

# %%
data = pd.read_csv("datasets/diabetes.csv")

# %%
print(f" Data Shape {data.shape}")

# %%
data.head(5)

# %%
data.hist(figsize=(14, 8))
plt.grid(False)
plt.show()

# %%
data.isnull().sum()

# %%
data.isna().sum()

# %%
data.describe().T

# %%
data.corr()

# %%
plt.figure(figsize=(12, 8))
sb.heatmap(
    data.corr(),
    cmap=plt.cm.coolwarm,
    annot=True,
    center=True,
    linewidths=1,
    linecolor="blue",
)
plt.show()

# %%
x, y = data.iloc[:, :-1], data.iloc[:, -1]

# %%
xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=42, stratify=y)

# %%
print(f" Xtrain shape {xtrain.shape}")
print(f" Xtest shape {xtest.shape}")
print(f" Ytrain shape {ytrain.shape}")
print(f" Ytest shape {ytest.shape}")

# %%
log_model = LogisticRegression(fit_intercept=True, max_iter=40000, verbose=0)

# %%
log_model.fit(xtrain, ytrain)

# %%
predicted = log_model.predict(xtest)

# %%
print(mt.classification_report(ytest, predicted))

# %%
print(f"Accuracy: {round(mt.accuracy_score(ytest,predicted)*100,2)}%")

# %%
scale = pr.StandardScaler()
xtrain_standardized = scale.fit_transform(xtrain)
xtest_standardized = scale.transform(xtest)

# %%
log_model_2 = LogisticRegression(fit_intercept=True, max_iter=40000, verbose=0)
log_model_2.fit(xtrain_standardized, ytrain)

# %%
predicted_2 = log_model_2.predict(xtest_standardized)

# %%
print(mt.classification_report(ytest, predicted_2))

# %%
print(f"Accuracy: {round(mt.accuracy_score(ytest,predicted_2)*100,2)}%")

# %%
print("Confusion matrix \n", mt.confusion_matrix(ytest, predicted_2))

# %%
