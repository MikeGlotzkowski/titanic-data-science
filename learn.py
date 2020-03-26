import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBClassifier, plot_tree
import matplotlib.pyplot as plt


titanic_data = pd.read_csv("./data/titanic.csv")
labelEncoder = LabelEncoder()
titanic_data['Sex'] = labelEncoder.fit_transform(titanic_data['Sex'])

# first try without name
titanic_data.drop(['Name'], axis=1, inplace=True)

# random split
msk = np.random.rand(len(titanic_data)) < 0.8
X = titanic_data[msk]
y = X.pop('Survived')

model = XGBClassifier()
kfold = KFold(n_splits=10)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(scores)

model.fit(X, y)
p
# plot single tree
plot_tree(model)
plt.show()
