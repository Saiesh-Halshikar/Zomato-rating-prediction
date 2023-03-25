import numpy as np
import pandas as pd

#import warnings
#warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

df = pd.read_csv('zomato_df.csv')
df.drop('Unnamed: 0', axis = 1, inplace = True)

X = df.drop('rate', axis = 1)
y = df['rate']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

et_model = ExtraTreesRegressor(n_estimators = 100)
et_model.fit(x_train, y_train)
y_pred = et_model.predict(x_test)

import pickle
pickle.dump(et_model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print(y_pred)
