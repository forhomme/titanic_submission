import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

# This is how you would load the model again
model = pickle.load(open('model.sav', 'rb'))
stack = pickle.load(open('stack.sav', 'rb'))
poly_lr = pickle.load(open('poly_lr.sav', 'rb'))

# Load data
test = pd.read_csv('test_final.csv', index_col=0)
pd.set_option('mode.chained_assignment', None)

X_test = test.drop('PassengerId', axis=1)
S_test = stack.transform(X_test)

y_test = model.predict(S_test)
test['Survived'] = y_test
submission = test[['PassengerId', 'Survived']]
submission.to_csv('titanic_submission.csv', index=False)
