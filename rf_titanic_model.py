import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn import metrics
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import VotingClassifier
from vecstack import stacking, StackingTransformer

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('train_final.csv', index_col=0)
X = data.drop(['PassengerId', 'Survived'], axis=1)
y = data['Survived']

# We will use cross-validation on the training set to tune parameters, then test on unseen data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, stratify=y, random_state=42)

# Put in our parameters for said classifiers
# Logistic Regression parameters
lr_params = {
    'C': 0.05
}
# KNN parameters
knn_params = {
    'n_neighbors': 5,
    'leaf_size': 2,
    'weights': 'uniform',
    'algorithm': 'auto'
}

# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'warm_start': True,
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'verbose': 0
}

# Decision Tree parameters
dt_params = {
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
}
# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate': 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}
# XGBoost parameters
xgboost_paramas = {
    'eta': 0.1,
    'max_depth': 5,
    'n_estimators': 500,
    'gamma': 0.9,
    'subsample': 0.8,
    'nthread': -1
}
# Support Vector Classifier parameters
svc_params = {
    'kernel': 'linear',
    'C': 0.025
    }

svc_params2 = {
    'kernel': 'rbf',
    'C': 0.05
}

# Fit base model to use for comparison
SEED = 0  # for reproducibility
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
knn = KNeighborsClassifier(n_jobs=-1, **knn_params)
nb = GaussianNB()
lr = LogisticRegression(**lr_params)
rf = RandomForestClassifier(random_state=SEED, **rf_params)
et = ExtraTreesClassifier(random_state=SEED, **et_params)
dt = DecisionTreeClassifier(random_state=SEED, **dt_params)
svc = SVC(random_state=SEED, **svc_params, probability=True)
rbf = SVC(random_state=SEED, **svc_params2, probability=True)

estimators = [('knn', knn), ('nb', nb), ('lr', lr), ('rf', rf), ('et', et), ('dt', dt), ('svc', svc),
              ('rbf', rbf)]
# voting
voting_models = VotingClassifier(estimators=estimators, voting='soft')

# bagging
bag_knn = BaggingClassifier(base_estimator=knn, random_state=SEED, n_estimators=500)
bag_dt = BaggingClassifier(base_estimator=dt, random_state=SEED, n_estimators=500)

# boosting
ada = AdaBoostClassifier(random_state=SEED, **ada_params)
gb = GradientBoostingClassifier(random_state=SEED, **gb_params)
xgboost = xgb.XGBClassifier(random_state=SEED, **xgboost_paramas)

# define estimator for stacking transformer
estimator = [('voting', voting_models), ('bag_knn', bag_knn), ('bag_dt', bag_dt), ('ada', ada), ('gb', gb),
             ('xgboost', xgboost)]
stack = StackingTransformer(estimator,
                            regression=False,
                            needs_proba=False,
                            variant='A',
                            metric=metrics.accuracy_score,
                            n_folds=4,
                            stratified=True,
                            shuffle=True,
                            random_state=0,
                            verbose=2)
stack = stack.fit(X_train, y_train)
filename = 'stack.sav'
pickle.dump(stack, open(filename, 'wb'))

# stacked feature
S_train = stack.transform(X_train)
S_test = stack.transform(X_test)

# Create the parameter grid
params = {
    'eta': [0.1, 0.2, 0.3, 0.4, 0.5],
    'max_depth': [4, 5, 6, 7],
    'n_estimators': [500, 1000, 1500, 2000]
}

gbm = xgb.XGBClassifier(gamma=0.9, subsample=0.8, colsample_bytree=0.8,
                        objective='binary:logistic', nthread=-1, scale_pos_weight=1)
# Create the grid search model
gs = GridSearchCV(estimator=gbm, param_grid=params, cv=cv, verbose=0,
                  scoring='brier_score_loss', return_train_score=True)

# Fit gs
gs.fit(S_train, y_train)
best_rf = gs.best_estimator_

# Save the best model to disk so you can use it again whenever you like (e.g. in another notebook etc)
filename = 'model.sav'
pickle.dump(best_rf, open(filename, 'wb'))

scores = cross_val_score(best_rf, X_train, y_train, scoring='brier_score_loss')
print('Brier loss:', "{0:.5f}".format(np.mean(scores)*-1))

# Create predictions
y_pred = best_rf.predict(S_test)
y_pred_prob = best_rf.predict_proba(S_test)

# Print results
print("Predicted survivor (test set):", sum(y_pred))
print("Sum of predicted survivor probabilities (aka xG):", "{0:.2f}".format(sum(y_pred_prob[:, 1])))
print("Actual survivor (test set):", sum(y_test))
print('')
print(metrics.classification_report(y_test, y_pred))

# Plot results
skplt.metrics.plot_confusion_matrix(y_test, y_pred)

skplt.metrics.plot_precision_recall(y_test, y_pred_prob)

# Get feature importance
# importances = pd.DataFrame({'feature': X.columns, 'importance': np.round(stack.feature_importances_, 3)})
# importances = importances.sort_values('importance', ascending=False)

# f, ax = plt.subplots(figsize=(6, 8))
# g = sns.barplot(x='importance', y='feature', data=importances,
                # color="blue", saturation=.2, label="Total")
# g.set(xlabel='Feature Importance', ylabel='Feature', title='Feature Importance in Predicting xG')
plt.show()