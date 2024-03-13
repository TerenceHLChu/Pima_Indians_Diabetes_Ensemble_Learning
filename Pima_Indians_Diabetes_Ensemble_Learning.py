import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV

np.random.seed(0)

# Set panda to display all of the dataframe's columns, without wrapping
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# -------------------
# LOAD AND CHECK DATA
# -------------------

column_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 
                'pedi', 'age', 'class']

# .csv file saved in the same directory as this script
path = 'pima-indians-diabetes.csv'

df = pd.read_csv(path, names=column_names)

df.info()

print('\n-----First 3 rows the dataframe-----\n', df.head(3))

print('\n-----Number of missing values in each column-----\n', df.isna().sum())

print('\n-----Dataframe statistics-----\n', df.describe())

print('\n-----Instances of each class-----\n', df['class'].value_counts())

# -----------------------------------------------------
# PRE-PROCESS AND PREPARE THE DATA FOR MACHINE LEARNING
# -----------------------------------------------------

transformer = StandardScaler()

X = df.drop(['class'], axis='columns') # Features
y = df['class'] # Class

# Check that X contains all features
print('\n', X.columns)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=42)
    
X_train = transformer.fit_transform(X_train)
X_test = transformer.fit_transform(X_test)

# ------------------------
# EXERCISE #1: HARD VOTING
# ------------------------

lr_clf_C = LogisticRegression(max_iter=1400)
rf_clf_C = RandomForestClassifier()
svc_clf_C = svm.SVC()
dt_clf_C = DecisionTreeClassifier(criterion='entropy', max_depth=42)
xt_clf_C = ExtraTreesClassifier()

voting_clf = VotingClassifier(
                estimators=[
                    ('lr', lr_clf_C),
                    ('rf', rf_clf_C),
                    ('svc', svc_clf_C),
                    ('dt', dt_clf_C),
                    ('xt', xt_clf_C)
                ], 
                voting='hard') # voting='hard' by default

voting_clf.fit(X_train, y_train)

print('\n-----Actual classes of first 3 instances-----\n', y_test[:3])

print('\n-----Prediction of first 3 instances by the hard voting classifier-----\n', 
      voting_clf.predict(X_test[:3]))

for name, clf in voting_clf.named_estimators_.items():
    print(f'\n-----Prediction of first 3 instances by {name}-----\n', 
          clf.predict(X_test[:3]))
    
# ------------------------    
# EXERCISE #2: SOFT VOTING
# ------------------------

# s_ = soft voting
# SVC needs to include the parameter, probability=True
# All other classifiers remain unchanged
s_lr_clf_C = LogisticRegression(max_iter=1400)
s_rf_clf_C = RandomForestClassifier()
s_svc_clf_C = svm.SVC(probability=True)
s_dt_clf_C = DecisionTreeClassifier(criterion='entropy', max_depth=42)
s_xt_clf_C = ExtraTreesClassifier()

soft_voting_clf = VotingClassifier(
                estimators=[
                    ('lr', s_lr_clf_C),
                    ('rf', s_rf_clf_C),
                    ('svc', s_svc_clf_C),
                    ('dt', s_dt_clf_C),
                    ('xt', s_xt_clf_C)
                ], 
                voting='soft') 

soft_voting_clf.fit(X_train, y_train)

print('\n-----Actual classes of first 3 instances-----\n', y_test[:3])

print('\n-----Prediction of first 3 instances by the soft voting classifier-----\n', 
      soft_voting_clf.predict(X_test[:3]))

for name, clf in soft_voting_clf.named_estimators_.items():
    print(f'\n-----Prediction of first 3 instances by {name}-----\n', 
          clf.predict(X_test[:3]))
    print(f'\n-----.predict_proba of first 3 instances by {name}-----\n', 
          clf.predict_proba(X_test[:3]))

# -----------------------------------------
# EXERCISE #3: RANDOM FORESTS & EXTRA TREES
# -----------------------------------------

pipeline1 = Pipeline([
                                ('std_scalar', transformer),
                                ('classifier', xt_clf_C)
                            ])

pipeline2 = Pipeline([
                                ('std_scalar', transformer),
                                ('classifier', dt_clf_C)
                            ])

# pl1/pl2 = pipeline1/pipeline2
pl1_clf = pipeline1.fit(X_train, y_train)
pl2_clf = pipeline2.fit(X_train, y_train)

stratified_k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

scores1 = cross_val_score(pl1_clf, X_train, y_train, 
                          cv=stratified_k_fold, scoring='accuracy')
print('\n-----Pipeline 1 scores-----\n', scores1)
print('\n-----Pipeline 1 mean scores-----\n', scores1.mean())

scores2 = cross_val_score(pl2_clf, X_train, y_train, 
                          cv=stratified_k_fold, scoring='accuracy')
print('\n-----Pipeline 2 scores-----\n', scores2)
print('\n-----Pipeline 2 mean scores-----\n', scores2.mean())

for clf in pl1_clf, pl2_clf:
    y_prediction = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_prediction)
    cm_visual = ConfusionMatrixDisplay(cm)
    cm_visual.plot()
    plt.show()
    print(f'\n-----Classification report for {clf.named_steps.values()}-----\n',
          classification_report(y_test, y_prediction))
  
# ----------------------------------------
# EXERCISE #4: EXTRA TREES AND GRID SEARCH
# ----------------------------------------

parameters = {
                'classifier__n_estimators': range(10, 3000, 20),
                'classifier__max_depth': range(1, 1000, 2),
             }

randomized_search_cv = RandomizedSearchCV(
                                            estimator=pl1_clf,
                                            scoring='accuracy',
                                            param_distributions=parameters,
                                            cv=5,
                                            n_iter=40,
                                            refit=True,
                                            verbose=3,
                                            random_state=42
                                          )

randomized_search_cv.fit(X_train, y_train)

print('\n-----best_params_-----\n', randomized_search_cv.best_params_)
print('\n-----best_score_-----\n', randomized_search_cv.best_score_)
print('\n-----best_estimator_-----\n', randomized_search_cv.best_estimator_)

best_model = randomized_search_cv.best_estimator_

y_prediction_best_model = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_prediction_best_model)
cm_visual = ConfusionMatrixDisplay(cm)
cm_visual.plot()
plt.show()

print('\n-----Classification report for the best estimator-----\n',
      classification_report(y_test, y_prediction_best_model))