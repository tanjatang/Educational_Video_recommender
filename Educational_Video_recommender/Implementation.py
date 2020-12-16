import pandas as pd
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# %% load the file
df = pd.read_csv("/dataset/tel_youtube_content_train(original).csv", error_bad_lines=False)
df_test = pd.read_csv("/dataset/tel_youtube_content_evaluation_students(original).csv")

#Clean the dataset by remove NaN value or fill 0
df=df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
df_test["external_rating"] = df_test["external_rating"].fillna(0)
df_test["viewcount"] = df_test["viewcount"].fillna(0)


# %% add some features
df["external_rating_viewcount"] = (df["external_rating"]*df["viewcount"])/(df["external_rating"]+0.9*df["viewcount"])
df_test["external_rating_viewcount"] = (df_test["external_rating"]*df_test["viewcount"])/(df_test["external_rating"]+0.9*df_test["viewcount"])

X = df.loc[:,["external_rating_viewcount"]]
Y = df.loc[:,'decision']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
X_sub = df_test.loc[:,["external_rating_viewcount"]]
# create a voting scheme
voting_clf = VotingClassifier(estimators=[
    ('Random_forest',RandomForestClassifier(n_estimators=50, random_state=0)),
    ("LDA",LinearDiscriminantAnalysis()),
    ('log_clf', LogisticRegression()),
    ('svm_clf', SVC(probability=True)),
    ('dt_clf', DecisionTreeClassifier(random_state=100))
], voting='soft')
# fit the classifier
voting_clf.fit(X_train, y_train)
#predict the result
y_pred = voting_clf.predict(X_test)
y_sub = voting_clf.predict(X_sub)
# save predicted data in csv file
df_test['decision'] = y_sub
df_test.to_csv("/results/group3_4.csv",index=False,sep=',')

print("accuracy_score",accuracy_score(y_test, y_pred))
print("F1 score",f1_score(y_test, y_pred, average='weighted'))