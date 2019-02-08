import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

df = pd.read_csv('clean_data.csv')

import random
index_insincere_q = np.array(df[df['target'] == 1].index) # len = 80810
index_sincere_q = np.array(df[df['target'] == 0].index)
index_sincere_q_reduc = random.sample(list(index_sincere_q), int(len(index_insincere_q)))

X = pd.concat([df['treated_question'][index_insincere_q], df['treated_question'][index_sincere_q_reduc]])
y = pd.concat([df['target'][index_insincere_q], df['target'][index_sincere_q_reduc]])

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=.1, random_state=2018)
print(df_train.shape, df_test.shape)

print("Split done")

print(df_train.head())
import random
index_insincere_q = np.array(df_train[df_train['target'] == 1].index) # len = 80810
index_sincere_q = np.array(df_train[df_train['target'] == 0].index)
index_sincere_q_reduc = random.sample(list(index_sincere_q), int(len(index_insincere_q)))

X = pd.concat([df_train['treated_question'][index_insincere_q], df_train['treated_question'][index_sincere_q_reduc]])
y = pd.concat([df_train['target'][index_insincere_q], df_train['target'][index_sincere_q_reduc]])

print(X.head())


from sklearn.pipeline import Pipeline

svd = TruncatedSVD(n_components=300, random_state=42)
tfvec = TfidfVectorizer(ngram_range=(1,3),stop_words='english', lowercase=False)

preprocessing_pipe = Pipeline([
    ('vectorizer', tfvec),
    ('svd', svd),
])

train_X = preprocessing_pipe.fit_transform(X.values)
test_X = preprocessing_pipe.transform(df_test['treated_question'].values)

np.save('trainX.npy',train_X)
np.save('testX.npy',test_X)

train_X = np.load('trainX.npy')
test_X = np.load('testX.npy')

print("Transformation done.")

train_Y = y.values
test_Y = df_test['target'].values

tuned_parameters = [{'learning_rate': [0.1], 'n_estimators': [10,50,100], 'booster': ["gbtree", "gblinear","dart"],
                          'seed': [42], 'max_depth': [3,5,10]}]
xgb = XGBClassifier()


from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, roc_curve



tuned_parameters = ParameterGrid(tuned_parameters)

for parameters in tuned_parameters:
    print(parameters)
    xgb = XGBClassifier(**parameters)
    xgb.fit(train_X,train_Y)
    preds_test = xgb.predict(train_X)
    preds = xgb.predict(test_X)
    print(f1_score(train_Y, preds_test, average='weighted'),accuracy_score(train_Y, preds_test),
          f1_score(test_Y, preds, average='weighted'),accuracy_score(test_Y, preds))
    print(roc_curve(train_Y, preds_test),roc_curve(test_Y, preds))
    print()


