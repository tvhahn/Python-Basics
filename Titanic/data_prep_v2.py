import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as s
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, accuracy_score

# df_train = pd.read_csv('train.csv')
# df_test = pd.read_csv('test.csv')

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    # can find mean using: train[train['Pclass'] == 1]['Age'].dropna().mean()
    if pd.isnull(Age):
        if Pclass == 1:
            return 38.12347
        elif Pclass == 2:
            return 29.8776
        else:
            return 25.14061
    else:
        return Age

def impute_fare(cols):
    Fare = cols[0]
    Pclass = cols[1]

    # can find mean using: train[train['Pclass'] == 1]['Fare'].dropna().mean()
    if pd.isnull(Fare):
        if Pclass == 1:
            return 84.1935
        elif Pclass == 2:
            return 20.662
        else:
            return 13.6756
    else:
        return Fare

def prep_train(df):
    df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)
    df['Fare'] = df[['Fare','Pclass']].apply(impute_fare,axis=1)
    df.drop('Cabin',axis=1,inplace=True)
    df.dropna(inplace=True)
    sex = pd.get_dummies(df['Sex'],drop_first=True)
    embark = pd.get_dummies(df['Embarked'],drop_first=True)
    p_class = pd.get_dummies(df['Pclass'],drop_first=True)
    df = pd.concat([df, p_class, sex, embark],axis=1)
    df.drop(['Sex','Embarked','Name','Ticket','Pclass'],axis=1,inplace=True)
    df.to_pickle("train2.pkl")
    return(df)


def prep_test(df):
    # TEST data-set
    df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)
    df['Fare'] = df[['Fare','Pclass']].apply(impute_fare,axis=1)
    df.drop('Cabin',axis=1,inplace=True)
    sex = pd.get_dummies(df['Sex'],drop_first=True)
    embark = pd.get_dummies(df['Embarked'],drop_first=True)
    p_class = pd.get_dummies(df['Pclass'],drop_first=True)
    df = pd.concat([df, p_class, sex, embark],axis=1)
    df.drop(['Sex','Embarked','Name','Ticket','Pclass'],axis=1,inplace=True)
    df.to_pickle("test2.pkl")
    return(df)

# train = prep_train(df_train)
# test = prep_test(df_test)

train = pd.read_pickle("train2.pkl")
test = pd.read_pickle("test2.pkl")
test_pid = pd.DataFrame(test['PassengerId']) # create a new DF for the passenger ID column
test = test.drop('PassengerId', axis=1) # drop the passenger ID column from the test data-set

# train-test-split
X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived','PassengerId'],axis=1),
                                                    train['Survived'], test_size=0.33)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
pred = logmodel.predict(X_test)
# print("Accuracy", accuracy_score(y_test, pred))

# function to test the model over N number of iterations
def test_model():
    error_rate = []
    for i in range(1,100):
        X_train1, X_test1, y_train1, y_test1 = train_test_split(train.drop(['Survived','PassengerId'],axis=1), train['Survived'], test_size=0.33)
        logmodel_test = LogisticRegression()
        logmodel_test.fit(X_train1,y_train1)
        p = logmodel_test.predict(X_test1)
        error_rate.append(np.mean(p != y_test1))
        i += 1

    print("Error Rate:", round(s.mean(error_rate),4))
    print("Accuracy:", round(1-s.mean(error_rate),4))

# logmodel = LogisticRegression()
# logmodel.fit(X_train,y_train)

test_model()

# prepare submission for kaggle
p_submit = logmodel.predict(test)
logreg_pred = pd.DataFrame()
logreg_pred['PassengerId'] = test_pid['PassengerId']
logreg_pred['Survived'] = p_submit

# output the predictions to a csv, ready to import to Kaggle
logreg_pred.to_csv('logreg_pred.csv',index=False)