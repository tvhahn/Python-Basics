import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, accuracy_score

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

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
    df = pd.concat([df, sex, embark],axis=1)
    df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
    df.to_pickle("train2.pkl")
    return(df)

prep_train(df_train)

def prep_test(df):
    # TEST data-set
    df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)
    df['Fare'] = df[['Fare','Pclass']].apply(impute_fare,axis=1)
    df.drop('Cabin',axis=1,inplace=True)
    sex = pd.get_dummies(df['Sex'],drop_first=True)
    embark = pd.get_dummies(df['Embarked'],drop_first=True)
    df = pd.concat([df, sex, embark],axis=1)
    df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
    df.to_pickle("test2.pkl")
    return(df)


# # load pickle for basic training dataset
# train = pd.read_pickle("train1.pkl")
# # create a data-frame with the passengerid
# train_pid = pd.DataFrame(train['PassengerId'])
# # load pickle for basic testing dataset
# test = pd.read_pickle("test1.pkl")
# # create a data-frame with the passengerid
# test_pid = pd.DataFrame(test['PassengerId'])