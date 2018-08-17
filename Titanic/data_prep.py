import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# TRAIN data-set
df_train['Age'] = df_train[['Age','Pclass']].apply(impute_age,axis=1)
df_train['Fare'] = df_train[['Fare','Pclass']].apply(impute_fare,axis=1)
df_train.drop('Cabin',axis=1,inplace=True)
df_train.dropna(inplace=True)
# print(len(df.index))
sex = pd.get_dummies(df_train['Sex'],drop_first=True)
embark = pd.get_dummies(df_train['Embarked'],drop_first=True)
df_train = pd.concat([df_train, sex, embark],axis=1)
df_train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
df_train.to_pickle("train1.pkl")

# TEST data-set
df_test['Age'] = df_test[['Age','Pclass']].apply(impute_age,axis=1)
df_test['Fare'] = df_test[['Fare','Pclass']].apply(impute_fare,axis=1)
df_test.drop('Cabin',axis=1,inplace=True)
print("No. of lines in submission:",len(df_test.index))
sex = pd.get_dummies(df_test['Sex'],drop_first=True)
embark = pd.get_dummies(df_test['Embarked'],drop_first=True)
df_test = pd.concat([df_test, sex, embark],axis=1)
df_test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
df_test.to_pickle("test1.pkl")

# end