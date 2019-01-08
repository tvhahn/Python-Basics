# I wanted to calculate the confusion matrix myself, so I found this on a stack-overflow type website. The hardest part was realizing that the y_test data is actually a pandas dataframe.

# from: https://datascience.stackexchange.com/questions/28493/confusion-matrix-get-items-fp-fn-tp-tn-python

# X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), train['Survived'], test_size=0.33)
# logmodel = LogisticRegression()
# logmodel.fit(X_train,y_train)
# predictions = logmodel.predict(X_test)

def perf_measure(y_actual, y_pred):
    y_actual = y_actual.tolist() # y_actual is a pandas dataframe
    y_pred = np.ndarray.tolist(y_pred)

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    i = 0
    l = len(y_pred)
    while i < l:
        if y_actual[i]==y_pred[i] and y_pred[i]==1:
            TP += 1
        if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
            FP += 1
        if y_actual[i]==y_pred[i] and y_pred[i]==0:
            TN += 1
        if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
            FN += 1
        i += 1

        metrics = [TP, FP, TN, FN]
        
    return(metrics)
