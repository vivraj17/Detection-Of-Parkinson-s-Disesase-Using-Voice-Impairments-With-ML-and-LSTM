from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

def classifyNB(features_train, labels_train):
    clf = GaussianNB()
    scores = cross_val_score(clf, features_train, labels_train, cv=10)
    print("accuracyNB_CV",max(scores), sep=" : ")
    return  clf.fit(features_train, labels_train)