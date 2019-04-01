from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def classifyRFC(features_train, labels_train):
    clf = RandomForestClassifier(n_estimators=100)
    scores = cross_val_score(clf, features_train, labels_train, cv=10)
    print("accuracyRFC_CV",max(scores), sep=" : ")
    return clf.fit(features_train, labels_train)