from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def classifyLREG(features_train, labels_train):
    clf = LogisticRegression(solver='lbfgs', random_state=1)
    scores = cross_val_score(clf, features_train, labels_train, cv=10)
    print("accuracyLREG_CV",max(scores), sep=" : ")
    return clf.fit(features_train, labels_train)