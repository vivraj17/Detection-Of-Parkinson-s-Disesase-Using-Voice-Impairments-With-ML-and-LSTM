from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def classifyK_NN(features_train, labels_train):
    clf = KNeighborsClassifier()
    scores = cross_val_score(clf, features_train, labels_train, cv=10)
    print("accuracyk-NN_CV",max(scores), sep=" : ")
    return clf.fit(features_train, labels_train)