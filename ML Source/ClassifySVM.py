from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def classifySVM(features_train, labels_train):
    clf = SVC(kernel="linear")
    scores = cross_val_score(clf, features_train, labels_train, cv=10)
    print("accuracySVM_CV",max(scores), sep=" : ")
    return clf.fit(features_train, labels_train)