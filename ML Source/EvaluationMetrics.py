def Accuracy(Prediction, TestData_Model):
    TP = 0
    TN = 0
    for i,j in zip(Prediction,TestData_Model):
        if i==j and i == 1:
            TP += 1
        elif i==j and i == 0:
            TN += 1
    print("Detected Ones", TP, sep=" : ")
    print("Detected Zeros", TN, sep=" : ")    
    return (TP+TN)/(len(Prediction))