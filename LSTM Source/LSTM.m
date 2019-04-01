ilename = "parkinson_speech_dataset.csv";
data = readtable(filename,'TextType','string');
idxEmpty = strlength(data.class_status) == 0;
data(idxEmpty,:) = [];
data.class_status = categorical(data.class_status);
f = figure;
f.Position(3) = 1.5*f.Position(3);

h = histogram(data.class_status);
xlabel("Class")
ylabel("Frequency")
title("Class Distribution")

classCounts = h.BinCounts;
classNames = h.Categories;
idxLowCounts = classCounts < 10;
infrequentClasses = classNames(idxLowCounts);
idxInfrequent = ismember(data.class_status,infrequentClasses);
data(idxInfrequent,:) = [];
data.class_status = removecats(data.class_status);
cvp = cvpartition(data.class_status,'Holdout',0.3);
dataTrain = data(training(cvp),:);
dataHeldOut = data(test(cvp),:);
cvp = cvpartition(dataHeldOut.class_status,'HoldOut',0.5);
dataValidation = dataHeldOut(training(cvp),:);
dataTest = dataHeldOut(test(cvp),:);
textDataTrain = dataTrain.class_status;
textDataValidation = dataValidation.class_status;
textDataTest = dataTest.class_status;
YTrain = dataTrain.class_status;
YValidation = dataValidation.class_status;
YTest = dataTest.class_status;

textDataTrain = lower(textDataTrain);
documentsTrain = tokenizedDocument(textDataTrain);
documentsTrain = erasePunctuation(documentsTrain);

textDataValidation = lower(textDataValidation);
documentsValidation = tokenizedDocument(textDataValidation);
documentsValidation = erasePunctuation(documentsValidation);
documentsTrain(1:5)
enc = wordEncoding(documentsTrain);

documentLengths = doclength(documentsTrain);
figure
histogram(documentLengths)
title("Shimmer and jitter")
xlabel("Shimmer")
ylabel("Jitter")

XTrain = doc2sequence(enc,documentsTrain,'Length',44);
XTrain(1:5)

XValidation = doc2sequence(enc,documentsValidation,'Length',44);
% --------------------------------------------------------------------

inputSize = 1;
embeddingDimension = 210;
numHiddenUnits = enc.NumWords;
hiddenSize = 140;
numClasses = numel(categories(YTrain));

layers = [ ...
    sequenceInputLayer(inputSize)
    wordEmbeddingLayer(embeddingDimension,numHiddenUnits)
    lstmLayer(hiddenSize,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',10, ...    
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'ValidationData',{XValidation,YValidation}, ...
    'Plots','training-progress', ...
    'Verbose',false);


net = trainNetwork(XTrain,YTrain,layers,options);

% --------------------------------------------------------------------


textDataTest = lower(textDataTest);
documentsTest = tokenizedDocument(textDataTest);
documentsTest = erasePunctuation(documentsTest);

XTest = doc2sequence(enc,documentsTest,'Length',44);
XTest(1:5)

YPred = classify(net,XTest);
accuracy = sum(YPred == YTest)/numel(YPred);