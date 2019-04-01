# Detection-Of-Parkinson-s-Disesase-Using-Voice-Impairments-With-ML-and-LSTM

Introduction

The purpose of this project is to make up a prediction model where we will be able to predict whether a patient is suffering from Parkinson's disease or not. 
In this project, we use various Machine Learning models to find their respective accuracies, then implement the proposed Deep Learning LSTM model 
that accurately predicts Parkinson's and gives better results, based on speech data.

To do so, we will work on UCI's Machine Learning repository speech dataset. We perform Feature Extraction, then Data Cleaning on the dataset. After obtaining the dataset in
the desired format, we load it into various predictive models including Logistic Regression, K_NN, Naive Bayes, SVM, and RFC. We then implement the proposed LSTM model.

Dataset

Dataset is downloaded from UCI's Machine Learning repository speech dataset. We clean the dataset to get the required format in the CSV files. 
The formatted dataset is stored in the file "Combined Data.xlsx".

Installation

The application has two parts. 
a) The first part is the comparison of various predictive models including Logistic Regression, K_NN, Naive Bayes, SVM, and RFC.
b) The second part consists of LSTM based classification.

Part a)

The application is run using the Python Interpreter. 

Python Environment

The project requires Python 3.6 interpreter.

To use the Python interpreter to run the project, first install the python packages being used in this project.

pip install numpy
pip install pylab
pip install pandas
pip install sklearn
pip install scikitplot

Following open source code has been used in the project files:
1. sklearn
2. numpy
3. pandas
4. pylab
5. scikitplot

To run the application:
python Patient_Main.py

Part b)

The application is run in MATLAB.

MATLAB Environment

Download MATLAB with DeepLearning toolkit. 
Keep the dataset and LSTM.m file in the same folder. 
Open the directory in MATLAB and click on Run.
If folder path is not specified during runtime, add current folder path to the system.
