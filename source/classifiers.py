from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from utils import toNumpyArray
from typing import Literal

def classify_data(X_train, y_train, X_test, method: Literal['naive-bayes', 'random-forest'] | None):
    if method == 'naive-bayes':
        return applyNaiveBayes(X_train, y_train, X_test)
    elif method == 'random-forest':
        return applyRandomForest(X_train, y_train, X_test)
    elif method == 'xgboost':
        return applyXgboost(X_train, y_train, X_test)
    else:
        raise Exception("Unknown classifier")
    

# You may add more classifier methods replicating this function
def applyNaiveBayes(X_train, y_train, X_test):
    '''
    Task: Given some features train a Naive Bayes classifier
          and return its predictions over a test set
    Input; X_train -> Train features
           y_train -> Train_labels
           X_test -> Test features 
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = MultinomialNB()
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict



def applyRandomForest(X_train, y_train, X_test): ## Improves 3 points of F1
    '''
    Task: Given some features train a Random Forest Classifier
          and return its predictions over a test set
    Input; X_train -> Train features
           y_train -> Train_labels
           X_test -> Test features 
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = RandomForestClassifier()
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict


def applyXgboost(X_train, y_train, X_test): 
    '''
    Task: Given some features train a Random Forest Classifier
          and return its predictions over a test set
    Input; X_train -> Train features
           y_train -> Train_labels
           X_test -> Test features 
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    label_mapping, reverse_mapping = _create_mappings(y_train)

    # Use the dictionary to convert string labels to numeric values
    y_numeric = [label_mapping[label] for label in y_train]
    
    clf = XGBClassifier()
    clf.fit(trainArray, y_numeric)
    y_predict = clf.predict(testArray)

    y_predict_string = [reverse_mapping[idx] for idx in y_predict]
    return y_predict_string


def _create_mappings(y_train):
    
    y_strings  = y_train.unique().tolist()
    label_mapping = {label: idx for idx, label in enumerate(set(y_strings))}
    reverse_mapping = {idx: label for label, idx in label_mapping.items()}

    return label_mapping, reverse_mapping
