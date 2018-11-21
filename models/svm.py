''' Support Vector Machine  '''
from sklearn.svm import SVC
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import seaborn as sns

def split_train_test(df): 
    train, test = train_test_split(df, test_size=0.5)
    train_y, test_y = train["Hit_Flop_Class"].values ,test["Hit_Flop_Class"].values 
    train_x, test_x =  train.drop(labels="Hit_Flop_Class", axis=1).values , test.drop(labels="Hit_Flop_Class", axis=1).values
    return train_x, train_y , test_x, test_y


def fit_model(df):
    train_x, train_y , test_x, test_y = split_train_test(df)    # Get the split of data
    model = 'rbf'
    if(model == 'rbf'):
        model = SVC(gamma='auto')                               # Gaussian SVC with auto gamma selection
    else:
        model = SVC(kernel='linear')                            # Default: Linear SVC
    model = model.fit(train_x, train_y)                         # Fit the model
    y_pred = model.predict(test_x)                              # Predict values
    accuracy = accuracy_score(test_y, y_pred)                   # Check the accuracy
    print("Accuracy of model: ",accuracy)
    
df = pd.read_csv("../data/cleaned_msd.csv")
fit_model(df)