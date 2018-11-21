''' Linear Regression   '''

from sklearn.linear_model import LinearRegression
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, mean_squared_error, r2_score 
from sklearn.metrics import confusion_matrix
import seaborn as sns

def split_train_test(df): 
    train, test = train_test_split(df, test_size=0.1)
    train_y, test_y = train["Hit_Flop_Class"].values ,test["Hit_Flop_Class"].values 
    train_x, test_x =  train.drop(labels="Hit_Flop_Class", axis=1).values , test.drop(labels="Hit_Flop_Class", axis=1).values
    return train_x, train_y , test_x, test_y


def fit_model(df):
    train_x, train_y , test_x, test_y = split_train_test(df)    # Get the split of data
    regr = LinearRegression()                                   # Linear Regression Model
    model = regr.fit(train_x, train_y)                          # Fit the model
    y_pred = model.predict(test_x)                              # Predict values
    mse = mean_squared_error(test_y, y_pred)                    # The mean squared error
    print("Mean squared error: %.2f"% mse)   

df = pd.read_csv("../data/msd_feature_selection_normalised.csv")
# print(df.shape)
fit_model(df)