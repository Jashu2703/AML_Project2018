'''  Regression Models   '''

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.svm import SVR
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score 

def split_train_test(df): 
    train, test = train_test_split(df, test_size=0.2)
    train_y, test_y = train["Hit_Flop_Class"].values ,test["Hit_Flop_Class"].values 
    train_x, test_x =  train.drop(labels="Hit_Flop_Class", axis=1).values , test.drop(labels="Hit_Flop_Class", axis=1).values
    return train_x, train_y , test_x, test_y


def fit_model(df, modelType):
    train_x, train_y , test_x, test_y = split_train_test(df)                # Get the split of data
    if(modelType == 'Lasso'):                                               # Lasso Model with cross validation
        regr = LassoCV(cv=50)                                               # cv => Number of folds
    elif(modelType == 'Ridge'):                                             # Ridge Model with cross validation
        regr = RidgeCV(cv=50)                                               # cv => Number of folds
    elif(modelType == 'LinearRegression'):                                  # Linear Regression Model
        regr = LinearRegression()
    elif(modelType == 'SVR'):                                               # Support Vector Regression Model
        regr = SVR() 
    model = regr.fit(train_x, train_y)                                      # Fit the model
    y_pred = model.predict(test_x)                                          # Predict values
    regr_mse = mean_squared_error(test_y, y_pred)                           # The mean squared error
    regr_r2_score =regr.score(test_x, test_y)
    return regr_mse, regr_r2_score

models = ['LinearRegression', 'Lasso', 'Ridge', 'SVR']
df = pd.read_csv("../data/msd_feature_selection_normalised.csv")
for model in models:
    mse, r2_score = fit_model(df,model)
    print("Model : %s"%model)
    print("Mean squared error: %.2f"% mse)   
    print("R2 Score: %.2f"%r2_score)
    print('--------------------------------------------------------')
