from sklearn.linear_model import LogisticRegression
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import seaborn as sns

def split_train_test(df): 
    train, test = train_test_split(df, test_size=0.3)
    train_y, test_y = train["Hit_Flop_Class"].values ,test["Hit_Flop_Class"].values 
    train_x, test_x =  train.drop(labels="Hit_Flop_Class", axis=1).values , test.drop(labels="Hit_Flop_Class", axis=1).values
    return train_x, train_y , test_x, test_y


def fit_model(df):
    train_x, train_y , test_x, test_y = split_train_test(df) #Get the split of data
    model = LogisticRegression().fit(train_x, train_y) #Fit the model
    y_pred = model.predict(test_x) #Predict values
    accuracy = accuracy_score(test_y, y_pred)  #Check the accuracy

    #Plotting roc 
    probs = model.predict_proba(test_x) #Calculate probabilities
    probs = probs[:, 1] #keep the positive class probabilities
    #plot_roc(test_y,probs)
    plot_confusion_matrix(test_y,y_pred)
    print("Accuracy of model: ",accuracy)

def plot_roc(test_y,probs):
    fpr, tpr, _ = roc_curve(test_y, probs)  # calculate roc curve
    plt.plot([0, 1], [0, 1], linestyle='--') # plot no skill
    plt.plot(fpr, tpr, marker='.') # plot the roc curve for the model
    plt.show() # show the plot

def plot_confusion_matrix(test_y,y_pred):
    cm = confusion_matrix(test_y,y_pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax)
    plt.show()
    
df = pd.read_csv("E:/Engineering/7th_sem/AML/Project/AML_Project2018/data/cleaned_msd.csv")
fit_model(df)