import pandas as pd 
#Utilities
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
#Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import tree
#Plot
import matplotlib.pyplot as plt
import seaborn as sns

def split_train_test(df): 
    train, test = train_test_split(df, test_size=0.2)
    train_y, test_y = train["Hit_Flop_Class"].values ,test["Hit_Flop_Class"].values 
    train_x, test_x =  train.drop(labels="Hit_Flop_Class", axis=1).values , test.drop(labels="Hit_Flop_Class", axis=1).values
    return train_x, train_y , test_x, test_y


def fit_with_cross_validation(df):
    test, train = df["Hit_Flop_Class"].values, df.drop(labels="Hit_Flop_Class", axis=1).values
    model = LogisticRegression()
    y_pred = cross_val_predict(model,train,test,cv=5)   
    print(accuracy_score(test, y_pred))

def plot_confusion_matrix(test_y,y_pred):
    cm = confusion_matrix(test_y,y_pred)                       
    ax = plt.subplot()                                          
    sns.heatmap(cm, annot=True, ax = ax,cmap="YlGnBu")
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels') 
    ax.set_title('Confusion Matrix') 
    ax.xaxis.set_ticklabels(['flop', 'hit'])
    ax.yaxis.set_ticklabels(['flop', 'hit'])
    plt.show()
    


def fit_models(df):
    #Making discrete labels
    df['Hit_Flop_Class'] = (df['Hit_Flop_Class'] > 0.5).astype(int)
    #Splitting
    train_x, train_y , test_x, test_y = split_train_test(df)    # Get the split of data
    models = [
        {
            'label': 'Logistic Regression',
            'model': LogisticRegression()
        },
        {
            'label': 'SVM RBF',
            'model': svm.SVC(probability=True) ,
        },
        {
            'label': 'LinearDiscriminantAnalysis',
            'model': LinearDiscriminantAnalysis()
        },
        {
            'label': 'MLPClassifier',
            'model': MLPClassifier()
        },
        
        {
            'label': 'Decision Tree',
            'model':  tree.DecisionTreeClassifier()
        },
        {
            'label': 'SVM Linear',
            'model':  svm.SVC(probability=True,kernel="linear")
        },
    ]

    for each_model in models:
        model = each_model['model']
        #Fitting
        model = model.fit(train_x, train_y)
        #Predicting
        y_pred = model.predict(test_x)                              

        #plot_confusion_matrix(test_y,y_pred)
        #Printing
        with  open("E:/Engineering/7th_sem/AML/Project/AML_Project2018/outb.txt","a") as target:
            target.write("----------------------"+each_model['label']+"----------------------\nAccuracy of model:" +json.dumps(accuracy_score(test_y, y_pred))+"\n"+json.dumps(classification_report(test_y, y_pred)+"\n"))
        #Ploting
        #Get the roc curve
        fpr, tpr, _ = roc_curve(test_y, model.predict_proba(test_x)[:,1])
        #Find accuracy
        auc = accuracy_score(test_y,y_pred)
        # Now, plot the computed values
        plt.plot(fpr, tpr,  marker='.',label='%s (area = %0.2f)' % (each_model['label'], auc))
    
    # Custom settings for the plot 
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()   # Display
        
        
df = pd.read_csv("E:/Engineering/7th_sem/AML/Project/AML_Project2018/data/msd_feature_selection_normalised.csv")
fit_models(df)