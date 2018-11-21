import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(81, input_dim=81, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model

def split_train_test(df): 
    train, test = train_test_split(df, test_size=0.2)
    train_y, test_y = train["Hit_Flop_Class"].values ,test["Hit_Flop_Class"].values 
    train_x, test_x =  train.drop(labels="Hit_Flop_Class", axis=1).values , test.drop(labels="Hit_Flop_Class", axis=1).values
    return train_x, train_y , test_x, test_y


data = pandas.read_csv('cleaned_msd.csv', index_col=0)

X_train, Y_train , X_test, Y_test = split_train_test(data)


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=500, batch_size=50)


estimator.fit(X_train, Y_train)
prediction = estimator.predict(X_test)
for i in range(len(prediction)):
    print("Prediction is", prediction[i], Y_test[i])

mse_val = mean_squared_error( Y_test, prediction)
print("MSE IS ", mse_val)
