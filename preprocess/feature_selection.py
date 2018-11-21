import pandas
import csv
import math
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_regression

with open('../data/msd.csv', 'r') as f:
    d_reader = csv.DictReader(f)
    headers = d_reader.fieldnames
print(len(headers))
data = pandas.read_csv('../data/msd.csv')
s = data[headers[0]]
tot_rows =  len(s)
major_head = []

to_rem_head = ['artistDigitalID','artistID','artistLatitude','artistLongitude','artistmbID',
'releaseDigitalID','trackID','trackDigitalID','segementsConfidence_mean','segementsConfidence_var',
'segmentsLoudnessMax_mean','segmentsLoudnessMax_var', 'analysisSampleRate','audioMD5']
sparse_head = ['artistmbTagsCount_mean','artistmbTagsCount_var','danceability','energy']

for sh in sparse_head:
    s = data[sh]
    val_ct = s.count()
    perc = (val_ct/tot_rows)*100
    print("Percentage for ", sh ,"is ", perc)
print(len(headers),len(to_rem_head), len(sparse_head))
major_head = list(set(headers) - set(to_rem_head) - set(sparse_head))
#Removinag all rows with any NAN value
cleaned_data =  data[major_head]
print("Cleaned data:  ",len(cleaned_data))
for i in major_head:
    cleaned_data =  cleaned_data[pandas.notnull(cleaned_data[i])]

coulmn_names = cleaned_data.columns.values  # Cloumn Names of dataframe

#Normalization
norm = cleaned_data.values  #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(norm)
cleaned_data = pandas.DataFrame(x_scaled)

print(cleaned_data.shape)

cleaned_data = pandas.DataFrame(cleaned_data.values, columns=coulmn_names)
print(cleaned_data)
Y = cleaned_data['songHotttnesss']
Y_new = Y.values
# print("Y_length is" , len(Y))
major_head = list(set(major_head) - set('songHotttnesss'))
X = cleaned_data[major_head]
X = X.astype(dtype='float64')

# Create an SelectKBest object to select features with two best ANOVA F-Values
fvalue_selector = SelectKBest(mutual_info_regression, k=81)
# Apply the SelectKBest object to the features and target
X_kbest = fvalue_selector.fit_transform(X, Y_new)
mask = fvalue_selector.get_support() #list of booleans
new_features = [] # The list of your K best features
print("The main headings are",len(major_head))
for bool, feature in zip(mask, major_head):
    if bool:
        new_features.append(feature)

dataframe = pandas.DataFrame(X_kbest, columns=new_features)
print(new_features)
 #Writing the output to a csv file.
dataframe['Hit_Flop_Class'] = Y_new
dataframe.to_csv(path_or_buf='../data/msd_feature_selection_normalised.csv', sep=',')