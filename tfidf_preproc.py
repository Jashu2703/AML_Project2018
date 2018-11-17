from sklearn.feature_extraction.text import TfidfVectorizer
import pandas
import csv
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


with open('msd.csv', 'r') as f:
    d_reader = csv.DictReader(f)
    headers = d_reader.fieldnames

print(len(headers))
data = pandas.read_csv('msd.csv')

s = data[headers[0]]
tot_rows =  len(s)
major_head = []

to_rem_head = ['artistDigitalID','artistID','artistLatitude','artistLongitude','artistmbID',
'releaseDigitalID','trackID','trackDigitalID','segementsConfidence_mean','segementsConfidence_var',
'segmentsLoudnessMax_mean','segmentsLoudnessMax_var', 'analysisSampleRate']

sparse_head = ['artistmbTagsCount_mean','artistmbTagsCount_var','danceability','energy','year','artistmbTags','release']

for sh in sparse_head:
    s = data[sh]
    val_ct = s.count()
    perc = (val_ct/tot_rows)*100
    print("Percentage for ", sh ,"is ", perc)


for i in headers:
    if i not in to_rem_head:
        major_head.append(i)

app_WtV = ['title']


encoded_vals = []
list_wds = []

for aph in app_WtV:
    for i in range(tot_rows):
        encoded_vals.append(data[aph][i][2:-1])

#print(encoded_vals)

vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                             min_df=2, stop_words='english',
                             use_idf=True)

text_tfidf = vectorizer.fit_transform(encoded_vals)

print(text_tfidf[0])

#vector = vectorizer.transform([encoded_vals[0]])

#print(vectorizer.vocabulary_)
#print(vectorizer.idf_)

#text_tfidf.save('tfidf.pk1')
#tfidf = gensim.models.TfidfModel.load('tfidf.pk1')

svd = TruncatedSVD(100)
lsa = make_pipeline(svd, Normalizer(copy=False))

# Run SVD on the training data, then project the training data.
#************************Is the final 2D aray to be used
text_lsa = lsa.fit_transform(text_tfidf)

print(text_lsa[0])

print(len(encoded_vals), len(text_lsa))
