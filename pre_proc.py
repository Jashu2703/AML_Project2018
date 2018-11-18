import pandas
import csv
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api

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

app_WtV = ['title']#,'artistLocation','artistName','release','artistmbTags','artistTerms']

encoded_vals = []
list_wds = []

for aph in app_WtV:
    for i in range(tot_rows):
        encoded_vals.append(data[aph][i][2:-1])
    #print(encoded_vals)
    for i in range(tot_rows): 
        list_wds.append(encoded_vals[i].split(' '))
  
  
'''# load the google word2vec model
filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)

result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)'''




word_vectors = api.load("glove-wiki-gigaword-100") 
vector = word_vectors['hello']
print(vector)
similarity = word_vectors.similarity('lady', 'woman')
print(similarity)

similarity = word_vectors.similarity('guess', 'woman')
print(similarity)
''' #print(list_wds)
    model = Word2Vec(list_wds, min_count=1)
    #print(model)
    # summarize vocabulary
    words = list(model.wv.vocab)
    #print(words)
    # access vector for one word
    print(cosine_similarity(model['king'], model['man']))
    # save model
    model.save('model.bin')'''
