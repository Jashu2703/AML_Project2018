from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd 
import numpy as np

df = pd.read_csv("E://Engineering//7th_sem//AML//Project//AML_Project2018//data//msd_final.csv")
bowcsv = "E://Engineering//7th_sem//AML//Project//AML_Project2018//data//bagofwords.csv"

#given corpus apply bag of words, then returns top n 
#words based on term frequency
#also return the matrix of row vs term frequency
def get_top_n_words(corpus, n=None):
    #get bag of words
    bow = CountVectorizer(stop_words='english')
    #fit the corpus
    bow_transform = bow.fit_transform(corpus)
    # get term frequency
    sum_words = bow_transform.sum(axis=0)
    #list of tuples (word,term_idx,count)
    words_freq = [(word, idx, sum_words[0, idx]) for word, idx in bow.vocabulary_.items()]
    #Sorted based on term frequency and return top n words
    words_freq = sorted(words_freq, key = lambda x: x[2], reverse=True)
    return words_freq[:n] , bow_transform.toarray()

#generates corpus from all the  strings in the csv
def generate_corpus(df):
    corpus = []
    string_columns = ["artistLocation","artistTerms","artistName","release","artistmbTags","similarArtists","title"]
    no_of_docs = df.shape[0]

    #Take each song , consider all the columns above
    #Concat all string columns of particular row
    #Each row is append to the list to form corpus
    for doc_index in range(no_of_docs):
        doc = ''
        for each_column in string_columns:            
            doc = doc + str(df[each_column][doc_index]) + " "
        corpus.append(doc)
    return corpus

#Given df applies bag of words
#Get top 900 words based on term frequency
#make words as columns and fill the corresponding occurence
#of word in the song
def generate_data(df):
    corpus = generate_corpus(df)
    #here 901 because there is nan is the top 900
    words , matrix= get_top_n_words(corpus,901)
    headers = [x[0] for x in words]
    indexes = [x[1] for x in words]

    with open(bowcsv,'w') as fd:
        song_header = "\t".join(str(x) for x in headers)
        fd.write(song_header)
        fd.write("\n")
    
    for row in matrix:
        song_row = [row[index] for index in indexes]
        song_row = "\t".join(str(x) for x in song_row)
        with open(bowcsv,'a') as fd:
                fd.write(song_row)
                fd.write("\n")
    
    print("-----------Done------------------")
    
generate_data(df)




"""
print(zip(bow.get_feature_names(), np.ravel(bow_transform.sum(axis=0))))
print(tfidf_transform)          #prints document_id, token_id 
print(bow.get_feature_names())  #to get feature names of token_id
print(bow_transform.toarray())  #to get count of feature in particular row
#apply tf idf measure
tfidf = TfidfTransformer()
tfidf_transform = tfidf.fit_transform(bow_transform)
"""