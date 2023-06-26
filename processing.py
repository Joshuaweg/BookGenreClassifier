import spacy
import numpy as np
import pandas as pd
import sys
import json
import matplotlib.pyplot as plt
from nltk.stem.porter import *
import timeit

start = timeit.default_timer()
f = open("books.json","w")
def n_grams(doc, n):
    return [doc[i:i+n] for i in range(len(doc)-n+1)]
def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list
def NgramUnions(lst1, lst2):
    l3 = lst1+lst2
    final_list = [list(x) for x in set(tuple(x) for x in l3)]
    return final_list
st = PorterStemmer()
nlp = spacy.load('en_core_web_lg')
all_stopwords = nlp.Defaults.stop_words
title2Tok={}
title2Lemma={}
title2Stem={}
title2Bigram={}
title2Trigram={}
title2Quadgrams={}
title2Pentagrams={}
toks = []
gram2 = []
gram3 = []
gram4 = []
gram5 = []
genre2Toks={}
genre2grams2={}
genre2grams3 = {}
genre2grams4 = {}
genre2grams5 = {}
master ={}

#reading in data as pandas dataframe
data = pd.read_csv("data\\genre_data.csv")
for g in data["Genres1"].unique():
    genre2Toks[g]=[]
    genre2grams2[g]=[]
    genre2grams3[g]=[]
    genre2grams4[g]=[]
    genre2grams5[g]=[]
print("Data for Genre Classification:")
print(data.head())
print(data.Title.head())
print(data.Author.head())
print(data.Description.head())
print(data.Genres1.head())
row = data.values[0]
print(row)
des = row[2]
print("A description: ",des)
tokens = nlp(des.lower())
tok_list = [(token.text,token.pos_) for token in tokens]
print("Tokens: ",tok_list)
print("token size: ",len(tok_list))
tok_list = [token.text for token in tokens if not(token.text in all_stopwords)]
print("Tokens without stop words: ",tok_list)
print("token size: ",len(tok_list))
tok_list = [token.text for token in tokens if not(token.text in all_stopwords) and not(token.is_punct) and not token.pos_=="PROPN" and not token.is_oov]
print("Tokens without stop words and punctuations: ",tok_list)
print("token size: ",len(tok_list))
tok_list = [token.lemma_ for token in tokens if not(token.text in all_stopwords) and not(token.is_punct) and not token.pos_=="PROPN" and not token.is_oov and not token.is_digit]
print("Tokens without stop words and punctuations as lemma: ",tok_list)
print("token size: ",len(tok_list))
tok_list = [st.stem(token.text) for token in tokens if not(token.text in all_stopwords) and not(token.is_punct) and not token.pos_=="PROPN" and not token.is_oov and not token.is_digit]
print("Tokens without stop words and punctuations as stem: ",tok_list)
print("token size: ",len(tok_list))
bigrams = n_grams(tok_list,2)
print("bigrams: ",bigrams)
print(len(bigrams))
trigrams = n_grams(tok_list,3)
print("trigrams: ",trigrams)
print(len(trigrams))
decigrams = n_grams(tok_list,10)
print("decigrams: ",decigrams)
print(len(decigrams))
for index, row in data.iterrows():

    des = row.Description
   # print("A description: ",des[:20])
    print(des)
    tokens = nlp(des.lower())
    tok_list = [token.text for token in tokens]
    tok_list = [*set(tok_list)]
    title2Tok[row["Title"]]=(row["Genres1"],len(tok_list),tok_list)
    master[row["Title"]]= dict(genre=row["Genres1"],length=len(tok_list),tokens=tok_list)
    print(title2Tok[row["Title"]])
    tok_list = [token.text for token in tokens if not(token.text in all_stopwords)]
    tok_list = [token.text for token in tokens if not(token.text in all_stopwords) and not(token.is_punct) and not token.pos_=="PROPN" and not token.is_oov and not token.is_digit]
    tok_list = [token.lemma_ for token in tokens if not(token.text in all_stopwords) and not(token.is_punct) and not token.pos_=="PROPN" and not token.is_oov and not token.is_digit]
    tok_list = [*set(tok_list)]
    title2Lemma[row["Title"]]=(row["Genres1"],len(tok_list),tok_list)
    tok_list = [st.stem(token.text) for token in tokens if not(token.text in all_stopwords) and not(token.is_punct) and not token.pos_=="PROPN" and not token.is_oov and not token.is_digit]
    tok_list = [*set(tok_list)]   
    title2Stem[row["Title"]]=(row["Genres1"],len(tok_list),tok_list)
    bigrams = n_grams(tok_list,2)
    bigrams =[list(x) for x in set(tuple(x) for x in bigrams)]
    title2Bigram[row["Title"]]=(row["Genres1"],len(bigrams),bigrams)
    trigrams = n_grams(tok_list,3)
    trigrams =[list(x) for x in set(tuple(x) for x in trigrams)]
    title2Trigram[row["Title"]]=(row["Genres1"],len(trigrams),trigrams)
    quadgrams = n_grams(tok_list,4)
    quadgrams =[list(x) for x in set(tuple(x) for x in quadgrams)]
    title2Quadgrams[row["Title"]]=(row["Genres1"],len(quadgrams),quadgrams)
    pentagrams = n_grams(tok_list,5)
    pentagrams =[list(x) for x in set(tuple(x) for x in pentagrams)]
    title2Pentagrams[row["Title"]]=(row["Genres1"],len(pentagrams),pentagrams)
obj = json.dumps(master,indent=4)
f.write(obj)
f.close()
sys.exit()

for t in title2Tok.keys():
    toks=Union(toks,title2Lemma[t][2])
    gram2=NgramUnions(gram2,title2Bigram[t][2])
    gram3=NgramUnions(gram3,title2Trigram[t][2])
    gram4=NgramUnions(gram4,title2Quadgrams[t][2])
    gram5=NgramUnions(gram5,title2Pentagrams[t][2])
    genre2Toks[title2Lemma[t][0]]=Union(genre2Toks[title2Lemma[t][0]],title2Lemma[t][2])
    genre2grams2[title2Lemma[t][0]]=NgramUnions(genre2grams2[title2Lemma[t][0]],title2Bigram[t][2])
    genre2grams3[title2Lemma[t][0]]=NgramUnions(genre2grams3[title2Lemma[t][0]],title2Trigram[t][2])
    genre2grams4[title2Lemma[t][0]]=NgramUnions(genre2grams4[title2Lemma[t][0]],title2Quadgrams[t][2])
    genre2grams5[title2Lemma[t][0]]=NgramUnions(genre2grams5[title2Lemma[t][0]],title2Pentagrams[t][2])
print("Total tokens: ",len(toks))
print("Total bigrams: ",len(gram2))
print("Total trigrams: ",len(gram3))
print("Total quadgrams: ",len(gram4))
print("Total pentagrams: ",len(gram5))

for g in genre2Toks.keys():
    print(g,":")
    print("tokens: ",len(genre2Toks[g]))
    print("bigrams: ",len(genre2grams2[g]))
    print("trigrams: ",len(genre2grams3[g]))
    print("quadgrams: ",len(genre2grams4[g]))
    print("pentagrams: ",len(genre2grams5[g]))

stop = timeit.default_timer()

print('Time: ', (stop - start)/60.0," Minutes")

    
