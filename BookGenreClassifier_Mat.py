import re
from pandas.core.missing import clean_fill_method
import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import spacy
import pandas as pd
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import torch.nn.functional as F
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.naive_bayes import GaussianNB
import pickle,gzip

# Get directory name
if os.path.exists("runs/BGC"):
    shutil.rmtree("runs/BGC")
writer = SummaryWriter("runs/BGC")

nb = MultinomialNB()
tfidf = TfidfVectorizer()
def n_grams(doc, n):
    return [doc[i:i+n] for i in range(len(doc)-n+1)]
def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list
def NgramUnions(lst1, lst2):
    l3 = lst1+lst2
    final_list = [list(x) for x in set(tuple(x) for x in l3)]
    return final_list

data = pd.read_csv('https://raw.githubusercontent.com/Joshuaweg/BookGenreClassifier/master/data/genre_data.csv')
target_category=[]

genres = []
count_genre = {}

cleaned_data=data[["Title","Author","Description","Genres1"]]
cleaned_data = cleaned_data[cleaned_data.Genres1.isin(['Fiction','Nonfiction','Fantasy'])]
#cleaned_data = cleaned_data[:660]
target_category=['Fiction','Nonfiction','Fantasy']
print(len(cleaned_data))
#cleaned_data=cleaned_data[:200]
#Description Vectorization
nlp_data=[]
docs =[]
dataByClass={}
import spacy.cli
spacy.cli.download("en_core_web_lg")
nlp = spacy.load('en_core_web_lg')
all_stopwords = nlp.Defaults.stop_words
tok_list = []
tok_pos = []
word2vec = {}
vec2word = {}
genre2vec ={}
vec2genre = {}
sharedWords = []
y = []

for c in target_category:
    dataByClass[c]=[]
r = 0
for index,row in cleaned_data.iterrows():
    title =row["Title"]
    author = row["Author"]
    description=row["Description"]
    description =re.sub(r'[^a-zA-Z ]','',description)
    gen = row["Genres1"]
    tokens = nlp(description.lower())
    t_title = nlp(title.lower())
    t_author= nlp(author.lower())
    if(len(tokens)<=483 and len(tokens)>86):
        y.append(gen)
        nsw_tokens = [token.lemma_ for token in tokens if not token.text in all_stopwords]
        nsw_tokens = [token for token in nsw_tokens if not token in t_title.text]
        nsw_tokens = [token for token in nsw_tokens if not (token in t_author.text)]
        if r<10:
            print(title,"\n",gen,"\n",description.lower())
        docs.append(" ".join(nsw_tokens))
        if r<10:
            writer.add_text(title," ".join(nsw_tokens)+"---"+gen)
            r+=1
        tok_list = Union(tok_list,nsw_tokens)
        dataByClass[row.Genres1]=Union( dataByClass[row.Genres1],nsw_tokens)
        if(index%1000==0):
            print(index,"Titles processed")

sharedWords=[tok for tok in dataByClass["Fiction"] if (tok in dataByClass["Nonfiction"] and tok in dataByClass["Fantasy"])]
for doc in docs:
    doc = [tok.text for tok in nlp(doc) if not tok.text in sharedWords]

i=0
for g in target_category:
    genre2vec[g]=i
    vec2genre[i]=g
    i+=1
y_set =torch.zeros([len(y)],dtype=torch.long)
l=0
for g in y:
    y_set[l]=genre2vec[g]
    l+=1

t_vectors=tfidf.fit_transform(docs)
with gzip.open('NBdescription_vectors.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
t_vectors = torch.tensor(t_vectors.toarray(),dtype=torch.float32)

#hyperparameters
x_train, x_test, y_train, y_test =train_test_split(t_vectors,y_set,test_size=.2,random_state=42)

input_size = len(t_vectors[0]) #number of tokens in corpus
hidden_size =  32
hidden_size2 = 8
num_classes = len(target_category) #number of distinct genres
num_epoch = 4

batch_size = 2
learning_rate = 0.005

mnb = nb.fit(x_train, y_train)
predictedMNB = nb.predict(x_test)
accMNB = accuracy_score(predictedMNB, y_test)
f1MNB = f1_score(predictedMNB, y_test, average="weighted")
cmatrixMNB = confusion_matrix(y_test, predictedMNB)

print(f"MultinomialNB Accuracy Score: {accMNB*100}")
print(f"MultinomialNB f1_score: {f1MNB*100}")
print(f"MultinomialNB confusion matrix: {cmatrixMNB}")

gnb = GaussianNB()

gnb.fit(x_train, y_train)
filename = 'NaiveBayes.pkl'
pickle.dump(gnb, open(filename, 'wb'))
predictedGNB = gnb.predict(x_test)
accuracyGNB = accuracy_score(predictedGNB, y_test)
f1GNB = f1_score(predictedGNB, y_test, average="weighted")
cmatrixGNB = confusion_matrix(y_test, predictedGNB)

print(f"GaussianNB Accuracy Score: {accuracyGNB*100}")
print(f"GaussianNB f1_score: {f1GNB*100}")
print(f"GaussianNB confusion matrix: {cmatrixGNB}")



#F1 score can be interpreted as a measure of overall model performance
#from 0 to 1, where 1 is the best. To be more specific, F1 score can be
# interpreted as the model's balanced ability to both capture positive
#cases (recall) and be accurate with the cases
#it does capture (precision).





