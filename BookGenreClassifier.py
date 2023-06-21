from re import I

from pandas.core.missing import clean_fill_method
import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import spacy
import numpy as np
import pandas as pd
import sys

data = pd.read_csv("data\\genre_data.csv")
#print(data[["Title","Author","Description","Genres1"]].head())
#print(len(data), "Books")

genres = []
count_genre = {}

for g in data["Genres1"].values:
    if not g in genres:
        genres.append(g)
        count_genre[g] = 0
        #print(g)
    count_genre[g]+=1
#print(genres)
#print(len(genres))
sorted_count_genre = sorted(count_genre.items(), key=lambda x:x[1])
mean = 0.0
sum = 0.0
for g in sorted_count_genre:
    sum += g[1]
mean = sum/len(genres)
variance = 0.0
for g in sorted_count_genre:
    variance += (g[1]-mean)**2
variance = variance/(len(genres))
stdev = variance**.5
#print(mean)
#print(stdev)
#print(variance)
sorted_count_genre = [g for g in sorted_count_genre if g[1]>=mean]

#print(sorted_count_genre)
#print(len(sorted_count_genre))
sum = 0
target_category = []
for g in sorted_count_genre:
    target_category.append(g[0])
#print(target_category)
cleaned_data = data[data["Genres1"].isin(target_category)]
cleaned_data=cleaned_data[["Title","Author","Description","Genres1"]]
#Description Vectorization
nlp = spacy.load('en_core_web_lg')
all_stopwords = nlp.Defaults.stop_words
tok_list = []
tok_pos = []
word2vec = {}
vec2word = {}
genre2vec ={}
vec2genre = {}
for index,row in cleaned_data.iterrows():
    title =row["Title"]
    author = row["Author"]
    description=row["Description"]
    tokens = nlp(description)
    t_title = nlp(title)
    t_author= nlp(author)
    nsw_tokens = [token for token in tokens if not token.text in all_stopwords]
    nsw_tokens = [token for token in nsw_tokens if not token.text in t_title.text]
    nsw_tokens = [token for token in nsw_tokens if not token.text in t_author.text]
    for t in nsw_tokens:
        if not(t.text in tok_list):
            tok_list.append(t.text)
    if index%1000==0:
        print(index,"Titles processed")
#token_vector = np.zeros(shape=(len(tok_list),len(tok_list)))
tok_pos = []
i = 0
for t in tok_list:
    tok_pos.append(i)
    word2vec[t]=i 
    vec2word[i]=t
    i+=1
i=0
for g in target_category:
    genre2vec[g]=i 
    vec2genre[i]=g
    i+=1
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

#hyperparameters

input_size = len(tok_list) #number of tokens in corpus
hidden_size =  6000
hidden_size2 = 10000
num_classes = len(target_category) #number of distinct genres
num_epoch = 10

batch_size = 1

learning_rate = 0.05

#Dataset
#will add code to read in dataset here

#two hidden layer NeuralNet
class NeuralNet(nn.Module):
    def __init__(self, input_size,hidden_size,hidden_size2,num_classes):
        super(NeuralNet, self).__init__()
        #hidden layer 1
        self.l1 = nn.Linear(input_size,hidden_size)
        #hidden layer 2
        self.l2 = nn.Linear(hidden_size,hidden_size2)
        #activation function
        self.activation = nn.ReLU()
        # output layer
        self.l3 = nn.Linear(hidden_size2,num_classes)
    def forward(self, x):
        out = self.l1(x) #input -> hidden
        out = self.activation(out) #activation on hidden
        out=self.l2(out)
        out=self.activation(out)
        out = self.l3(out) # hidden -> output
        return out

model = NeuralNet(input_size,hidden_size,hidden_size2,num_classes)
model = model.to(device)
# loss optimizer
criterion =nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

#training loop will need to add
for epoch in range(num_epoch):
    for index, row in cleaned_data.iterrows():
        title =row["Title"]
        author = row["Author"]
        description=row["Description"]
        tokens = nlp(description)
        t_title = nlp(title)
        t_author= nlp(author)
        nsw_tokens = [token for token in tokens if not token.text in all_stopwords]
        nsw_tokens = [token for token in nsw_tokens if not token.text in t_title.text]
        nsw_tokens = [token for token in nsw_tokens if not token.text in t_author.text]
        in_layer = torch.zeros(input_size,dtype=torch.float32)
        label = torch.zeros(num_classes,dtype=torch.float32).to(device)
        for token in nsw_tokens:
            ind = word2vec[token.text]
            if not( in_layer[ind] == 1.0):
                in_layer[ind] = 1.0;
        in_layer=in_layer.to(device)
        label[genre2vec[row["Genres1"]]] = 1.0
        outputs = model(in_layer)
        loss = criterion(outputs,label)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(index%1000==0):
            print(f'epoch {epoch+1}, step {index}, loss={loss.item():.4f}')