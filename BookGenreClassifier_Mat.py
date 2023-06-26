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
from sklearn.model_selection import train_test_split

def n_grams(doc, n):
    return [doc[i:i+n] for i in range(len(doc)-n+1)]
def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list
def NgramUnions(lst1, lst2):
    l3 = lst1+lst2
    final_list = [list(x) for x in set(tuple(x) for x in l3)]
    return final_list

data = pd.read_csv("data\\genre_data.csv")
target_category=[]
#print(data[["Title","Author","Description","Genres1"]].head())
#print(len(data), "Books")

genres = []
count_genre = {}

#for g in data["Genres1"].values:
#    if not g in genres:
#        genres.append(g)
#        count_genre[g] = 0
#        #print(g)
#    count_genre[g]+=1
##print(genres)
##print(len(genres))
#sorted_count_genre = sorted(count_genre.items(), key=lambda x:x[1])
#mean = 0.0
#sum = 0.0
#for g in sorted_count_genre:
#    sum += g[1]
#mean = sum/len(genres)
#variance = 0.0
#for g in sorted_count_genre:
#    variance += (g[1]-mean)**2
#variance = variance/(len(genres))
#stdev = variance**.5
##print(mean)
##print(stdev)
##print(variance)
#sorted_count_genre = [g for g in sorted_count_genre if g[1]>=mean]

##print(sorted_count_genre)
##print(len(sorted_count_genre))
#sum = 0
#target_category = []
#for g in sorted_count_genre:
#    target_category.append(g[0])
#print(target_category)
#cleaned_data = data[data["Genres1"].isin(target_category)]
cleaned_data=data[["Title","Author","Description","Genres1"]]
cleaned_data = cleaned_data[cleaned_data.Genres1.isin(['Fiction','Nonfiction'])]
#cleaned_data = cleaned_data[:660]
target_category=['Fiction','Nonfiction']
print(len(cleaned_data))
#cleaned_data=cleaned_data[:200]
#Description Vectorization
nlp_data=[]
dataByClass={}
nlp = spacy.load('en_core_web_lg')
all_stopwords = nlp.Defaults.stop_words
tok_list = []
tok_pos = []
word2vec = {}
vec2word = {}
genre2vec ={}
vec2genre = {}
sharedWords = []
#intersect_tokens =[]
for c in target_category:
    dataByClass[c]=[]
for index,row in cleaned_data.iterrows():
    title =row["Title"]
    author = row["Author"]
    description=row["Description"]
    tokens = nlp(description.lower())
    t_title = nlp(title.lower())
    t_author= nlp(author.lower())
    #print("Description Length: ",len(description))
    nsw_tokens = [token.lemma_ for token in tokens if not token.text in all_stopwords]
    #print("Description without stop words: ",len(nsw_tokens))
    nsw_tokens = [token for token in nsw_tokens if not token in t_title.text]
    #print("Description without title: ",len(nsw_tokens))
    nsw_tokens = [token for token in nsw_tokens if not (token in t_author.text)]
    #nsw_tokens = [token for token in nsw_tokens if len(token) >= 7]
    #print("Description without Author Name: ",len(nsw_tokens))

    tok_list = Union(tok_list,nsw_tokens)
    dataByClass[row.Genres1]=Union( dataByClass[row.Genres1],nsw_tokens)
    if(index%1000==0):
        print(index,"Titles processed")
print(len(tok_list))
print(len(dataByClass["Fiction"]))
print(len(dataByClass["Nonfiction"]))
sharedWords=[tok for tok in dataByClass["Fiction"] if tok  in dataByClass["Nonfiction"]]
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
x_train, x_test, y_train, y_test =train_test_split(cleaned_data[["Title","Author","Description"]],cleaned_data["Genres1"],test_size=.2,random_state=42)
#print(x_train)
input_size = len(tok_list) #number of tokens in corpus
hidden_size =  40
hidden_size2 = 4
num_classes = len(target_category) #number of distinct genres
num_epoch = 2

batch_size = 1

learning_rate = 0.005

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
train = cleaned_data[:2127]
test =cleaned_data[2127:]
for epoch in range(num_epoch):
    ite = 0
    for row in x_train.values:
        title =row[0]
        author = row[1]
        description=row[2]
        tokens = nlp(description.lower())
        t_title = nlp(title.lower())
        t_author= nlp(author.lower())
        #print("Description Length: ",len(description))
        nsw_tokens = [token.lemma_ for token in tokens if not token.text in all_stopwords]
        #print("Description without stop words: ",len(nsw_tokens))
        nsw_tokens = [token for token in nsw_tokens if not token in t_title.text]
        #print("Description without title: ",len(nsw_tokens))
        nsw_tokens = [token for token in nsw_tokens if not token in t_author.text]
        nsw_tokens = [token for token in nsw_tokens if len(token) >= 7]
        nsw_tokens = [token for token in nsw_tokens if not (token in sharedWords)]
       #print("Description without Author Name: ",len(nsw_tokens))
        in_layer = torch.zeros(input_size,dtype=torch.float32)
        label = torch.zeros(num_classes,dtype=torch.float32).to(device)
        for token in nsw_tokens:
            ind = word2vec[token]
            if not( in_layer[ind] == 1.0):
                in_layer[ind] = 1.0;
        in_layer=in_layer.to(device)
        label[genre2vec[y_train.values[ite]]] = 1.0
        outputs = model(in_layer)
        loss = criterion(outputs,label)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if((ite+1)%100==0):
            print(f'epoch {epoch+1}, step {ite+1}, loss={loss.item():.4f}')
        ite+=1
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    ite = 0
    for row in x_test.values:
       
        title =row[0]
        author = row[1]
        description=row[2]
        tokens = nlp(description.lower())
        t_title = nlp(title.lower())
        t_author= nlp(author.lower())
        #print("Description Length: ",len(description))
        nsw_tokens = [token.lemma_ for token in tokens if not token.text in all_stopwords]
        #print("Description without stop words: ",len(nsw_tokens))
        nsw_tokens = [token for token in nsw_tokens if not token in t_title.text]
        #print("Description without title: ",len(nsw_tokens))
        nsw_tokens = [token for token in nsw_tokens if not token in t_author.text]
        nsw_tokens = [token for token in nsw_tokens if len(token) >= 7]
        nsw_tokens = [token for token in nsw_tokens if not (token in sharedWords)]
       #print("Description without Author Name: ",len(nsw_tokens))
        in_layer = torch.zeros(input_size,dtype=torch.float32)
        label = torch.zeros(num_classes,dtype=torch.float32).to(device)
        for token in nsw_tokens:
            ind = word2vec[token]
            if not( in_layer[ind] == 1.0):
                in_layer[ind] = 1.0;
        in_layer=in_layer.to(device)
        label[genre2vec[y_test.values[ite]]] = 1.0
       # print(label)
        outputs = model(in_layer)
        max_pos = 0;
        max_val = -2;
        for o in outputs:
            if max_pos == 0:
                max_val = o;
            if o>max_val:
                max_pos = o
        prediction = torch.zeros(num_classes,dtype=torch.float32)
        prediction[max_pos]=1.0
        n_samples += 1
        #print(prediction)
        if torch.equal(prediction,label):
            n_correct += 1
        ite+=1
        
        
   

acc = 100.0* (n_correct/n_samples)
print(f'accuracy = {acc}')
