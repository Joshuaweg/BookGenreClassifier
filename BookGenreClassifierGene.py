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
import numpy as np
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
from torch.nn.utils import prune
# Get directory name
if os.path.exists("runs/BGC"):
    shutil.rmtree("runs/BGC")
writer = SummaryWriter("runs/BGC")

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
docs =[]
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
y = []
#intersect_tokens =[]
for c in target_category:
    dataByClass[c]=[]
r = 0
for index,row in cleaned_data.iterrows():
    title =row["Title"]
    author = row["Author"]
    description=row["Description"]
    #description =re.sub(r'[^a-zA-Z ]','',description)
    gen = row["Genres1"]
    tokens = nlp(description + title + author)
    t_title = nlp(title.lower())
    t_author= nlp(author.lower())
    #print("Description Length: ",len(description))
    if(len(tokens)<=500 and len(tokens)>110):
        y.append(gen)
        nsw_tokens = [token.lemma_ for token in tokens if not token.text in all_stopwords]
        #print("Description without stop words: ",len(nsw_tokens))
        nsw_tokens = [token for token in nsw_tokens if not token in t_title.text]
        #print("Description without title: ",len(nsw_tokens))
        nsw_tokens = [token for token in nsw_tokens if not (token in t_author.text)]
        #nsw_tokens = [token for token in nsw_tokens if len(token) >= 7]
        #print("Description without Author Name: ",len(nsw_tokens))
        #if r<10:
            #print(title,"\n",gen,"\n",description)
        docs.append(" ".join(nsw_tokens))
        if r<100:
            writer.add_text(title," ".join(nsw_tokens)+"---"+gen)
            r+=1
        tok_list = Union(tok_list,nsw_tokens)
        dataByClass[row.Genres1]=Union( dataByClass[row.Genres1],nsw_tokens)
        if(index%1000==0):

            print(index,"Titles processed")
#print(len(tok_list))
#print(len(dataByClass["Fiction"]))
#print(len(dataByClass["Nonfiction"]))
sharedWords=[tok for tok in dataByClass["Fiction"] if tok  in dataByClass["Nonfiction"]]
i = 0
for doc in docs:
    i += 1
    doc = [tok.text for tok in nlp(doc) if not tok.text in sharedWords]
    if (i%100==0):
        print(doc)
t_vectors=tfidf.fit_transform(docs)
t_vectors = torch.tensor(t_vectors.toarray(),dtype=torch.float32)
print(t_vectors.shape)
#print(f"Total variance explained: {np.sum(svd.explained_variance_ratio_):.2f}")


i=0
for g in target_category:
    #print(g)
    genre2vec[g]=i 
    vec2genre[i]=g
    i+=1
y_set =torch.zeros([len(y)],dtype=torch.long)
l=0
for g in y:
    if g == "Fiction":
        y_set[l]=0
    else:
        y_set[l]=1
    l+=1
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

#hyperparameters
x_train, x_test, y_train, y_test =train_test_split(t_vectors,y_set,test_size=.2,random_state=42)

#print(x_train)
input_size = len(t_vectors[0]) #number of tokens in corpus
hidden_size =  32
hidden_size2 = 8
num_classes = len(target_category) #number of distinct genres
num_epoch = 4

batch_size = 2

learning_rate = 0.005

#Dataset
#will add code to read in dataset here

#two hidden layer NeuralNet


class NeuralNet(nn.Module):
    def __init__(self, input_size,hidden_size,hidden_size2,num_classes):
        super(NeuralNet, self).__init__()

        #hidden layer 1
        self.l1 = nn.Linear(input_size,hidden_size)
        #dropout1
        self.dropout1 = nn.Dropout(0.3)
        #hidden layer 2
        self.l2 = nn.Linear(hidden_size,hidden_size2)
        #dropout2
        self.dropout2 = nn.Dropout(0.3)
        #activation function
        self.activation = nn.ReLU()
        # output layer
        self.l3 = nn.Linear(hidden_size2,num_classes)
    def forward(self, x):

        out = self.l1(x) #input -> hidden
        out = self.activation(out) #activation on hidden
        x = self.dropout1(x) #first dropout
        out=self.l2(out)
        out=self.activation(out)
        x = self.dropout2(x) # second dropout
        out = self.l3(out) # hidden -> output
        return out
class Texts(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, vectors, labels, transform=None):
        'Initialization'
        self.labels = labels
        self.vectors = vectors
        self.transform =transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.vectors)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.vectors[index]
        y = self.labels[index]

        return X, y
  

model = NeuralNet(input_size,hidden_size,hidden_size2,num_classes)
model = model.to(device)
# Global Pruning 
parameters = ((model.l1, "weight"), (model.l2, "weight"))
prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=3)
ds = Texts(x_train,y_train, transform=transforms.ToTensor())
test_ds =Texts(x_test,y_test, transform=transforms.ToTensor())
train_gen = DataLoader(dataset=ds,batch_size=batch_size,shuffle=True)
test_gen = DataLoader(dataset=test_ds,batch_size=batch_size,shuffle=False)
print(train_gen)
# loss optimizer
criterion =nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
writer.add_graph(model,torch.zeros(input_size).to(device))
#training loop will need to add
train = cleaned_data[:2127]
test =cleaned_data[2127:]
n_total_steps = len(train_gen)
running_loss = 0.0
running_correct = 0.0
for epoch in range(num_epoch):
    ite = 0
    for i, (vectors,labels) in enumerate(train_gen):
        
        #print(vectors.shape)
       #print("Description without Author Name: ",len(nsw_tokens))
        vectors=vectors.to(device)
        labels=labels.to(device)
        
        outputs = model(vectors)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predictions = torch.max(outputs, 1)
        running_correct +=(predictions == labels).sum().item()

        #print("Running Loss: ",running_loss)
        #print("prediction: ",prediction)
        #print("label: ",label)
           # print(running_correct)
        
        if((i+1)%50==0):
            print(f'epoch {epoch+1}/{num_epoch}, step {i+1}/{n_total_steps}, loss={loss.item():.4f}')
            writer.add_scalar('training loss', running_loss/10, (epoch*n_total_steps)+1+i)
            writer.add_scalar('accuracy', running_correct/10, (epoch*n_total_steps)+1+i)
            running_loss=0.0
            running_correct=0.0
        ite+=1
labels = []
preds =[]
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    ite = 0
    for vectors, labels1 in test_gen:
        vectors=vectors.to(device)
        labels1 = labels1.to(device)
        outputs = model(vectors)

        _, predictions = torch.max(outputs, 1)
        n_samples += labels1.shape[0]
        n_correct += (predictions == labels1).sum().item()
        class_predictions = [F.softmax(output,dim=0) for output in outputs]
        #print(class_predictions)
        preds.append(class_predictions)
        labels.append(labels1)
        ite+=1
    preds = torch.cat([torch.stack(batch) for batch in preds])
    labels = torch.cat(labels)
        
acc = 100.0* (n_correct/n_samples)
print(f'accuracy = {acc}')
classes = range(2)
for i in classes:
    labels_i = labels==i
    preds_i = preds[:, i]
    writer.add_pr_curve(str(i),labels_i,preds_i,global_step=0)
writer.close()
