"""
Program to classify book genres with a feedforward neural network
Takes in data from good reads dataset
"""
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
import gzip,pickle
# Get directory name
def n_grams(doc, n):
    return [doc[i:i+n] for i in range(len(doc)-n+1)]
def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list
def NgramUnions(lst1, lst2):
    l3 = lst1+lst2
    final_list = [list(x) for x in set(tuple(x) for x in l3)]
    return final_list
class NeuralNet(nn.Module):
    """ class for feed forward neural network """
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
def train():
    if os.path.exists("runs/BGC"):
        shutil.rmtree("runs/BGC")
    writer = SummaryWriter("runs/BGC")

    tfidf = TfidfVectorizer()
    data = pd.read_csv("data\\genre_data.csv")
    target_category=[]

    genres = []
    count_genre = {}


    cleaned_data=data[["Title","Author","Description","Genres1"]]
    cleaned_data = cleaned_data[cleaned_data.Genres1.isin(['Fiction','Nonfiction','Fantasy'])]
    #cleaned_data=cleaned_data[cleaned_data.Genres1.isin(['Fantasy','Historical Fiction','Classics','Young Adult','Mystery','Romance','Science Fiction','History','Thriller','Horror','Self Help'])]
    #cleaned_data=cleaned_data.groupby("Genres1").head(120)
    #cleaned_data = cleaned_data[:660]
    target_category=['Fiction','Nonfiction','Fantasy']
    #target_category=['Fantasy','Historical Fiction','Classics','Young Adult','Mystery','Romance','Science Fiction','History','Thriller','Horror','Self Help']
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
# cleaning up data and getting relevent words from description
    for index,row in cleaned_data.iterrows():
        title =row["Title"]
        author = row["Author"]
        description=row["Description"]
    #description =re.sub(r'[^a-zA-Z ]','',description)
        gen = row["Genres1"]
        tokens = nlp(description + title + author)
        t_title = nlp(title.lower())
        t_author= nlp(author.lower())
        if(len(tokens)<=500 and len(tokens)>110):
            y.append(gen)
            nsw_tokens = [token.lemma_ for token in tokens if not token.text in all_stopwords]
            nsw_tokens = [token for token in nsw_tokens if not token in t_title.text]
            nsw_tokens = [token for token in nsw_tokens if not (token in t_author.text)]
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

#sharedWords=[tok for tok in dataByClass["Fiction"] if (tok in dataByClass["Nonfiction"] or tok in dataByClass["Fantasy"])]
    i = 0
    for doc in docs:
        i += 1
        doc = [tok.text for tok in nlp(doc)]
        if (i%100==0):
            print(doc)
# vectorizing descriptions
    t_vectors=tfidf.fit_transform(docs)
    with gzip.open('2description_vectorsNN.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
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
        y_set[l]=genre2vec[g]
        l+=1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

#hyperparameters
    x_train, x_test, y_train, y_test =train_test_split(t_vectors,y_set,test_size=.2,random_state=42)

#print(x_train)
    input_size = len(t_vectors[0]) #number of tokens in corpus
    hidden_size =  100
    hidden_size2 = 155
    num_classes = len(target_category) #number of distinct genres
    num_epoch = 4

    batch_size = 10

    learning_rate = 0.005

#two hidden layer NeuralNet

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

# training
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
        
            if((i+1)%10==0):
                print(f'epoch {epoch+1}/{num_epoch}, step {i+1}/{n_total_steps}, loss={loss.item():.4f}')
                writer.add_scalar('training loss', running_loss/10, (epoch*n_total_steps)+1+i)
                writer.add_scalar('accuracy', running_correct/10, (epoch*n_total_steps)+1+i)
                print(target_category[labels[-1].item()])
                print(target_category[predictions[-1].item()])
                running_loss=0.0
                running_correct=0.0
            ite+=1
    labels = []
    preds =[]
    confusion_matrix = torch.zeros(num_classes, num_classes)
# testing
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
            for clss, prd in zip(labels1,predictions):
                confusion_matrix[clss.long(), prd.long()] += 1
        preds = torch.cat([torch.stack(batch) for batch in preds])
        labels = torch.cat(labels)

# getting and printing accuracy
    acc = 100.0* (n_correct/n_samples)
    print(f'accuracy = {acc}')
    print(confusion_matrix)
    torch.save(model,"NN2.pt")
    classes = range(len(target_category))
    for i in classes:
        labels_i = labels==i
        preds_i = preds[:, i]
        writer.add_pr_curve(str(i),labels_i,preds_i,global_step=0)
    writer.close()


if __name__=="__main__":
    train()