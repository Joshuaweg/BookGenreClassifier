import re
import json
from turtle import Vec2D
from pandas.core.missing import clean_fill_method
from mpl_toolkits import mplot3d
import torch
import pickle
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import spacy
import numpy as np
import pandas as pd
import sys, threading, traceback
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import gzip
# Get directory name
if os.path.exists("runs/BGC-RNN3"):
    shutil.rmtree("runs/BGC-RNN3")
writer = SummaryWriter("runs/BGC-RNN3")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tfidf = TfidfVectorizer(analyzer="word",use_idf=True,smooth_idf=True)
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
doc2Mat = []
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
cleaned_data = cleaned_data[cleaned_data.Genres1.isin(['Fantasy','Fiction','Nonfiction'])]
#cleaned_data=cleaned_data[cleaned_data.Genres1.isin(['Fantasy','Historical Fiction','Classics','Young Adult','Mystery','Romance','Science Fiction','History','Thriller','Horror','Self Help'])]
#cleaned_data=cleaned_data.groupby("Genres1").head(120)
r_labels= cleaned_data[["Genres1"]]
#cleaned_data = cleaned_data[:32]
target_category=['Fantasy','Fiction','Nonfiction']
#target_category=['Fantasy','Historical Fiction','Classics','Young Adult','Mystery','Romance','Science Fiction','History','Thriller','Horror','Self Help']
#r_labels= cleaned_data[["Genres1"]]
#cleaned_data = cleaned_data[:660]
#cleaned_data=cleaned_data[:4]
print(len(cleaned_data))
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
fic =0
nonfic = 0
fant = 0
for index,row in cleaned_data.iterrows():
    title =row["Title"]
    author = row["Author"]
    description=row["Description"]
    description =re.sub(r'[^a-zA-Z ]','',description)
    gen = row["Genres1"]
    tokens = nlp(description.lower())
    t_title = nlp(title.lower())
    t_author= nlp(author.lower())
    #print("Description Length: ",len(description))
    if(len(tokens)<=1000 and len(tokens)>1):
        y.append(gen)
        nsw_tokens = [token.lemma_ for token in tokens if not token.text in all_stopwords and not token.is_punct and not token.is_digit and not token.pos_ == "PROPN"]
        #print("Description without stop words: ",len(nsw_tokens))
        nsw_tokens = [token for token in nsw_tokens if not token in t_title.text]
        #print("Description without title: ",len(nsw_tokens))
        nsw_tokens = [token for token in nsw_tokens if not (token in t_author.text)]
        #nsw_tokens = [token for token in nsw_tokens if len(token) >= 7]
        #print("Description without Author Name: ",len(nsw_tokens))
        if len(nsw_tokens)>350:
            nsw_tokens = nsw_tokens[:350]
        #print(nsw_tokens)
        if fant<10 and gen=="Fantasy":
            writer.add_text(title," ".join(nsw_tokens)+"---"+gen)
            fant+=1
        if fic<10 and gen=="Fiction":
            writer.add_text(title," ".join(nsw_tokens)+"---"+gen)
            fic+=1
        if nonfic<10 and gen=="Nonfiction":
            writer.add_text(title," ".join(nsw_tokens)+"---"+gen)
            nonfic+=1
        #print(len(nsw_tokens))
        #assert len(nsw_tokens)==200
        docs.append(" ".join(nsw_tokens))
        tok_list = Union(tok_list,nsw_tokens)
        dataByClass[row.Genres1]=Union( dataByClass[row.Genres1],nsw_tokens)
        if(index%1000==0):
            print(index,"Titles processed")
#print(len(tok_list))
#print(len(dataByClass["Fiction"]))
#print(len(dataByClass["Nonfiction"]))
#sharedWords=[tok for tok in dataByClass["Fiction"] if tok  in dataByClass["Nonfiction"]]
t_vectors=tfidf.fit_transform(docs)
with gzip.open('3description_vectorsRNN.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
t_vectors = torch.tensor(t_vectors.toarray(),dtype=torch.float32)
doc_terms =[doc.split() for doc in docs]
temp = []
docVectors =[]
word2vec ={}
for doc in doc_terms:
    temp=temp+doc
#print(len(temp))
unique_terms =set(temp)
text_data=list(unique_terms)
term_document =[]
#print(text_data)
print(len(text_data))
label_encode = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse_output=False)
int_enc = label_encode.fit_transform(text_data)
int_enc =int_enc.reshape(len(int_enc),1)
#print(int_enc)
onehot_enc = onehot_encoder.fit_transform(int_enc)

i = 0
for t in text_data:
    term_document.append([])
    for doc in docs:
        term_document[i].append(doc.count(t))
    i+=1
term_document=np.array(term_document)
print(term_document.shape)
tsne = TSNE(n_components=1000, random_state=42,method='exact')
gr_appr =TSNE(n_components=2, random_state=42)
svd = TruncatedSVD(n_components=700, n_iter=1, random_state=42)
vecs = svd.fit_transform(np.array(term_document))
print(vecs.shape)
print(svd.explained_variance_ratio_.sum())
for (w,vec) in zip(text_data,vecs):
    word2vec[w]=vec
with open("3word_embeddings.json", "w") as outfile:
    json.dump({k: v.tolist() for k, v in word2vec.items()}, outfile)
j = 0
for doc in docs:
    doc2Mat.append([])
    for w in doc.split():
        doc2Mat[j].append(word2vec[w])
        #docVectors.append(word2vec[w])
    j+=1
for mt in doc2Mat:
    while(len(mt)<350):
        mt.append(np.zeros(len(vecs[0]),dtype=np.float32))
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
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
x_train, x_test, y_train, y_test =train_test_split(doc2Mat,y_set,test_size=.2,random_state=42)
xg_train, xg_test, yg_train, yg_test =train_test_split(t_vectors,y_set,test_size=.2,random_state=42)
#print(x_train.shape)
input_size = len(vecs[0]) #number of tokens in corpus
hidden_size = 2000
num_classes = len(target_category) #number of distinct genres
num_epoch = 10
sequence_length = 350
num_layers = 2
batch_size = 20

learning_rate = 0.00004

#Dataset
#will add code to read in dataset here

#two hidden layer NeuralNet

def category_from_output(output):
    category_idx = torch.argmax(output).item()
    return target_category[category_idx]
class RNN(nn.Module):
    def __init__(self, input_size,hidden_size,num_layers,num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers= num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length,num_classes)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        out, _ = self.rnn(x.float(),h0.float())
        out = out.reshape(out.shape[0], -1)
        #print(out.shape)
        out = self.fc(out.float())
        #print(out.shape)
        #print(out)
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

        return np.array(X), y
ds = Texts(x_train,y_train, transform=transforms.ToTensor())
train_gen = DataLoader(dataset=ds,batch_size=batch_size,shuffle=True)
test_ds =Texts(x_test,y_test, transform=transforms.ToTensor())
test_gen = DataLoader(dataset=test_ds,batch_size=batch_size,shuffle=False)
model = RNN(input_size,hidden_size,num_layers,num_classes)
criterion = nn.CrossEntropyLoss()
#torch.autograd.set_detect_anomaly(True)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
writer.add_graph(model.to(device),torch.zeros([batch_size,sequence_length,input_size]).to(device))
model = model.to(device)
current_loss = 0
running_correct = 0
all_losses=[]
plot_steps, print_steps = 5,5
n_iters = 10
n_total_steps = len(train_gen)
end_train = False
for epoch in range(num_epoch):
    j = 0
    for i, (des,cat) in enumerate(train_gen):
        #des = pad_sequence(des, batch_first=True, padding_value=0)
        #des=pad_sequence(des)
        des = des.squeeze(1)
        #print(des.shape)
        des = des.to(device)
        #des = F.pad(des, (batch_size,sequence_length,len(text_data)))
        #print(des.shape)
        cat = cat.to(device)
        output = model(des)
        loss = criterion(output,cat)
        #print(output)
       # print(cat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current_loss+=loss.item()
        _, predictions = torch.max(output, 1)
        running_correct +=(predictions == cat).sum().item()
        if (i+1)%plot_steps == 0:
            writer.add_scalar('training loss', current_loss/(plot_steps), (epoch*n_total_steps)+1+i)
            writer.add_scalar('accuracy', running_correct/(plot_steps), (epoch*n_total_steps)+1+i)
            current_loss = 0.0
            running_correct=0.0
        if end_train:
            break
        if(i+1)%print_steps == 0:
            guess = category_from_output(output[-1])
            #print(cat[-1].item())
            correct= "CORRECT" if guess == vec2genre[cat[-1].item()] else f"WRONG ({vec2genre[cat[-1].item()]})"
            print(f"{i+1} {epoch+1} {loss:.4f} / {guess} {correct}")
    if end_train:
        break
labels = []
preds =[]
r_preds=[]
confusion_matrix = torch.zeros(num_classes, num_classes)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    ite = 0
    for i, (des, cat1) in enumerate(test_gen):
        #des=pad_sequence(des)
        des = des.to(device).squeeze(1)
        cat1 = cat1.to(device)
        outputs = model(des)
        _, predictions = torch.max(outputs, 1)
        n_samples += cat1.shape[0]
        n_correct += (predictions == cat1).sum().item()
        class_predictions = [F.softmax(output,dim=0) for output in outputs]
        for clss, prd in zip(cat1,predictions):
            confusion_matrix[clss.long(), prd.long()] += 1
        #print(class_predictions)
        r_preds.append(predictions)
        preds.append(class_predictions)
        labels.append(cat1)
        ite+=1
    r_preds=torch.cat(r_preds)
    preds = torch.cat([torch.stack(batch) for batch in preds])
    labels = torch.cat(labels)
torch.save(model,"RNN3.pt")

#loading pytorch models
# model = torch.load("RNN3.pt")
#model.eval()
acc = 100.0* (n_correct/n_samples)
print(f'accuracy = {acc}')
print(confusion_matrix)
classes = range(len(target_category))
for i in classes:
    labels_i = labels==i
    preds_i = preds[:, i]
    writer.add_pr_curve(str(i),labels_i,preds_i,global_step=0)
writer.close()
Y_Real = gr_appr.fit_transform(xg_test)
fig, (axs1,axs2) = plt.subplots(1,2)
scat1=axs1.scatter(Y_Real[:,0],Y_Real[:,1], c=labels.cpu(),cmap="Paired")
axs1.legend(handles=scat1.legend_elements()[0],
                    loc="lower left", title="Genres",labels=target_category)
axs1.set_title("Actual")
scat2=axs2.scatter(Y_Real[:,0],Y_Real[:,1], c=r_preds.cpu(),cmap="Paired")
axs2.legend(handles=scat2.legend_elements()[0],
                    loc="lower left", title="Genres",labels=target_category)
axs2.set_title("Predicted")
plt.show()
classes = range(len(target_category))
for i in classes:
    labels_i = labels==i
    preds_i = preds[:, i]
    writer.add_pr_curve(str(i),labels_i,preds_i,global_step=0)
writer.close()
