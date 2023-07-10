import tkinter as tk
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import pickle
import gzip
import re
from sklearn.naive_bayes import MultinomialNB
import json
import numpy as np

tfidf = TfidfVectorizer()
nlp = spacy.load('en_core_web_lg')
cat_3 = ["Fiction","Nonfiction","Fantasy"]
rcat_3 =['Fantasy','Fiction','Nonfiction']
vectorizer = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nlp = spacy.load('en_core_web_lg')
all_stopwords = nlp.Defaults.stop_words
sequence_length=300
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

class RNN(nn.Module):
    def __init__(self, input_size,hidden_size,num_layers,num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers= num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length,num_classes)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        print(x.shape)
        print(h0.shape)
        out, _ = self.rnn(x.float(),h0.float())
        out = out.reshape(out.shape[0], -1)
        #print(out.shape)
        out = self.fc(out.float())
        #print(out.shape)
        #print(out)
        return out
RNNmodel = torch.load("RNN3.pt",map_location ='cpu')
RNNmodel.eval()
NNmodel = torch.load("NN3.pt",map_location ='cpu')
NNmodel.eval()
NBmodel=None
with open('NaiveBayes.pkl', 'rb') as ifp:
    NBmodel = pickle.load(ifp)
with gzip.open('3description_vectors.pkl', 'rb') as ifp:
    tfidf = pickle.load(ifp)
class BookGUI:
    def __init__(self):

        self.root = tk.Tk()
        self.root.title("Book Genre Classifier")
        self.label = tk.Label(self.root, text="Description", font=('Arial', 18))
        self.label.pack(padx=10, pady=10)
        self.textbox = tk.Text(self.root, height=5, font=("Arial", 16))
        self.textbox.pack(padx=10)
        self.genre_label = tk.Label(self.root, text="Enter Genre Here:", font=("Arial", 16))
        self.genre_label.pack(padx=10, pady=10)
        self.variable = tk.StringVar(self.root)
        self.variable.set("Fiction")
        self.genre = tk.OptionMenu(self.root, self.variable, "Ficton", "NonFiction", "Fantasy")
        self.genre.pack(padx=10, pady=10)
        self.button = tk.Button(self.root, text="Enter", font=("Arial", 18), command=self.get_classNN)
        self.button.pack(padx=10, pady=10)
        self.root.mainloop()

    def get_classNN(self):
        description = self.textbox.get('1.0', tk.END)
        description ==re.sub(r'[^a-zA-Z ]','',description)
        tokens = nlp(description)
        nsw_tokens = [token.lemma_ for token in tokens if not token.text in all_stopwords]
        d_vectors=tfidf.transform([" ".join(nsw_tokens)])
        d_vectors = torch.tensor(d_vectors.toarray(),dtype=torch.float32)
        output = NNmodel(d_vectors);
        print(output)
        _, predictions = torch.max(output, 1)
        #print(predictions)
        print(cat_3[predictions.item()])
    def get_classRNN(self):
        word_3 =None
        with open("3word_embeddings.json") as json_file:
            word_3 = json.load(json_file)
        description = self.textbox.get('1.0', tk.END)
        description ==re.sub(r'[^a-zA-Z ]','',description)
        embeddings=[]
        tokens = nlp(description)
        nsw_tokens = [token.lemma_ for token in tokens if not token.text in all_stopwords]
        if len(nsw_tokens)>300:
            nsw_tokens = nsw_tokens[:300]
        for t in nsw_tokens:
            if t in word_3.keys():
                embeddings.append(word_3[t])
        while len(embeddings)< 300:
            embeddings.append(np.zeros(len(embeddings[0]),dtype=np.float32))
        embeddings = torch.tensor(embeddings,dtype=torch.float32)
        embeddings = embeddings.unsqueeze(0)
        embeddings = embeddings.to(device)
        output = RNNmodel(embeddings);
        print(output)
        _, predictions = torch.max(output, 1)
        #print(predictions)
        print(rcat_3[predictions.item()])
    def get_classNB(self):
        with gzip.open('NBdescription_vectors.pkl', 'rb') as ifp:
            tfidf = pickle.load(ifp)
        description = self.textbox.get('1.0', tk.END)
        description ==re.sub(r'[^a-zA-Z ]','',description)
        tokens = nlp(description)
        nsw_tokens = [token.lemma_ for token in tokens if not token.text in all_stopwords]
        d_vectors=tfidf.transform([" ".join(nsw_tokens)])
        d_vectors = torch.tensor(d_vectors.toarray(),dtype=torch.float32)
        output = NBmodel.predict(d_vectors)
        print(output)
        print(cat_3[output[0]])





BookGUI()