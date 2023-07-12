"""
Graphical User Interface for Book Genre Classifier
Program allows users to enter a book description and 
check how book will be classified from 4 different models
"""
import tkinter as tk
from tkinter import messagebox
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
cat_3 = ["Fiction","NonFiction","Fantasy"]
rcat_3 =['Fantasy','Fiction','NonFiction']
vectorizer = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nlp = spacy.load('en_core_web_lg')
all_stopwords = nlp.Defaults.stop_words
sequence_length=350
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

class RNN(nn.Module):
    """ class for recurrent neural network """
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
""" loading models for classification"""
RNNmodel = torch.load("RNN3.pt",map_location ='cpu')
RNNmodel.eval()
NNmodel = torch.load("NN3.pt",map_location ='cpu')
NNmodel.eval()
NBmodel=None
SVMmodel=None
with open('NaiveBayes.pkl', 'rb') as ifp:
    NBmodel = pickle.load(ifp)
with open('svm.pkl', 'rb') as ifp:
    SVMmodel = pickle.load(ifp)
class BookGUI:
    """ class for graphical user interface """
    def __init__(self):

        self.root = tk.Tk()
        self.root.title("Book Genre Classifier")
        self.root.geometry("700x460")
        self.background=tk.PhotoImage(file="books_pic.png") # background image
        self.background_label = tk.Label(self.root, image=self.background)
        self.background_label.place(x=0, y=0)
        self.label = tk.Label(self.root, text="Description", font=('Arial', 18), bg="AntiqueWhite1")
        self.label.pack(padx=10, pady=10)
        self.textbox = tk.Text(self.root, height=5, font=("Arial", 14)) # description entry box
        self.textbox.pack(padx=20, pady=10)
        self.genre_label = tk.Label(self.root, text="Enter Genre Here:", font=("Arial", 16), bg="AntiqueWhite1")
        self.genre_label.pack(padx=10, pady=10)
        self.variable = tk.StringVar(self.root)
        self.variable.set("Fiction")
        self.genre = tk.OptionMenu(self.root, self.variable, "Fiction", "NonFiction", "Fantasy") # drop down for genres
        self.genre.config(bg="AntiqueWhite1")
        self.genre.pack(padx=10, pady=10)
        self.model_var = tk.IntVar()
        self.NN = tk.Radiobutton(self.root, text="Feed Forward", variable=self.model_var, value=1, bg="AntiqueWhite1")
        self.NN.pack()
        self.RNN = tk.Radiobutton(self.root, text="RNN", variable=self.model_var, value=2, bg="AntiqueWhite1")
        self.RNN.pack()
        self.NB = tk.Radiobutton(self.root, text="Naive Bayes", variable=self.model_var, value=3, bg="AntiqueWhite1")
        self.NB.pack()
        self.SVM = tk.Radiobutton(self.root, text="Support Vector Machine", variable=self.model_var, value=4, bg="AntiqueWhite1")
        self.SVM.pack()
        self.button = tk.Button(self.root, text="Enter", font=("Arial", 18), bg="SkyBlue1", command=self.get_model)
        self.button.pack(padx=10, pady=20)
        self.root.mainloop()

    def get_classNN(self):
        """ function to get classification from NN """
        with gzip.open('3description_vectorsNN.pkl', 'rb') as ifp:
            tfidf = pickle.load(ifp)
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
        result = cat_3[predictions.item()]
        option = self.variable.get()
        if result == option:
            messagebox.showinfo(title="results", message="Prediction:"+result+"\nActual:"+option+"\nMatch")
        else:
            messagebox.showinfo(title="results", message="Prediction:"+result+"\nActual:"+option+"\nNo Match")

    def get_model(self):
        """ funtion to get the model user wants to use """
        model_to_use = self.model_var.get()
        if model_to_use == 1:
            self.get_classNN()
        elif model_to_use == 2:
            self.get_classRNN()
        elif model_to_use == 3:
            self.get_classNB()
        elif model_to_use == 4:
            self.get_classSVM()

    def get_classRNN(self):
        """ function to get RNN classification"""
        word_3 =None
        with open("3word_embeddings.json") as json_file:
            word_3 = json.load(json_file)
        description = self.textbox.get('1.0', tk.END)
        description ==re.sub(r'[^a-zA-Z ]','',description)
        embeddings=[]
        tokens = nlp(description)
        nsw_tokens = [token.lemma_ for token in tokens if not token.text in all_stopwords]
        if len(nsw_tokens)>350:
            nsw_tokens = nsw_tokens[:350]
        for t in nsw_tokens:
            if t in word_3.keys():
                embeddings.append(word_3[t])
        while len(embeddings)< 350:
            embeddings.append(np.zeros(len(embeddings[0]),dtype=np.float32))
        embeddings = torch.tensor(embeddings,dtype=torch.float32)
        embeddings = embeddings.unsqueeze(0)
        embeddings = embeddings.to(device)
        output = RNNmodel(embeddings);
        print(output)
        _, predictions = torch.max(output, 1)
        #print(predictions)
        print(rcat_3[predictions.item()])
        result = rcat_3[predictions.item()]
        option = self.variable.get()
        if result == option:
            messagebox.showinfo(title="results", message="Prediction:"+result+"\nActual:"+option+"\nMatch")
        else:
            messagebox.showinfo(title="results", message="Prediction:"+result+"\nActual:"+option+"\nNo Match")

    def get_classNB(self):
        """function to get naive bayes classification """
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
        result = cat_3[output[0]]
        option = self.variable.get()
        if result == option:
            messagebox.showinfo(title="results", message="Prediction:"+result+"\nActual:"+option+"\nMatch")
        else:
            messagebox.showinfo(title="results", message="Prediction:"+result+"\nActual:"+option+"\nNo Match")

    def get_classSVM(self):
        """ function to get classification from support vector machine"""
        with open('SVM_vectorizer.pkl', 'rb') as ifp:
            tfidf = pickle.load(ifp)
        description = self.textbox.get('1.0', tk.END)
        description ==re.sub(r'[^a-zA-Z ]','',description)
        tokens = nlp(description)
        nsw_tokens = [token.lemma_ for token in tokens if not token.text in all_stopwords]
        d_vectors=tfidf.transform([" ".join(nsw_tokens)])
        d_vectors = torch.tensor(d_vectors.toarray(),dtype=torch.float32)
        output = SVMmodel.predict(d_vectors)
        print(output)
        print(cat_3[output[0]])
        result = cat_3[output[0]]
        option = self.variable.get()
        if result == option:
            messagebox.showinfo(title="results", message="Prediction:"+result+"\nActual:"+option+"\nMatch")
        else:
            messagebox.showinfo(title="results", message="Prediction:"+result+"\nActual:"+option+"\nNo Match")


BookGUI()