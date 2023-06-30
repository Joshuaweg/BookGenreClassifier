import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import spacy
from sklearn.model_selection import train_test_split
import nltk

factor1 = 200
neuron1 = 8000
neuron2 = 4000

url = 'https://raw.githubusercontent.com/Joshuaweg/BookGenreClassifier/master/data/genre_data.csv'
db_original = pd.read_csv(url)
# Dataset is now stored in a Pandas Dataframe

# List of database headers
db_original_headers = db_original.columns.values.tolist()

db = db_original[["Title","Author","Description", "Genres1"]]

db_fic_nonfic = db[db.Genres1.isin(["Fiction", "Nonfiction"])]

target_category=["Fiction", "Nonfiction"]

db_headers = db_fic_nonfic.columns.values.tolist()

# Altering database to contain lowercase characters
db_fic_nonfic = db_fic_nonfic.apply(lambda x: x.astype(str).str.lower())

# Removing Title from within Description
db_fic_nonfic['Description'] = db_fic_nonfic.apply(lambda row: row['Description'].replace(str(row['Title']), ''), axis=1)
#db_fic_nonfic['Description2'] == db_fic_nonfic['Description']

# Removing Author from within Description
db_fic_nonfic['Description'] = db_fic_nonfic.apply(lambda row: row['Description'].replace(str(row['Author']), ''), axis=1)

# Removing following text from column
db_fic_nonfic['Description'] = db_fic_nonfic.apply(lambda row: row['Description'].replace('new york times', ''), axis=1)
db_fic_nonfic['Description'] = db_fic_nonfic.apply(lambda row: row['Description'].replace('bestselling', ''), axis=1)
db_fic_nonfic['Description'] = db_fic_nonfic.apply(lambda row: row['Description'].replace('author', ''), axis=1)
db_fic_nonfic['Description'] = db_fic_nonfic.apply(lambda row: row['Description'].replace('book year', ''), axis=1)


# Dropping columns Title and Author
db_fic_nonfic = db_fic_nonfic.drop(columns=['Title', 'Author'])

# Removing numerical values from Description column
db_fic_nonfic['Description'] = db_fic_nonfic['Description'].str.replace('\d+', '', regex=True)
# Removign punctuations
import re
db_fic_nonfic['Description']=[re.sub('[^\w\s]+', '', s) for s in db_fic_nonfic['Description'].tolist()]

import nltk
from nltk.corpus import stopwords
stopword_package = 'stopwords'

nltk.download('stopwords')
stop = stopwords.words("english")
# Removing stopwords from Descriptions
db_fic_nonfic.Description = db_fic_nonfic.Description.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# Remove proper names ????

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
  words = text.split()
  words = [lemmatizer.lemmatize(word, pos='v') for word in words]
  return ' '.join(words)
db_fic_nonfic['Description'] = db_fic_nonfic.Description.apply(lemmatize_words)

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english') # check if stopwords are removed
#db_fic_nonfic['Description'] = db_fic_nonfic['Description'].apply(lambda x: [stemmer.stem(y) for y in x])
db_fic_nonfic['Description'].apply(lambda x: stemmer.stem(x))

from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer(ngram_range=(1, 5),
                        max_features=factor1,
                        stop_words='english')
matrix = model.fit_transform(db_fic_nonfic['Description']).toarray()
df_output = pd.DataFrame(data=matrix, columns=model.vocabulary_.keys())
features_tensor = torch.tensor(df_output.values).to(torch.float32)

# Convert to tensors and split into train and test sets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# Using target_category as labels [Fiction, Nonfiction]
le = preprocessing.LabelEncoder()

labels = le.fit_transform(db_fic_nonfic.Genres1)
labels_tensor = torch.as_tensor(labels).to(torch.float32)
labels_tensor.shape

features_tensor_train, features_tensor_test, labels_tensor_train, labels_tensor_test = train_test_split(features_tensor.unsqueeze(dim=1),
                                                                                                        labels_tensor,
                                                                                                        test_size=0.3,
                                                                                                        random_state=85)

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

# Model class that subclasses nn.Module
class Cateogrizer(nn.Module):
  def __init__(self):
    super().__init__()
    # Creating nn.Linear layers capable of handling features_tensor and labels_tensor
    #self.layer_1 = nn.Linear(in_features=2, out_features=800)
    #self.layer_2 = nn.Linear(in_features=800, out_features=2)
    self.layer_1 = nn.Sequential(
        nn.Linear(in_features=factor1, out_features=neuron1),
        #nn.ReLU(inplace=True), #inplace (bool) – can optionally do the operation in-place. Default: False
        nn.ReLU(),
       #nn.Linear(in_features=8000, out_features=neuron2),
        #nn.ReLU(),
        nn.Linear(in_features=neuron1, out_features=1)
    )


    # Forward method containing the forward pass computation
  def forward(self, x):
    # Return the output of layer_2, a single feature, the same shape as y
    return self.layer_1(x)

model_0 = Cateogrizer().to(device)
model_0

# Make predictions with the model
untrained_preds = model_0(features_tensor_test.to(device))
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(labels_tensor_test)}, Shape: {labels_tensor_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 test labels:\n{labels_tensor_test[:10]}")

# Create a loss function
loss_fn = nn.BCEWithLogitsLoss()

# Create optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.03)

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc

# View the frist 5 outputs of the forward pass on the test data
labels_logits = model_0(features_tensor_test.to(device))[:5]

labels_pred_probs = torch.relu(labels_logits)

labels_preds = torch.round(labels_pred_probs)

labels_pred_labels = torch.round(torch.relu(model_0(features_tensor_test.to(device))[:5]))
print(torch.eq(labels_preds.squeeze(), labels_pred_labels.squeeze()))

from scipy import optimize
# Building a training and testing loop
torch.manual_seed(85)

# Number of epochs
epochs = 100

# Data inserted into target device
features_tensor_train, labels_tensor_train = features_tensor_train.to(device), labels_tensor_train.to(device)
features_tensor_test, labels_tensor_test = features_tensor_test.to(device), labels_tensor_test.to(device)

# Building training and evaluation loop
for epoch in range(epochs):
  # Training
  model_0.train()

  # Forward pass (model outputs raw logits)
  labels_logits = model_0(features_tensor_train).squeeze()
  labels_pred = torch.round(torch.relu(labels_logits))

  # Calculate loss/accuracy
  loss = loss_fn(labels_logits,
                 labels_tensor_train)
  acc = accuracy_fn(y_true=labels_tensor_train,
                    y_pred=labels_pred)

  # Optimizer zero grad
  optimizer.zero_grad()

  # Loss backwards
  loss.backward()

  # Optimizer step
  optimizer.step()

  # Testing
  model_0.eval()
  with torch.inference_mode():
    # Forward pass
    test_logits = model_0(features_tensor_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))
    # Calculate loss/accuracy
    test_loss = loss_fn(test_logits,
                        labels_tensor_test)
    test_acc = accuracy_fn(y_true=labels_tensor_test,
                           y_pred=test_pred)

    # Printing whats happening in batches
    if epoch % 10 == 0:
      print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

