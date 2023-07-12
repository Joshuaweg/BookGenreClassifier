import re
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import GaussianNB

# Get directory name
if os.path.exists("runs/BGC"):
    shutil.rmtree("runs/BGC")
writer = SummaryWriter("runs/BGC")


def Union(lst1, lst2):
    """Union: merges two lists and removes duplicates values
    :param lst1: list for Union
    :param lst2: list for Union
    return: merged list"""
    final_list = list(set(lst1) | set(lst2))
    return final_list


# Using gitHub url to extract data for use
data = pd.read_csv('https://raw.githubusercontent.com/Joshuaweg/BookGenreClassifier/master/data/genre_data.csv')
# In this dataset, there are 10,000 entries containing book titles,
# their author, a description of the book, a list of genres that the books
# can be categorized under, average rating from the goodreads.com website,
# including the number of reviews, and the URL.
genres = []
count_genre = {}

#  For our purpose, the first step was to narrow down the columns,
#  which came to [Title, Author, Description, List of Genres].
cleaned_data = data[["Title", "Author", "Description", "Genres1"]]
# each book is classified under multiple genres,
# The single Genres column was split to have a separate column for
# each category, totaling 7 Genre columns. Of the 7, we are working
# with the first column.
cleaned_data = cleaned_data[cleaned_data.Genres1.isin(['Fiction', 'Nonfiction'])]
target_category = ['Fiction', 'Nonfiction']
print(len(cleaned_data))

docs = []
dataByClass = {}

# importing stacy en_core_web_lg
import spacy.cli
spacy.cli.download("en_core_web_lg")
nlp = spacy.load('en_core_web_lg')

all_stopwords = nlp.Defaults.stop_words
tok_list = []
genre2vec = {}
vec2genre = {}
sharedWords = []
y = []

for c in target_category:
    dataByClass[c] = []
r = 0

# Data is cleaned by removing author's name, the title of the book from the
#  description. The data is also having stopwords removed and lemmanization
# to reduce the number of words
for index, row in cleaned_data.iterrows():
    title = row["Title"]
    author = row["Author"]
    description = row["Description"]
    description = re.sub(r'[^a-zA-Z ]', '', description)
    gen = row["Genres1"]
    tokens = nlp(description.lower())
    t_title = nlp(title.lower())
    t_author = nlp(author.lower())

    # Range of descriptions is reduced with the help of an analysis of the
    # data with Excel. Histogram was used to help remove outliers to have a
    # better understanding of the data, and it's core information
    if 500 >= len(tokens) > 110:
        y.append(gen)
        nsw_tokens = [token.lemma_ for token in tokens if not token.text in all_stopwords]
        nsw_tokens = [token for token in nsw_tokens if not token in t_title.text]
        nsw_tokens = [token for token in nsw_tokens if not (token in t_author.text)]
        if r < 10:
            print(title, "\n", gen, "\n", description)
        docs.append(" ".join(nsw_tokens))
        if r < 100:
            writer.add_text(title, " ".join(nsw_tokens) + "---" + gen)
            r += 1
        tok_list = Union(tok_list, nsw_tokens)
        dataByClass[row.Genres1] = Union(dataByClass[row.Genres1], nsw_tokens)
        if index % 1000 == 0:
            print(index, "Titles processed")

sharedWords = [tok for tok in dataByClass["Fiction"] if tok in dataByClass["Nonfiction"]]

# Shared words are removed to reduce the number of words processed
for doc in docs:
    doc = [tok.text for tok in nlp(doc) if not tok.text in sharedWords]

i = 0
for g in target_category:
    genre2vec[g] = i
    vec2genre[i] = g
    i += 1
y_set = torch.zeros([len(y)], dtype=torch.long)
l = 0
for g in y:
    if g == "Fiction":
        y_set[l] = 0
    else:
        y_set[l] = 1
    l += 1

# TfidfVectorizer is used to help map the most frequent words to feature
# indices and compute a word occurrence frequency matrix
tfidf = TfidfVectorizer()

# Multinomial Navie Bayes: used to handle text data with
# discrete features such as word frequency counts
nb = MultinomialNB()

# Gaussian Naive Bayes: similar to Multinomial Naive Bayes, but based on the
# probabilistic approach and Gaussian distribution.
gnb = GaussianNB()

# Using TfidfVectorizer, the data uses the fit() method to calculate various required parameters,
# and the transform() method applies calculated parameters to standardize the data.
t_vectors = tfidf.fit_transform(docs)
t_vectors = torch.tensor(t_vectors.toarray(), dtype=torch.float32)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(t_vectors, y_set, test_size=.2, random_state=42)

# Fitting MultinomialNB
mnb = nb.fit(x_train, y_train)
predictedMNB = nb.predict(x_test)
accMNB = accuracy_score(predictedMNB, y_test)
f1MNB = f1_score(predictedMNB, y_test, average="weighted")
cmatrixMNB = confusion_matrix(y_test, predictedMNB)

print(f"MultinomialNB Accuracy Score: {accMNB}")
print(f"MultinomialNB f1_score: {f1MNB}")
print(f"MultinomialNB confusion matrix: {cmatrixMNB}")

# Fitting GaussianNB
gnb.fit(x_train, y_train)
predictedGNB = gnb.predict(x_test)
accuracyGNB = accuracy_score(predictedGNB, y_test)
f1GNB = f1_score(predictedGNB, y_test, average="weighted")
cmatrixGNB = confusion_matrix(y_test, predictedGNB)

print(f"GaussianNB Accuracy Score: {accuracyGNB}")
print(f"GaussianNB f1_score: {f1GNB}")
print(f"GaussianNB confusion matrix: {cmatrixGNB}")

# F1 score can be interpreted as a measure of overall model performance
# from 0 to 1, where 1 is the best. To be more specific, F1 score can be
# interpreted as the model's balanced ability to both capture positive
# cases (recall) and be accurate with the cases
# it does capture (precision).
