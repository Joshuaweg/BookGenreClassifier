#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
sns.set(color_codes=True)
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pickle


from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer

def train():
    """
    This will train a Support Vector Machine to Classify Description
    into Genres
    prints out accuracy, F1-Scores
    """
    df = pd.read_csv('data\\genre_data.csv')

    df = df.loc[df['Genres1'].isin(['Nonfiction', 'Fiction', 'Fantasy'])]



    df = df.reset_index()




    df = df.astype({'Description':'string','Genres1':'string'})



    print('% of Missing Values for each Columns')
    print('======================================')
    df.apply(lambda x :(x.isnull().mean())*100)


#import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('stopwords')



# Step - a : Remove blank rows if any.
    df['Description'].dropna(inplace=True)
# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    df['Description'] = [entry.lower() for entry in df['Description']]
# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
    df['Description']= [word_tokenize(entry) for entry in df['Description']]

# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun

# Remove Stop words, Non-Numeric and Word Stemming/Lemmenting.
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index,entry in enumerate(df['Description']):
        if index % 500 == 0:
            print(index,"Titles processed")
    # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
    # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
        df.loc[index,'desc_final'] = str(Final_words)

    le = LabelEncoder()
    df['Genres1'] = le.fit_transform(df['Genres1'])
    print(le.classes_)

    x_train, x_test, y_train, y_test = train_test_split(df['desc_final'], df['Genres1'], test_size = 0.2, shuffle=True)

# vectorize final descriptions
    Tfidf_vect = TfidfVectorizer(max_features=2000)
    Tfidf_vect.fit(df['desc_final'])
    filename = 'SVM_vectorizer.pkl'
    pickle.dump(Tfidf_vect, open(filename, 'wb'))

    Train_X_Tfidf = Tfidf_vect.transform(x_train)
    Test_X_Tfidf = Tfidf_vect.transform(x_test)
# define and train model
# print results 
    model = svm.SVC(probability=True)
    model.fit(Train_X_Tfidf, y_train)
    filename = 'svm.pkl'
    pickle.dump(model, open(filename, 'wb'))
    svm_pred = model.predict(Test_X_Tfidf)
    print(classification_report(y_test, svm_pred))




    print(le.classes_)

if __name__=="__main__":
    train()






