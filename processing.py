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
token_vector = np.zeros(shape=(len(tok_list),len(tok_list)))
tok_pos = []
for i in range(len(tok_list)):
    t_vector = np.zeros(len(tok_list))
    t_vector[i] = 1.0
    token_vector[i] = t_vector
    tok_pos.append(i)


    
