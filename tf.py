import json
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list
docs = []
tfidf = TfidfVectorizer()
with open('books.json','r') as books:
    data=json.load(books)
    #print(len(data.keys()))
    for t in data.keys():
        docs.append(" ".join(data[t]["tokens"]))
        #print(len(docs))
    results = tfidf.fit_transform(docs)
    print(len(tfidf.vocabulary_))
    print(results)
    print(results.toarray()[0][:200])
    print(results.toarray()[0][200:400])
    print(results.toarray()[0][400:600])
    print(results.toarray()[0][600:800])
    print(results.toarray()[0][800:1000])
    print(results.toarray()[0][1000:1200])
    print(results.toarray()[0][1200:1400])

