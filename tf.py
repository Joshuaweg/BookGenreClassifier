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
    print(len(data.keys()))
    for t in data.keys():
        docs.append(" ".join(data[t]["tokens"]))
        print(len(docs))
    results = tfidf.fit_transform(docs)
    print(tfidf.vocabulary_)
    print(results)
    print(results.toarray())

