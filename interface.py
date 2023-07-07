import tkinter as tk
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

tfidf = TfidfVectorizer()
nlp = spacy.load('en_core_web_lg')

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
        self.button = tk.Button(self.root, text="Enter", font=("Arial", 18), command=self.get_description)
        self.button.pack(padx=10, pady=10)
        self.root.mainloop()

    def get_description(self):
        description = self.textbox.get('1.0', tk.END)
        d_tokens = description.split()
        d_vectors=tfidf.fit_transform(d_tokens)
        d_vectors = torch.tensor(d_vectors.toarray(),dtype=torch.float32)
        print(d_vectors)





BookGUI()