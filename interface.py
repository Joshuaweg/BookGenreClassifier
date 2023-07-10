import tkinter as tk
from tkinter import messagebox
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

tfidf = TfidfVectorizer()
nlp = spacy.load('en_core_web_lg')

class BookGUI:
    def __init__(self):
        super().__init__()
        self.root = tk.Tk()
        self.root.title("Book Genre Classifier")
        self.root.geometry("700x460")
        self.background=tk.PhotoImage(file="books_pic.png")
        self.background_label = tk.Label(self.root, image=self.background)
        self.background_label.place(x=0, y=0)
        self.label = tk.Label(self.root, text="Description", font=('Arial', 18), bg="AntiqueWhite1")
        self.label.pack(padx=10, pady=10)
        self.textbox = tk.Text(self.root, height=5, font=("Arial", 14))
        self.textbox.pack(padx=20, pady=10)
        self.genre_label = tk.Label(self.root, text="Enter Genre Here:", font=("Arial", 16), bg="AntiqueWhite1")
        self.genre_label.pack(padx=10, pady=10)
        self.variable = tk.StringVar(self.root)
        self.variable.set("Fiction")
        self.genre = tk.OptionMenu(self.root, self.variable, "Fiction", "NonFiction", "Fantasy")
        self.genre.config(bg="AntiqueWhite1")
        self.genre.pack(padx=10, pady=10)
        self.var = tk.IntVar()
        self.NN = tk.Radiobutton(self.root, text="Feed Forward", variable=self.var, value=1, bg="AntiqueWhite1")
        self.NN.pack()
        self.RNN = tk.Radiobutton(self.root, text="RNN", variable=self.var, value=2, bg="AntiqueWhite1")
        self.RNN.pack()
        self.NB = tk.Radiobutton(self.root, text="Naive Bayes", variable=self.var, value=3, bg="AntiqueWhite1")
        self.NB.pack()
        self.button = tk.Button(self.root, text="Enter", font=("Arial", 18), bg="SkyBlue1", command=self.get_description)
        self.button.pack(padx=10, pady=30)
        self.root.mainloop()

    def get_description(self):
        description = self.textbox.get('1.0', tk.END)
        #d_tokens = description.split()
        #d_vectors=tfidf.transform(description)
        #d_vectors = torch.tensor(d_vectors.toarray(),dtype=torch.float32)
        result= "Fiction"
        option = self.variable.get()
        print(result)
        print(option)
        if result == option:
            messagebox.showinfo(title="results", message="results are correct")
        else:
            messagebox.showinfo(title="results", message="results are incorrect")
BookGUI()