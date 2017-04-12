# training
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

print(">>> Load data")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

vocab = pickle.load(open("vocab/vocab_all.pkl", "rb"))

train["question1"] = train["question1"].fillna("")
train["question2"] = train["question2"].fillna("")
test["question1"] = test["question1"].fillna("")
test["question2"] = test["question2"].fillna("")

print(">>> Fit TF-IDF model")
questions = list(set(train["question1"] + train["question2"] + test["question1"] + test["question2"]))
questions = [q for q in questions if str(q) != 'nan']

vectorizer = TfidfVectorizer(
                vocabulary=vocab, 
                stop_words="english"
            )
tfidf = vectorizer.fit(questions)

print(">>> Transform data")
print("train - question1")
train_q1 = tfidf.transform(train["question1"])

print("train - question2")
train_q2 = tfidf.transform(train["question2"])

print("test  - question1")
test_q1 = tfidf.transform(test["question1"])

print("test  - question2")
test_q2 = tfidf.transform(test["question2"])



print("Done")