import gc
import pandas as pd
import _pickle as pickle

print(">>> Load data")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train["question1"] = train["question1"].fillna("")
train["question2"] = train["question2"].fillna("")
test["question1"] = test["question1"].fillna("")
test["question2"] = test["question2"].fillna("")

print(">>> Fit TF-IDF model")
questions = list(set(train["question1"] + train["question2"] + test["question1"] + test["question2"]))
questions = [q for q in questions if str(q) != 'nan']
tfidf = pickle.load(open("model/tfidf_1024_model.pkl", "rb"))
tfidf = tfidf.fit(questions)

print(">>> Transform train data")
df_train = pd.DataFrame({
    "q1": list(tfidf.transform(train["question1"])),
    "q2": list(tfidf.transform(train["question2"])),
    "dup": train["is_duplicate"]
})

# del vocab
del questions
del tfidf
del train
del test
gc.collect()

print(">>> Save df_train")
pickle.dump(df_train, open("dataframe/df_train.pkl", "wb"))