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

print(">>> Transform test data")
df_test = pd.DataFrame({
    "test_id": test["test_id"],
    "q1": list(tfidf.transform(test["question1"])),
    "q2": list(tfidf.transform(test["question2"]))
})

del train
del test
del questions
del tfidf
gc.collect()

print(">>> Save df_test")
# pickle.dump(df_test, open("model/df_test.pkl", "wb"))
df_test.to_hdf("dataframe/df_test.hdf", "df_test")