def getRandomIndices(num, max_index):
    if (num is None or max_index is None or \
        num <= 0 or max_index <= 0 or max_index < num):
        return []
    ret = list()
    for _ in range(num):
        rand = random.randint(0, max_index)
        while (rand in ret):
            rand = random.randint(0, max_index)
        ret.append(rand)
    return ret


def getRandomBatch(num, frame): 
    if num is None or frame is None:
        return ([], [])
    max_index = frame.shape[0] - 1
    indices = getRandomIndices(num, max_index)
    return getBatchByIndices(indices, frame)


def getBatchByIndices(indices, frame, labeled=True):
    if indices is None or frame is None:
        return ([], [])
    x = list()
    y = list()
    for i in indices:
        _q1 = frame.q1[i].toarray()[0]
        _q2 = frame.q2[i].toarray()[0]
        x.append(np.concatenate((_q1, _q2)))
        if labeled:
            y.append([0, 1] if frame.dup[i] == 1 else [1, 0])
    return (x, y) if labeled else x


# training
import gc
import random
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

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

print("Restore saved model")
NUM_INPUT_UNITS = df_test.q1[0].shape[1] * 2
NUM_HID1_UNITS = 1024
NUM_HID2_UNITS = 512
NUM_OUTPUT = 2

BATCH_SIZE = 1000

x = tf.placeholder(tf.float32, [None, NUM_INPUT_UNITS])
y = tf.placeholder(tf.float32, [None, NUM_OUTPUT])
keep_prob = tf.placeholder(tf.float32)

hid1_W = tf.Variable(tf.random_normal([NUM_INPUT_UNITS, NUM_HID1_UNITS]))
hid1_b = tf.Variable(tf.random_normal([NUM_HID1_UNITS]))
hid1 = tf.nn.sigmoid(tf.matmul(x, hid1_W) + hid1_b)
hid1_drop = tf.nn.dropout(hid1, keep_prob)

hid2_W = tf.Variable(tf.random_normal([NUM_HID1_UNITS, NUM_HID2_UNITS]))
hid2_b = tf.Variable(tf.random_normal([NUM_HID2_UNITS]))
hid2 = tf.nn.sigmoid(tf.matmul(hid1_drop, hid2_W) + hid2_b)
hid2_drop = tf.nn.dropout(hid2, keep_prob)

out_W = tf.Variable(tf.random_normal([NUM_HID2_UNITS, NUM_OUTPUT]))
out_b = tf.Variable(tf.random_normal([NUM_OUTPUT]))
output = tf.matmul(hid2_drop, out_W) + out_b

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "model/nn_simple.ckpt")

print("Predict")
num_data = df_test.shape[0]
_BATCH_SIZE = 10000
steps = int(np.ceil(num_data / _BATCH_SIZE))
predictions = list()
for step in range(steps):
    start = step * _BATCH_SIZE
    end = start + _BATCH_SIZE
    if end > num_data:
        end = num_data
    indices = np.array(range(start, end))
    _x = getBatchByIndices(indices, df_test, labeled=False)
    pred = sess.run(tf.nn.softmax(output), feed_dict={x: _x, keep_prob: 1.0})
    predictions = predictions + pred[:,1].tolist()
    del _x
    _ = gc.collect()
    if step % 10 == 0:
        print("Step", step, "/", steps)


df_predictions = pd.DataFrame({
    "test_id": df_test["test_id"],
    "is_duplicate": pd.Series(predictions)
})

print("Num duplicates", sum(df_predictions.is_duplicate > 0.5))
print("Num not duplicates", sum(df_predictions.is_duplicate <= 0.5))
print("Average", df_predictions.is_duplicate.mean())


df_predictions.to_csv("submission/nn_simple_mod.csv", index=False)
print("Done")

