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
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
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
num_vocabs = tfidf.max_features

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


sess = tf.InteractiveSession()

MAX_FEATURES = num_vocabs
NUM_HID_UNITS_1 = 1024
NUM_HID_UNITS_2 = 512
NUM_OUTPUT = 2

BATCH_SIZE = 1000

x = tf.placeholder(tf.float32, [None, MAX_FEATURES * 2])
y = tf.placeholder(tf.float32, [None, NUM_OUTPUT])
keep_prob = tf.placeholder(tf.float32)

hid1_W = tf.Variable(tf.random_normal([MAX_FEATURES * 2, NUM_HID_UNITS_1]))
hid1_b = tf.Variable(tf.random_normal([NUM_HID_UNITS_1]))
hid1 = tf.nn.sigmoid(tf.matmul(x, hid1_W) + hid1_b)
hid1_drop = tf.nn.dropout(hid1, keep_prob)

hid2_W = tf.Variable(tf.random_normal([NUM_HID_UNITS_1, NUM_HID_UNITS_2]))
hid2_b = tf.Variable(tf.random_normal([NUM_HID_UNITS_2]))
hid2 = tf.nn.sigmoid(tf.matmul(hid1_drop, hid2_W) + hid2_b)
hid2_drop = tf.nn.dropout(hid2, keep_prob)

out_W = tf.Variable(tf.random_normal([NUM_HID_UNITS_2, NUM_OUTPUT]))
out_b = tf.Variable(tf.random_normal([NUM_OUTPUT]))
output = tf.matmul(hid2_drop, out_W) + out_b

xentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
train_step = tf.train.AdamOptimizer(1e-2).minimize(xentropy)


tf.global_variables_initializer().run()
num_data = df_train.shape[0]
steps = int(np.ceil(num_data / BATCH_SIZE))
EPOCH = 1
for _ in range(EPOCH):
    for step in range(steps):
        start = step * BATCH_SIZE
        end = start + BATCH_SIZE
        if end > num_data:
            end = num_data
        indices = np.array(range(start, end))
        _x, _y = getBatchByIndices(indices, df_train)
        _, cost = sess.run([train_step, xentropy], feed_dict={x: _x, y: _y, keep_prob: 0.5})
        if step % 100 == 0:
            print("Step", step, " Start", start, " End", (end - 1), " Cost", cost)


print("Cost =", cost)

del _x
del _y
gc.collect()

saver = tf.train.Saver()
saver.save(sess, "model/nn_simple.ckpt")


correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
_x, _y = getRandomBatch(10000, df_train)
print("Accuracy =", sess.run(accuracy, feed_dict={x: _x, y: _y, keep_prob: 1.0}))

sess.close()
del sess
del _x
del _y
gc.collect()

print("Done")
