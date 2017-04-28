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
df_train = pd.read_hdf("dataframe/df_train.hdf", "df_train")

print(">>> Start session")
sess = tf.InteractiveSession()

NUM_INPUT_UNITS = df_train.q1[0].shape[1] * 2
NUM_HID1_UNITS = 1024
NUM_HID2_UNITS = 512
NUM_OUTPUT = 2

BATCH_SIZE = 1000

print(">>> Make nn")
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

xentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
train_step = tf.train.AdamOptimizer(1e-3).minimize(xentropy)

saver = tf.train.Saver()

tf.global_variables_initializer().run()
num_data = df_train.shape[0]
steps = int(np.ceil(num_data / BATCH_SIZE))
EPOCH = 30
print(">>> Start training")
for epoch in range(EPOCH):
    print(">>> EPOCH", epoch)
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

    print("--- Saved ---")
    saver.save(sess, "model/nn_simple.ckpt")


print(">>> Cost =", cost)

del _x
del _y
gc.collect()


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
