import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fname_dat = 'dat/world_cases.csv'
n_epochs = 10000

df = pd.read_csv(fname_dat, sep=',')
t_dat = df['date_diff'].to_numpy()
y0_dat = df['sum'].to_numpy()

t_dat = t_dat.reshape([t_dat.size, 1]).astype(np.float32)
y0_dat = y0_dat.reshape([y0_dat.size, 1]).astype(np.float32)


def triple_layer(t, n1=10, loc1=0, n2=5, loc2=0, n3=1, loc3=0):

    in_dim = t.get_shape().as_list()[1]
    w1 = tf.Variable(tf.random_normal([in_dim, n1], mean=loc1, stddev=loc1 * 0.1))
    b1 = tf.Variable(tf.random_normal([n1]))
    h1 = tf.matmul(t, w1) + b1
    h1 = tf.nn.relu(h1)

    in_dim = h1.get_shape().as_list()[1]
    w2 = tf.Variable(tf.random_normal([in_dim, n2], mean=loc2, stddev=loc2 * 0.1))
    b2 = tf.Variable(tf.random_normal([n2]))
    h2 = tf.matmul(h1, w2) + b2
    h2 = tf.nn.relu(h2)

    in_dim = h2.get_shape().as_list()[1]
    w3 = tf.Variable(tf.random_normal([in_dim, 1], mean=loc3, stddev=loc3 * 0.1))
    b3 = tf.Variable(tf.random_normal([1]))
    h3 = tf.matmul(h2, w3) + b3
    h3 = tf.nn.relu(h3)

    return h3


t = tf.placeholder(tf.float32, [None, 1])
t0 = tf.Variable(10.)
a = triple_layer(t - t0, loc1=2, loc2=10, loc3=1, n3=1)
r0 = tf.Variable(tf.random_uniform([1,], 1, 1.4))
b = triple_layer(t - t0, n3=1)

y = a * (r0 ** b)

loss = tf.reduce_mean(tf.squared_difference(y, y0_dat))

optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
optimizer = optimizer.minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(n_epochs):
    _, current_loss = sess.run([optimizer, loss], feed_dict={t: t_dat})
    print('Epoch {}/{}, L = {}.'.format(i, n_epochs, current_loss))

t_extr = np.arange(100).reshape([100, 1])
y_pred = sess.run(y, feed_dict={t: t_extr})

t_extr, y_pred, t_dat, y0_dat = np.squeeze(t_extr), np.squeeze(y_pred), np.squeeze(t_dat), np.squeeze(y0_dat)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(t_extr, y_pred)
plt.scatter(t_dat, y0_dat)
# plt.yscale('log')
ax.set_xticks(t_extr[::10])
ax.set_xticklabels((t_extr - len(t_dat))[::10])
plt.xlabel('Days from now')
plt.ylabel('Confirmed cases')
plt.show()