import tensorflow as tf
node1 = tf.constant(3.0, dtype = tf.float32)
#also tf.float32 implicitly
node2 = tf.constant(4.0)
# Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
# print(node1, node2)

# Session communicate with hardware
sess = tf.Session()
# [3.0, 4.0]
# print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
# node3:  Tensor("Add:0", shape=(), dtype=float32)
# print("node3:", node3)
# sess.run(node3):  7.0
# print("sess.run(node3)", sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
# + provides a shortcut for tf.add(a, b)
adder_node = a + b
# 7.5
print(sess.run(adder_node, {a: 3, b: 4.5}))
# [3. 7.]
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3
# 22.5
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

W = tf.Variable([0.3], dtype = tf.float32)
b = tf.Variable([-0.3], dtype = tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
init = tf.global_variables_initializer()
# until sess.run, the variables are uninitialized
sess.run(init)
# [ 0.          0.30000001  0.60000002  0.90000004]
print(sess.run(linear_model, {x:[1, 2, 3, 4]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
# every parameter's delta get square, and get sum
loss = tf.reduce_sum(squared_deltas)
# 23.66
print(sess.run(loss, {x: [1, 2, 3, 4], y:[0, -1, -2, -3]}))


fixW = tf.assign(W, [-1.0])
fixb = tf.assign(b, [1.0])
sess.run([fixW, fixb])
# 0.0
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))













