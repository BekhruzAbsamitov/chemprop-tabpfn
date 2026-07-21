#!/usr/bin/env python3
import tensorflow as tf
import time

print("GPU Available: ", tf.config.list_physical_devices())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


with tf.compat.v1.Session() as sess:
    matrix1 = tf.constant([1.0,2.0,3.0,4.0], shape=[2, 2])
    matrix2 = tf.linalg.inv(matrix1)
    product = tf.linalg.matmul(matrix1, matrix2)

    t0 = time.time()
    result = sess.run(product)
    t1 = time.time()
    print("result of matrix multiplication")
    print("===============================")
    print(result)
    print("Time: ", t1-t0)
print("===============================")
