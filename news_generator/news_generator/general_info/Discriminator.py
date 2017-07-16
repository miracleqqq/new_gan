from process_data import input_data
import tensorflow as tf
import sys, os
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d

BATCH_SIZE = 64
SEQ_LEN = 2
SCORE_DIM = 20  #generator score dimention
DIM = 128
LAMBDA = 10 # Gradient penalty lambda hyperparameter.
ITERS = 2000
CRITIC_ITERS = 10
data = input_data()
charmap = data._charmap




def Discriminator(inputs, scores):
    output = tf.concat([inputs, scores], 1)
    output = tf.transpose(output, [0,2,1])
    output = lib.ops.conv1d.Conv1D('Discriminator.Input', len(charmap), DIM, 1, output)
    output = data.ResBlock('Discriminator.1', output)
    output = data.ResBlock('Discriminator.2', output)
    output = data.ResBlock('Discriminator.3', output)
    output = data.ResBlock('Discriminator.4', output)
    output = data.ResBlock('Discriminator.5', output)
    output = tf.reshape(output, [-1, (2+SEQ_LEN)*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', (2+SEQ_LEN)*DIM, 1, output)
    return output

#test
"""
real_inputs_discrete = tf.placeholder(tf.int32, shape=[BATCH_SIZE, SEQ_LEN])
real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
score = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 2, len(charmap)])
Discriminator(real_inputs, score) 
"""