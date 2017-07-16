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

def Gennerator(n_samples, scores, prev_outputs=None):
    output = data.make_noise(shape=[n_samples, 32])
    output = tf.concat([scores, output], 1)
    output = lib.ops.linear.Linear('Generator.Input', SCORE_DIM+32, SEQ_LEN*DIM, output)
    output = tf.reshape(output, [-1, DIM, SEQ_LEN])
    output = data.ResBlock('Generator.1', output)
    output = data.ResBlock('Generator.2', output)
    output = data.ResBlock('Generator.3', output)
    output = data.ResBlock('Generator.4', output)
    output = data.ResBlock('Generator.5', output)
    output = lib.ops.conv1d.Conv1D('Generator.Output', DIM, len(charmap), 1, output)
    
    #output = tf.expand_dims(output, 2)
    output = tf.transpose(output, [0, 2, 1])
    output = data.softmax(output)
    return output

#test
"""
score = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 20])
Generator(BATCH_SIZE, score)
"""