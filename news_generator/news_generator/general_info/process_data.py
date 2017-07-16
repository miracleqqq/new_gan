import sys, os
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
import sys, os
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
import xlrd
import tensorflow as tf
import tflib as lib

BATCH_SIZE = 64
SEQ_LEN = 2
SCORE_DIM = 20  #generator score dimention
DIM = 128
LAMBDA = 10 # Gradient penalty lambda hyperparameter.
ITERS = 2000
CRITIC_ITERS = 10

class input_data(object):
    
    def __init__(self):
        team = []
        score_t = []
        how_win_t = []
        sentence = []
        #with open('data_set.csv') as f:
        with xlrd.open_workbook('data_set.xls') as w:
            sheet = w.sheet_by_index(0)
            #f_csv = csv.reader(f)
            #headers = next(f_csv)
            for row in range(sheet.nrows):
                teams = sheet.row_values(row)[0].split('_')
                team.append((teams[0], teams[1]))
                scores = sheet.row_values(row)[1].split('-')
                score_t.append((int(scores[0]), int(scores[1])))
                how_win_t.append(sheet.row_values(row)[2])
                sentence.append(sheet.row_values(row)[3])
        
        t_len = len(score_t)
        score = score_t[0:int(t_len*0.8)]
        score_test = score_t[int(t_len*0.8):]
        how_win = how_win_t[0:int(t_len*0.8)]
        how_win_test = how_win_t[int(t_len*0.8):]

        inv_charmap = []
        inv_charmap.append('unk')
        charmap = {'unk':0}
        for word in how_win:
            if word not in inv_charmap:
                charmap[word] = len(inv_charmap)
                inv_charmap.append(word)
        
        self._score = score
        self._score_test = score_test
        self._how_win = how_win
        self._how_win_test = how_win_test
        self._charmap = charmap
        self._inv_charmap = inv_charmap
        for i in range(len(score)):
            print(score[i], how_win[i])
        
        

    def get_next_batch(self,batch_size, index):
        length = len(self._score)
        start_point = index * batch_size
        batch_gen_score = []
        batch_disc_score = []
        batch_how_win = []
        for i in range(batch_size):
            current_point = start_point + i
            if current_point >= length:
                current_point -= length
        
            one_score = self._score[current_point]
            gen_score = [one_score[0]/200.0 for i in range(int(SCORE_DIM/2))] + [one_score[1]/200.0 for i in range(int(SCORE_DIM/2))]
            batch_gen_score.append(gen_score)
        
            disc_score = [[one_score[0]/200.0 for i in range(len(self._charmap))]] + [[one_score[1]/200.0 for i in range(len(self._charmap))]]
            batch_disc_score.append(disc_score)
        
            batch_how_win.append([self._charmap[self._how_win[current_point]],0])

    
        return batch_gen_score, batch_disc_score, batch_how_win
    
    def make_noise(self,shape):
        return tf.random_normal(shape)
    
    def ResBlock(self,name, inputs): 
        output = inputs
        output = tf.nn.relu(output)
        output = lib.ops.conv1d.Conv1D(name+'.1', DIM, DIM, 5, output)
        output = tf.nn.relu(output)
        #output = tf.expand_dims(output, 2)
        #print(output)
        output = lib.ops.conv1d.Conv1D(name+'.2', DIM, DIM, 5, output)
        #output = tf.expand_dims(output, 2)
        return inputs + (0.3*output)
        
    def softmax(self,logits):
        return tf.reshape(
            tf.nn.softmax(
                tf.reshape(logits, [-1, len(self._charmap)])),tf.shape(logits)
            )