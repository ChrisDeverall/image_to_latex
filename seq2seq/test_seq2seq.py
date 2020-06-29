import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time
import json
import csv
import sys


input_filename = sys.argv[1]
target_filename = sys.argv[2]
checkpoint_filename = sys.argv[3]
input_test_filename = sys.argv[4]
# target_test_filename = sys.argv[5]
predictions_filename = sys.argv[5]

def read_and_format_target(filename):
    latex_list = []
    with open(filename) as my_file:
        csv_reader = csv.reader(my_file, delimiter=',')
        for row in csv_reader:
            my_latex = row[1]
            my_latex = my_latex[2:-2]
            
            # my_latex = my_latex.replace("\\ t i m e s", "\times")
            my_latex =  "<start> " + my_latex + " <end>"
            latex_list.append(my_latex)
    return latex_list


def read_and_format_input(filename):
    with open(filename) as my_file:
        data = json.load(my_file) 
    return data

def make_label_list(data_in):
    symbol_label_list = []
    for sequence in data_in:
        sequence_string = "<start>"

        for symbol_dict in sequence:
            sequence_string=sequence_string +" "+ symbol_dict["label"]

        sequence_string = sequence_string + " <end>"
        symbol_label_list.append(sequence_string)
    return symbol_label_list

def read_and_format_input(filename):
    with open(filename) as my_file:
        data = json.load(my_file) 
    return data

def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)

  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

  return tensor, lang_tokenizer

def make_pos_list(data_in, tensor_shape):
    formula_number, max_length = tensor_shape
    
    symbol_pos_list = np.zeros((formula_number, max_length, 6),dtype=np.int32)
    
    for formula_index in range(formula_number):
        
        for symbol_index in range(max_length): 

            if symbol_index >=len(data_in[formula_index])+1:
                break # this means that you are at end of dict
            
            if symbol_index==0:
                mid_x = 0
                mid_y = 0 
                width = 0
                height = 0
                offset_x = 0
                offset_y = 0
            else:
                offset_index = symbol_index - 1 #because dict doesnt contain <start>

                mid_x = data_in[formula_index][offset_index]["x"] + int(data_in[formula_index][offset_index]["w"]/2)
                mid_y = data_in[formula_index][offset_index]["y"] + int(data_in[formula_index][offset_index]["h"]/2)
                offset_x = mid_x - prev_x
                offset_y = mid_y - prev_y            
                width = data_in[formula_index][offset_index]["w"]
                height = data_in[formula_index][offset_index]["h"]
                
            symbol_pos_list[formula_index][symbol_index] = np.array([int(mid_x), int(mid_y), int(offset_x), int(offset_y), int(width), int(height)])
            prev_x = mid_x
            prev_y = mid_y
            
    return symbol_pos_list

target_list = read_and_format_target(target_filename)
data = read_and_format_input(input_filename)
label_list = make_label_list(data)

input_tensor, inp_lang  = tokenize(label_list)
target_tensor, targ_lang  = tokenize(target_list)
input_tensor = tf.reshape(input_tensor,(input_tensor.shape[0], input_tensor.shape[1],1))

position_tensor = make_pos_list(data, (input_tensor.shape[0], input_tensor.shape[1]))
input_tensor = np.concatenate([input_tensor, position_tensor],axis=-1)

max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]



BUFFER_SIZE = len(input_tensor)   # number of train sequence pairs
BATCH_SIZE = 128   # hyper-param, decrease as smaller dataset
steps_per_epoch = len(input_tensor)//BATCH_SIZE # number of steps to go through dataset (rounded down)
embedding_dim = 256 # output dimensions of dense vector
units = 512    # output dimension of encoder
vocab_inp_size = len(inp_lang.word_index)+1 # +1 because no 0 in word_index
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch,example_target_batch = next(iter(dataset))

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim-6)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    labels = x[:,:,0]
    position_info = tf.dtypes.cast(x[:,:,1:], tf.float32)
    x = self.embedding(labels)
    x = tf.concat([x,position_info],-1)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))


class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights


decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))#
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

def evaluate(inputs):
  attention_plot = np.zeros((max_length_targ, max_length_inp))

  result = ''

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()

    result += targ_lang.index_word[predicted_id] + ' '

    if targ_lang.index_word[predicted_id] == '<end>':
      return result, attention_plot

    dec_input = tf.expand_dims([predicted_id], 0)

  return result, attention_plot


#make this param!
checkpoint_dir = checkpoint_filename 
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


test_data = read_and_format_input(input_test_filename)

test_label_list = make_label_list(test_data)
# test_target_list = read_and_format_target(target_test_filename)


inputs = []
for sentence in test_label_list:
    inputs.append([inp_lang.word_index[i] for i in sentence.split(' ')])
inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,maxlen=max_length_inp,padding='post')
inputs = tf.reshape(inputs,(inputs.shape[0], inputs.shape[1],1))

pos_inputs = make_pos_list(test_data, (inputs.shape[0],inputs.shape[1]))
inputs = tf.concat([inputs, pos_inputs],axis=-1)


with open(predictions_filename, 'a') as file:
    writer = csv.writer(file)
    for i in range(len(inputs)):
        if i+1 == len(inputs):
            output, _ = evaluate(inputs[-1:]) #this is to maintain dimension
        else:
            output, _ = evaluate(inputs[i:i+1])
        output = " ".join(output.split()[:-1])
        output = "$$" + output + "$$"
        my_row = [str(i),output]
        writer.writerow(my_row)
        if i % 1000==0:
          print(i)