import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import*

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, lstm_size, input_length):
        super(Encoder, self).__init__()
        self.lstm_size = lstm_size
        self.enc_embed = Embedding(input_dim = vocab_size, output_dim = embedding_size)
        self.enc_lstm = Bidirectional(LSTM(lstm_size, return_sequences = True, return_state = True, dropout = 0.4))
    
    def call(self, input_sequence, states):
        embedding = self.enc_embed(input_sequence)
        output_state, enc_frwd_h, enc_frwd_c, enc_bkwd_h, enc_bkwd_c = self.enc_lstm(embedding, initial_state = states)
        return output_state, enc_frwd_h, enc_frwd_c, enc_bkwd_h, enc_bkwd_c
    
    def initialize_states(self, batch_size):
        return [tf.zeros((batch_size, self.lstm_size)), tf.zeros((batch_size, self.lstm_size)),
                tf.zeros((batch_size, self.lstm_size)), tf.zeros((batch_size, self.lstm_size))]

class Attention(tf.keras.layers.Layer):
    def __init__(self,scoring_function, att_units):
        super(Attention, self).__init__()
        self.scoring_function = scoring_function
        if scoring_function == 'dot':
            self.dot = Dot(axes = (1, 2))
        elif scoring_function == 'general':
            self.W = Dense(att_units)
            self.dot = Dot(axes = (1, 2))
        elif scoring_function == 'concat':
            self.W1 = Dense(att_units)
            self.W2 = Dense(att_units)
            self.W3 = Dense(att_units)
            self.V = Dense(1)
            
    def call(self, dec_frwd_state, dec_bkwd_state, encoder_output):
        dec_frwd_state = tf.expand_dims(dec_frwd_state, 1) 
        dec_bkwd_state = tf.expand_dims(dec_bkwd_state, 1)
#         
        if self.scoring_function == 'dot':
            score = tf.transpose(self.dot([tf.transpose(decoder_hidden_state, (0, 2, 1)), encoder_output]), (0, 2,1))           
        elif self.scoring_function == 'general':
            mul = self.W(encoder_output)
            score = tf.transpose(self.dot([tf.transpose(decoder_hidden_state, (0, 2, 1)), mul]), (0, 2,1))           
        elif self.scoring_function == 'concat':
            inter = self.W1(dec_frwd_state) + self.W2(dec_bkwd_state) + self.W3(encoder_output)
            tan = tf.nn.tanh(inter)
            score = self.V(tan)
        attention_weights = tf.nn.softmax(score, axis =1)
        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class OneStepDecoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):
        super(OneStepDecoder, self).__init__()
      # Initialize decoder embedding layer, LSTM and any other objects needed
        self.embed_dec = Embedding(input_dim = vocab_size, output_dim = embedding_dim)
        self.lstm = Bidirectional(LSTM(dec_units, return_sequences = True, return_state = True, dropout = 0.4))
        self.attention = Attention(scoring_function = score_fun, att_units = att_units)
        self.fc = Dense(vocab_size)
    
    def call(self,input_to_decoder, encoder_output, state_frwd_h, state_frwd_c, state_bkwd_h, state_bkwd_c):
        embed = self.embed_dec(input_to_decoder)
        context_vect, attention_weights = self.attention(state_frwd_h, state_bkwd_h, encoder_output)    
        final_inp = tf.concat([tf.expand_dims(context_vect, 1), embed], axis = -1)
        out, dec_frwd_h, dec_frwd_c, dec_bkwd_h, dec_bkwd_c = self.lstm(final_inp, [state_frwd_h, state_frwd_c, state_bkwd_h, state_bkwd_c])
        out = tf.reshape(out, (-1, out.shape[2]))
        out = Dropout(0.5)(out)
        output = self.fc(out)
        return output, dec_frwd_h, dec_frwd_c, dec_bkwd_h, dec_bkwd_c, attention_weights, context_vect

class pred_Encoder_decoder(tf.keras.Model): 
    def __init__(self, inp_vocab_size, out_vocab_size, embedding_dim, enc_units, dec_units, max_len_ita, max_len_eng, score_fun, att_units, word_to_index):
        #Intialize objects from encoder decoder
        super(pred_Encoder_decoder, self).__init__()
        self.encoder = Encoder(inp_vocab_size, embedding_dim, enc_units, max_len_ita)
        self.one_step_decoder = OneStepDecoder(out_vocab_size, embedding_dim, max_len_eng, dec_units, score_fun, att_units)
        self.word_to_index = word_to_index
        self.max_len = max_len_ita
        
    def call(self, params):
        enc_inp = params[0]
        initial_state = self.encoder.initialize_states(1)
        enc_output, enc_frwd_h, enc_frwd_c, enc_bkwd_h, enc_bkwd_c = self.encoder(enc_inp, initial_state)
        pred = tf.expand_dims([self.word_to_index['<SOW>']], 0)
        all_pred = []
        all_attention = []

        dec_frwd_h = enc_frwd_h
        dec_frwd_c = enc_frwd_c
        dec_bkwd_h = enc_bkwd_h
        dec_bkwd_c = enc_bkwd_c
        for timestep in range(self.max_len):
            # Call onestepdecoder for each token in decoder_input
            output, dec_frwd_h, dec_frwd_c, dec_bkwd_h, dec_bkwd_c, attention, _ = self.one_step_decoder(pred, enc_output, dec_frwd_h, dec_frwd_c, dec_bkwd_h, dec_bkwd_c)
            pred = tf.argmax(output, axis = -1)
            all_pred.append(pred)
            pred = tf.expand_dims(pred, 0)
            all_attention.append(attention)

        return all_pred, all_attention

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

# sentence = 'teh'
# result, _ = predict(sentence)

# print('input:', sentence)
# print('output:',result)


# sentence = 'The rabbit holV ewnt straight on liek a tnnel ofr some way any then dipped suddnely down so suddnenly tat Alice had nobt a moment to think aPout stopipng herself before she found hersefl falling dow a verZy deeup wLell'
# print('Input:', sentence)
# print('\nOutput:', make_correction(sentence))

