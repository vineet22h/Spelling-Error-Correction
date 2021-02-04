from flask import Flask, render_template, flash, redirect, url_for, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import os
from nltk import ngrams

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from model.prediction import Encoder, Attention, OneStepDecoder, pred_Encoder_decoder, loss_function

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

app = Flask(__name__)
app.config['UPLOAD_FOLDE'] = 'static/uploads/'
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

maxlen = 32
NGRAM_VOCAB = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', \
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', \
               ' ', '<SOW>', '<EOW>']

def split_on_star(input_data):
    return tf.strings.split(input_data, sep = '*')

trigram_vec = TextVectorization(output_sequence_length= maxlen+2, standardize = None, split = split_on_star, max_tokens = len(NGRAM_VOCAB)+2, output_mode='int')
trigram_vec.adapt(NGRAM_VOCAB)

trigram_index_to_word = {idx: word for idx, word in enumerate(trigram_vec.get_vocabulary())}
trigram_word_to_index = {word: idx for idx, word in enumerate(trigram_vec.get_vocabulary())}

vocab_size = len(trigram_vec.get_vocabulary())
embedding_dim = 100
lstm_size = 256
att_units = 256
maxlen = 34

pred_model = pred_Encoder_decoder(vocab_size, vocab_size, embedding_dim, lstm_size, lstm_size, maxlen, maxlen, 'concat', att_units, trigram_word_to_index)
pred_model.compile(optimizer = 'Adam', loss = loss_function)
pred_model.build(input_shape= (None, 1, maxlen))
pred_model.load_weights('model/concat_best_trigram.h5')

def predict(seq, vectorizer, index_to_word, gram = 'uni'):
    if gram =='uni':
        seq = '<SOW> '+' '.join(list(seq))+' <EOW>'
    else:
        seq = '<SOW>*'+'*'.join(list(seq))+'*<EOW>'
    seq = vectorizer([seq])
    pred, _ = pred_model.predict(tf.expand_dims(seq, 0))
    output = []
    for i in pred:
        word = index_to_word[i[0]]
        if word == '<EOW>':
            break
        output.append(word)
    return ''.join(output)

def make_correction(sentence):
    results = []
    splits = sentence.split()
    if len(splits) <3:
        return sentence
    splits = list(ngrams(sentence.split(), 3))
    predictions = []
    for i in splits:
        predictions.append(predict(' '.join(i), trigram_vec, trigram_index_to_word, 'tri').split())
    predictions = np.array(predictions)
    shape= predictions.shape[0]
    # print(predictions)
    if shape == 1:
        return ' '.join(predictions[0])
    if shape == 2:
        return ' '.join(predictions[0,:2])+' '+' '.join(predictions[1,1])
    if shape >= 3:
        return ' '.join(predictions[0,:2])+' '+' '.join(predictions[1:-1,1])+' '+' '.join(predictions[-1,1:])

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

@app.route('/')
def upload_form():
    print('in upload_form')
    return render_template('main.html')

@app.route('/convert', methods=['POST'])
def convert():
    print('inside convert')
    inp = request.form['input']

    return jsonify({
        'output' : make_correction(inp)
    })

if __name__ == "__main__":
    app.run(port = 5500)