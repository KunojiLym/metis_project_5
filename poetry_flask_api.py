import pandas as pd
import numpy as np

from collections import namedtuple

from gensim.models import KeyedVectors
from gensim.scripts import glove2word2vec
import os

from sql_embeddings import call_embedding

from keras.models import load_model
from keras.backend.tensorflow_backend import set_session, clear_session

import tensorflow as tf
from multiprocessing import Pool
from numba import cuda

endline_token = '<nEXt>'
endpoem_token = '<eNd>'

def load_corpus(corpus_file):

    return pd.read_pickle(corpus_file)

def load_embeddings(embed, tmp):

    if not os.path.isfile(tmp):
        _ = glove2word2vec(embed, tmp)

    return KeyedVectors.load_word2vec_format(tmp)

def build_word_vector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += call_embedding(word, 'StanfordGlove').reshape((1, size))
            count += 1
        except:
            continue
    if count != 0:
        vec /= count
    return vec

def sample(preds, temperature=0.5):
    
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    
    return np.argmax(probs)

def convert_token(token, output='plain'):
    
    newline = {'html': '<br />\n', 'plain': '\n'}
    endpoem = {'html': '</p>\n', 'plain': ''}

    if token == endline_token:
        return newline[output]
    elif token == endpoem_token:
        return endpoem[output]
    elif token != '':
        return token

flatten = lambda l: [item for sublist in l for item in sublist]

def generate_dictionary(corpus):
    
    corpuswords_raw = flatten(list(corpus))

    words = sorted(set(corpuswords_raw))
    word_to_int = dict((w, i) for i, w in enumerate(words))
    int_to_word = dict((i, w) for i, w in enumerate(words))

    return {'word_to_int': word_to_int,
            'int_to_word': int_to_word}

def load_poem_embeddings(embed_file):

    if os.path.isfile(embed_file):
        loaded = np.load(embed_file, allow_pickle=True)
        
        return {'X': loaded['X'], 
                'Y': loaded['Y'], 
                'seq_length': len(loaded['X'][0][0])}
    else:
        return None

def generate_poem_api(args_dictionary):
    
    
    start_word = args_dictionary.get('start_word', default='')
    temperature = float(args_dictionary.get('temperature', default='0.5'))
    max_words = int(args_dictionary.get('max_words', default='200'))
    
    poem = generate_poem(start_word = start_word,
                         temperature = temperature,
                         max_words = max_words)
    
    print(args_dictionary)
    return poem

def generate_poem(start_word='', temperature=0.5, max_words=200):

    print('configure gpu session...')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                            # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    print('loading model...')
    model = load_model('./weights/word_embedding/wordembed-weights-109-2.4030.hdf5')
    print('done loading')

    if start_word == '':
        start = np.random.randint(0, n_vocab_words)
    else:
        try:
            start = word_to_int[start_word]
        except:
            start = np.random.randint(0, n_vocab_words)

    pattern = list(np.zeros(seq_len - 1)) + [start]
    gen_poem = [pattern[-1]]

    # generate words
    print('generating words...  ', end='')
    for i in range(max_words):
        x = np.array([[build_word_vector([int_to_word[word]], 300)[0] for word in pattern]])
        prediction = model.predict(x, verbose=0)[0]
        index = sample(prediction, temperature)
        
        print(index, len(int_to_word))
        while index >= len(int_to_word):
            index = sample(prediction, temperature)
            
        result = int_to_word[index]
        
        if result == '':
            prediction[index] = 0
            index = np.argmax(prediction)
            result = int_to_word[index]
        
        if result == endpoem_token:
            break;
        
        pattern.append(index)
        if result != '':
            gen_poem.append(index)
        pattern = pattern[1:len(pattern)]

        print('next word... ', end='')

    print('done predicting')
    print('cleaning up...')
    sess.close()
    cuda.select_device(0)
    cuda.close()
    del model

    print('done')
    print(f"temperature: {temperature}, start_word: {start_word}")
    return [int_to_word[value] for value in gen_poem]

def to_html(word_list):

    new_word_list = [convert_token(token, output='html') for token in word_list]

    return '<p>\n' + ' '.join(new_word_list)


## pre-processing

print('loading corpus...')
corpus = load_corpus('./data/haikus_train_df.pickle')
n_poems = len(corpus['text_withtokens'])
print('n_poems = ', n_poems)

print('loading dictionaries...')
dictionaries = generate_dictionary(corpus['text_withtokens'])
word_to_int = dictionaries['word_to_int']
int_to_word = dictionaries['int_to_word']
n_vocab_words = len(word_to_int)
print('n_vocab_words = ', n_vocab_words)

#print('loading embeddings...')
#glove_file = './data/image_to_text/glove.840B.300d.txt'
#tmp_file = './data/image_to_text/glovetmp.txt'
#glove_model = load_embeddings(glove_file, tmp_file)

print('loading sequences...')
poem_embeds = load_poem_embeddings('./data/haiku_train_wordembed.npz')
word_poemX = poem_embeds['X']
seq_len = poem_embeds['seq_length']

poem_html = ''
print('done loading')

