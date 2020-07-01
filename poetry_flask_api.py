import pandas as pd
import numpy as np

from collections import namedtuple

from gensim.models import KeyedVectors
from gensim.scripts import glove2word2vec
import os

from keras.models import load_model

endline_token = '<nEXt>'
endpoem_token = '<eNd>'

def load_corpus(corpus_file):

    return pd.read_pickle(corpus_file)

def load_embeddings(embed, tmp):

    if not os.path.isfile(tmp_file):
        _ = glove2word2vec(glove_file, tmp_file)

    return KeyedVectors.load_word2vec_format(tmp_file)

def build_word_vector(text, size, embeddings):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += embeddings[word].reshape((1, size))
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
        loaded = np.load(embed_file)
        
        return {'X': loaded['X'], 
                'Y': loaded['Y'], 
                'seq_length': len(loaded['X'][0])}
    else:
        return None

def generate_poem(start_word='', temperature=0.5, max_words=200):

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
    for i in range(max_words):
        x = np.array([[build_word_vector([int_to_word[word]], 300, glove_model)[0] for word in pattern]])
        prediction = model.predict(x, verbose=0)[0]
        index = sample(prediction, temperature)
        
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

    return [int_to_word[value] for value in gen_poem]

def to_html(word_list):

    new_word_list = [convert_token(token, output='html') for token in word_list]

    return '<p>\n' + ' '.join(new_word_list)


## pre-processing

print('loading corpus...')
corpus = load_corpus('./data/haikus_train_df.pickle')
n_poems = len(corpus)

print('loading dictionaries...')
dictionaries = generate_dictionary(corpus)
word_to_int = dictionaries['word_to_int']
int_to_word = dictionaries['int_to_word']
n_vocab_words = len(word_to_int)

print('loading embeddings...')
glove_file = './data/image_to_text/glove.840B.300d.txt'
tmp_file = './data/image_to_text/glovetmp.txt'
glove_model = load_embeddings(glove_file, tmp_file)

print('loading sequences...')
poem_embeds = load_poem_embeddings('./data/haiku_train_wordembed.npz')
word_poemX = poem_embeds['X']
seq_len = poem_embeds['seq_len']

print('loading model...')
model = load_model('./weights/word_embedding/wordembed-weights-70-2.5563.hdf5')

poem_html = ''
print('done loading')

