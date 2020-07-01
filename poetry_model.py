from keras.models import Sequential, load_model
from keras.layers import Dense, Input, LSTM, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

class PoetryGenerator():
    """
    Base class for poetry generation
    """
    
    def __init__(self, engine, tokenize='wordembed'):
        
        super().__init__()
        
        self.engine = engine
        self.tokenize = tokenize
    
    def load_corpus(self, train, valid, special_tokens):
        """
        Assumes that special tokens have already been put into the dataset
        
        Parameters
        ---
        train:          training corpus
        valid:          validation corpus
        special_tokens: dictionary of the form token_type: token
                        must include 'newline' and 'endpoem' tokens
                        other possible tokens include 'newstanza'
        """
        

    def transform(self, corpus, seq_len):
    
        poem_count = len(corpus)
        self.pattern_count = 0
        
        # prepare the dataset of input to output pairs encoded as integers
        self.seq_len = seq_len

        self.poemX = []
        self.poemY = []
        self.pattern_count = 0

        self.corpusX = []
        self.corpusY = []
        for poem_index in range(0, poem_count):

            textX = []
            textY = []
            
            poem = corpus[poem_index]
            # add padding to poem
            poem = list(np.full(seq_length - 1, '')) + list(poem)
            
            for i in range(0,  len(poem) - seq_len, 1):
                seq_in = poem[i:i + seq_len]
                seq_out = poem[i + seq_len]
                textX.append([self.token_to_int[token] for token in seq_in])
                textY.append(self.token_to_int[seq_out])

            self.pattern_count = max(self.pattern_count, len(textX))

            self.poemX.append(textX)
            self.poemY.append(textY)

            self.corpusX += textX
            self.corpusY += textY
    
    def create_dict(self):
        
        # create corpus_raw
        corpus_raw = flatten(list(corpus))
        self.tokens = sorted(set(list(corpus_raw)))
        self.token_to_int = dict((t, i) for i, t in enumerate(self.tokens))
        self.int_to_token = dict((i, t) for i, t in enumerate(self.tokens))
        
        self.token_count = len(corpus_raw)
        self.vocab_count = len(tokens)

    
    def fit(self):
                
        self.fitted = True
    
    def generate(self, temperature=1.0):
        
        if not self.fitted:
            raise ValueError('Model not fitted')