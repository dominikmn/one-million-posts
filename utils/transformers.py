import string
import numpy as np

import ast
import tqdm
import pickle

from utils import cleaning

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s: %(message)s")
logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
logger.setLevel(logging.INFO)

def load_embedding_vectors(embedding_style, file=None):
    """
    Helper function in order to load word2vec or glove vector embedding files.
    The expected file format is the one found on https://deepset.ai/german-word-embeddings.

    Args: 
      embedding_style: {'word2vec', 'glove'}, default=None
      file: An embedding vector file containing words and their corresponding embedding vectors.
    Returns: 
      embedding_dict: dict whose keys are words and the values are the related vectors.
    """
    logger.info(f"Loading embedding dictionary for {embedding_style}.")
    if embedding_style not in ('word2vec', 'glove'):
       raise ValueError("embedding_style must be any of {'word2vec', 'glove'}") 
    file_cached = f'./cache/embedding_{embedding_style}.pickle'
    try:
        with open(file_cached, 'rb') as f_pickle:
            logger.info(f"Taking cached embedding dictionary from {file_cached} ...")
            embedding_dict = pickle.load(f_pickle) 
            logger.info(f"Embedding dictionary loaded.")
    except FileNotFoundError:
        logger.info(f"No existing embedding dictionary found in the cache: ./cache/embedding_{embedding_style}.pickle")
        if file is None:
            file = f'./embeddings/{embedding_style}_vectors.txt'
        embedding_dict = dict()
        with open(file, 'r') as f:
            logger.info(f"Computing new embedding dictionary from source file {file} ...")
            for line in tqdm.tqdm(f.readlines()):
                split_line = line.split()
                if embedding_style == 'word2vec':
                    word = ast.literal_eval(split_line[0]).decode('utf-8')
                elif embedding_style == 'glove':
                    word = split_line[0]
                embedding = np.array([float(val) for val in split_line[1:]], dtype=np.float32)
                embedding_dict[word] = embedding
            if embedding_style == 'glove':
                embedding_dict['UNK'] = embedding_dict['<unk>']
                del embedding_dict['<unk>']
            logger.info(f"Embedding dictionary computed.")
        with open(file_cached, 'wb') as f_pickle:
            logger.info(f"Dumping embedding dictionary as pickle {file_cached} ...")
            pickle.dump(embedding_dict, file=f_pickle)
            logger.info(f"Embedding dictionary dumped.")
    return embedding_dict

class MeanEmbeddingVectorizer(object):
    r"""
    Convert a collection of documents to a matrix of mean word vector values.

    Args:
      embedding_dict : dict
        A dictionary that maps words to their corresponding vector in the embedding space.
      preprocessor : callable, default=None
        Add an additional preprocessor before str.lower() and str.translate() are applied.
    """
    def __init__(self, embedding_dict, preprocessor=None):
        self.embedding_dict = embedding_dict
        self.preprocessor = preprocessor

    def fit(self, X, y):
        return self
    
    def _preprocess(self, sentence):
        if self.preprocessor is not None:
            sentence = self.preprocessor(sentence)
        else:
            s = cleaning.normalize(sentence)
        if not s: 
            s='UNK'
        return s
        
    def transform(self, X):
        """
        Args: 
          X: iterable 
            1-dim object containing sentences where the sentence is represented as one string.
        Returns: 
          transformed: np.matrix
            A numpy matrix of the dimension (number of samples, common size of the word vectors)
        """
        transformed = np.matrix([
            np.mean([self.embedding_dict[w] if w in self.embedding_dict else self.embedding_dict['UNK'] for w in self._preprocess(sentence).split()], axis=0)
            for sentence in X
        ])
        return transformed