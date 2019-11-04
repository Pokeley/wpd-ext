
from gensim.models import KeyedVectors
#from gensim.models.wrappers.fasttext import FastText

model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')

print(type(model))