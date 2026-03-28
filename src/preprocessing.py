import spacy
import unidecode
from nltk.corpus import stopwords
from nltk.stem.rslp import RSLPStemmer  
import numpy as np


nlp = spacy.load('pt_core_news_md')
stopwords = list(stopwords.words('portuguese'))
PT_STOPWORDS = [unidecode.unidecode(word) for word in stopwords]
STEMMER = RSLPStemmer()

def tokenize(text):
    return text.split()


def remove_stopwords(tokens):

    without_stopwords = []

    for tok in tokens:
        if tok not in PT_STOPWORDS:
            without_stopwords.append(tok)
    
    return without_stopwords


def stemming(tokens):

    stems = []

    for tok in tokens:
        stems.append(STEMMER.stem(tok))

    return stems


def lemmatization(text):

    lemmas = []
    doc = nlp(text)

    for txt in doc:
        lemmas.append(txt.lemma_)

    return lemmas


def vectorization_docs(series):

    vectors_document = [] 

    for lemmas_list in series:
        vectors_words = []
        for lemma in lemmas_list:
            if nlp.vocab.has_vector(lemma):
                vectors_words.append(nlp.vocab[lemma].vector)
        
        if vectors_words:

            mean_vector = np.mean(vectors_words, axis=0)
            vectors_document.append(mean_vector)
        
        else:
            vectors_document.append(np.zeros(300)) #para caso não haja um vetor útil

    
    return np.array(vectors_document)