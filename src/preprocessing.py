import spacy
import unidecode
from nltk.corpus import stopwords
from nltk.stem.rslp import RSLPStemmer  

nlp = spacy.load('pt_core_news_sm')
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