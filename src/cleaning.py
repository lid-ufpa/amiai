from unidecode import unidecode
import string
import pandas

def clean_text(serie):

    clean_serie = []

    for text in serie:
        txt_cln = unidecode(text) 
        txt_cln = txt_cln.translate(str.maketrans('', '', string.punctuation)).strip()
        clean_serie.append(txt_cln)
    
    return pandas.Series(clean_serie, index=serie.index)
