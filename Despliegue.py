# -*- coding: utf-8 -*-
"""
# **Análisis de Sentimientos - Despliegue**

*   Se carga el modelo
*   Se cargan los datos futuros
*   Se preparan los datos futuros
*   Se aplica el modelo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Instalación de paquetes para tratamiento de texto
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from wordcloud import WordCloud

nltk.download('popular')
stopwords.words('spanish')
stemmer = SnowballStemmer('spanish')

#Cargamos el modelo
def dummy_fun(doc):
    return doc

import pickle
filename = 'modelo_text.pkl'
KNNmodel, tfidf,variables, labelencoder= pickle.load(open(filename, 'rb'))

#Cargamos los datos futuros
data = pd.read_excel("sentimientos_test.xlsx", sheet_name=0)
data.head()

#Limpieza
def tokenizar(texto):
  tokens = word_tokenize(texto)
  words = [w.lower() for w in tokens if w.isalnum()]
  return words
data['tokens'] = data['comentario'].apply(lambda x: tokenizar(x))
data.head()

#Eliminamos stopwords
from nltk.corpus import stopwords

sw= stopwords.words('spanish')
sw.append("profesor")
sw.remove('nada')

def limpiar_stopwords(lista):
  clean_tokens = lista[:]
  for token in lista:
    if token in sw:
      clean_tokens.remove(token)
  return clean_tokens

# Limpiamos los tokens
data['sin_stopwords'] = data['tokens'].apply(lambda x: limpiar_stopwords(x))
data.head()

#Reducción a la raíz

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')

def stem_tokens(lista):
  lista_stem = []
  for token in lista:
    lista_stem.append(stemmer.stem(token))
  return lista_stem

data['stemming'] = data['sin_stopwords'].apply(lambda x: stem_tokens(x))
data.head()

#Reducción a la raíz

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')

def stem_tokens(lista):
  lista_stem = []
  for token in lista:
    lista_stem.append(stemmer.stem(token))
  return lista_stem

data['stemming'] = data['sin_stopwords'].apply(lambda x: stem_tokens(x))
data.head()

#Nube de palabras: datos de despliegue

from wordcloud import WordCloud

lista_palabras = data["stemming"].tolist()
tokens = [keyword.strip() for sublista in lista_palabras for keyword in sublista]
texto= ' '.join(tokens)
wc = WordCloud(background_color="white", max_words=1000, margin=0)
wc.generate(texto)
wc.to_file("nube1.png")
plt.figure(figsize=(15,15))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

X_tfidf = tfidf.transform(data['stemming'] )

data_tfidf=pd.DataFrame(X_tfidf.todense(),columns=tfidf.get_feature_names())

data_tfidf.head()

#Hacemos la predicción
Y_pred = model.predict(data_tfidf)
print(Y_pred)

data['prediccion']=labelencoder.inverse_transform(Y_pred)
data
