# -*- coding: utf-8 -*-
"""
# **Análisis de sentimientos - Creación del Modelo**
1. Preparación del texto
2. División de datos
3. Aprendizaje
4. Evaluación
5. Guardamos el modelo
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

#Cargamos los datos
data = pd.read_excel("sentimientos.xlsx", sheet_name=0)
data.head()

#Limpieza del texto
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

#Nube de palabras

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

#Representación en vector de características tfidf

from sklearn.feature_extraction.text import TfidfVectorizer

def dummy_fun(doc):
    return doc

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None)

X = tfidf.fit_transform(data['stemming'])
data_tfidf=pd.DataFrame(X.todense(),columns=tfidf.get_feature_names())
data_tfidf

#Se adiciona el sentimiento al dataframe de palabras data_tfidf
data_tfidf["sentimiento"]=data["sentimiento"]
data_tfidf.head()

#Se codifican las categorias de la VARIABLE OBJETIVO

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data_tfidf["sentimiento"]=labelencoder.fit_transform(data_tfidf["sentimiento"])

data_tfidf.head()

#División 70-30
from sklearn.model_selection import train_test_split
X = data_tfidf.drop("sentimiento", axis = 1)
Y = data_tfidf['sentimiento']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
Y_train.value_counts().plot(kind='bar')

#Método árboles de clasificación
from sklearn import tree
modelTree = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=2, max_depth=10)
modelTree = modelTree.fit(X_train, Y_train) #70%

#Graficar el árbol
from sklearn.tree import plot_tree
nombres_variables=X_train.columns.values
nom_clases= labelencoder.classes_
plt.figure(figsize=(5,5))
plot_tree(modelTree, feature_names=nombres_variables, class_names= nom_clases, filled=True,fontsize=8)
plt.show()

#Evaluación sobre el conjunto de prueba
from sklearn.metrics import accuracy_score
prediccion = modelTree.predict(X_test) #30%
exactitud= accuracy_score(y_true=Y_test, y_pred=prediccion)
print(exactitud)

#KNN
from sklearn import neighbors
from sklearn import metrics

KNNmodel = neighbors.KNeighborsClassifier(n_neighbors=17,metric = 'euclidean')
KNNmodel.fit(X_train,Y_train)
predictKNN = KNNmodel.predict(X_test)
mae = metrics.mean_squared_error(Y_test,predictKNN)
print(f'para el modelo KNN, se puede esperar un error medio absoluto de {mae}')

Y_predict = KNNmodel.predict(X_test)
exactitud = accuracy_score(Y_test, Y_predict)
print(exactitud)

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

#Asignar semillas para generar numeros aleatorios
import tensorflow as tf
tf.random.set_seed(3)

#Arquitectura de la red neuronal
model_deep = Sequential()
model_deep.add(Dense(13,input_dim = 24, activation = "relu"))# 13 entradas y capa oculta de 5 neuronas
model_deep.add(Dense(8, activation = "relu")) #capa oculta de 3 neuronas
model_deep.add(Dense(4, activation = "softmax")) #capa de salida (valores de la variable objetivo) de 1 neuronas

#Aprendizaje
optimizer = keras.optimizers.SGD(learning_rate=0.03, momentum = 0.02)
model_deep.compile(loss="sparse_categorical_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
history = model_deep.fit(X_train,Y_train, epochs=500)

import pickle

filename = "modelo_text.pkl"
variables = X.columns.values
pickle.dump([KNNmodel, tfidf,labelencoder, variables],open(filename,'wb'))
