import tensorflow as tf
import numpy as np 
import keras

# Load Model and Word:
def load_():
    words = np.load('files/words.npy', allow_pickle=True)
    model = keras.models.load_model('models/first-emb-2hiddenGRU(128)-changevocabsize')
    return model, words

# Pipeline funtion to make inference 
def pipeline_inference (review, words, model): 

  '''
  Función que prepara una review aislada y hace inferencia con el modelo. 

  Entradas: 
    review: String con la review a hacer inferencia
    words: Arreglo de numpy que contiene las palabras del vocabulario
    model: Modelo entrenado con el que se realiza inferencia

  Salida: 

    pred: Predicción del modelo

  '''
  
  # Parameters
  n_characters=1000
  vocab_size=60000
  num_oov_buckets=5000

  # Create table from words
  words = tf.constant(words)
  word_ids = tf.range(len(words), dtype=tf.int64)
  vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
  table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

  # Encode review
  review_e = review.encode()
  
  # Convert review to tensor
  X_rev = tf.convert_to_tensor(review_e)

  # Preprocess string
  X_rev = tf.strings.substr(X_rev, 0, n_characters) #primeros n caracteres
  X_rev = tf.strings.regex_replace(X_rev, rb"<br\s*/?>", b" ") #remplasar tags html por espacios
  X_rev = tf.strings.regex_replace(X_rev, b"[^a-zA-Z']", b" ")
  X_rev = tf.strings.lower(X_rev) #  Se pone todo en minuscula
  X_rev = tf.strings.split(X_rev) #Tokenization

  # Look on table
  X_rev = table.lookup(X_rev)

  # Add dimension that correspond to batch
  X_rev = tf.expand_dims(X_rev, axis=0)

  # Make inference
  pred = model.predict(X_rev)  

  return pred[0][0]