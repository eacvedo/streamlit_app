import tensorflow as tf
import numpy as np 
import keras

# Load Model and Word:
def load_():
    words = np.load('files/words.npy', allow_pickle=True)
    model = keras.models.load_model('models/BEST-first-emb-1hiddenGRU(128)-w-masking-reg-4')
    return model, words

# Pipeline funtion to make inference 
def pipeline_inference (review, model): 

  '''
  Función que prepara una review aislada y hace inferencia con el modelo. 

  Entradas: 
    review: String con la review a hacer inferencia
    model: Modelo entrenado con el que se realiza inferencia

  Salida: 

    pred: Predicción del modelo

  '''
  
  # Make inference
  pred = model.predict([review])  

  return pred[0][0]