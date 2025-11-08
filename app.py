import os
import numpy as np
import pandas as pd
import pickle
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.models
from tensorflow.keras.models import model_from_json
import streamlit
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

nombre = 'TU NOMBRE AQUÍ'

with open('models/tokenizer.pickle', 'rb') as tk:
    tokenizer = pickle.load(tk)

json_file = open('models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
lstm_model = model_from_json(loaded_model_json)
lstm_model.load_weights('models/.model.weights.h5')


def nps_prediction(comentario):
    nps = []
    input_comentario = [comentario]
    input_comentario = [x.lower() for x in input_comentario]
    imput_comentario = [re.sub(r'[^a-zA-z0-9\s]', '', x) for x in input_comentario]

    input_feature = tokenizer.texts_to_sequences(imput_comentario)
    input_feature = pad_sequences(input_feature, maxlen=1000, padding='pre')
    nps = lstm_model.predict(input_feature)[0]
    if(np.argmax(nps) == 0):
        pred = 'Detractor'
    elif(np.argmax(nps) == 1):
        pred = 'Pasivo'
    else:
        pred = 'Promotor'

    return pred

def run():
    streamlit.title("NPS predicción")
    html_temp = '''
'''
    streamlit.markdown(html_temp, unsafe_allow_html=True)

    comentario = streamlit.text_area("Ingrese el comentario del cliente:")
    prediccion = ''
    if streamlit.button("Predecir NPS"):
        pred_nps = nps_prediction(comentario)
        streamlit.success(f"El NPS predicho es: {pred_nps}")
    streamlit.success("Desarrollado por : TU_NOMBRE_AQUI".format(prediccion))

if __name__ == '__main__':
    run()