from flask import Flask, request, Response
import os
import json
from app.src.models.predict import predict_pipeline, predict_pipeline_from_cos
from app.src.models.train_model import training_pipeline, training_pipeline_from_cos
import warnings
import pandas as pd

# Quitar warnings innecesarios de la salida
warnings.filterwarnings('ignore')

# -*- coding: utf-8 -*-
app = Flask(__name__)

# On IBM Cloud Cloud Foundry, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8000
port = int(os.getenv('PORT', 8000))

@app.route('/predict', methods=['GET'])
def predict_route():
    """
        Función de lanzamiento del pipeline de inferencia de datos alojados en local.

        Returns:
           dict.  Mensaje de salida (predicción)
    """
    data = pd.read_csv('x_test.csv')
    y_pred = predict_pipeline(data)
    # Lanzar la ejecución del pipeline de inferencia
    print(y_pred)
    if y_pred == "":
        return ("No hay modelo entrenado. Entrena primero un modelo para poder hacer predicciones")
    else:
        return {'Predicted value': y_pred}

@app.route('/predict_data', methods=['POST'])
def predict_route2():
    """
        Función de lanzamiento del pipeline de inferencia de datos pasados como JSON con petición POST.

        Returns:
           dict.  Mensaje de salida (predicción)
    """
    # http://0.0.0.0:8000/predict_data
    # Obtener los datos pasados por el request
    data = request.get_json()
    #data = [20762,"city_11",0.55,"Female","Has relevent experience","no_enrollment","Graduate","STEM","4","10/49","Pvt Ltd","1",2]
    #data = [29725,"city_40",0.7759999999999999,"Male","No relevent experience","no_enrollment","Graduate","STEM","15","50-99","Pvt Ltd","">4","47",0.0]
    y_pred = predict_pipeline(data)
    # Lanzar la ejecución del pipeline de inferencia
    print(y_pred)
    if y_pred == "":
        return ("No hay modelo entrenado. Entrena primero un modelo para poder hacer predicciones")
    else:
        return {'Predicted value': y_pred}

@app.route('/predict_from_cos', methods=['POST'])
def predict_route3():
    """
        Función de lanzamiento del pipeline de inferencia con datos alojados en la nube.

        Returns:
           dict.  Mensaje de salida (predicción)
    """

    # Obtener los datos pasados por el request
    data = request.get_json()
    y_pred = predict_pipeline_from_cos(data)
    # Lanzar la ejecución del pipeline de inferencia
    print(y_pred)
    if y_pred == "":
        return ("No hay modelo entrenado. Entrena primero un modelo para poder hacer predicciones")
    else:
        return {'Predicted value': y_pred}

@app.route('/train', methods=['GET'])
def predict_route4():
    """
        Función de lanzamiento del pipeline de entrenamiento con datos alojados en local.

        Returns:
           dict.  Mensaje de salida (predicción)
    """

    # Entrenar datos a partir de csv guardado en local
    data = pd.read_csv('ds_job.csv')
    y_pred = training_pipeline(data)
    # Lanzar la ejecución del pipeline de entrenamiento
    print("Modelo entrenado")
    return 'Modelo entrenado'

@app.route('/train_model', methods=['POST'])
def predict_route5():
    """
        Función de lanzamiento del pipeline de entrenamiento a partir de datos subidos a la nube.

        Returns:
           dict.  Mensaje de salida (predicción)
    """

    # Obtener los datos pasados por el request
    print("Entrenando desde COS")
    data = request.get_json()

    # Lanzar la ejecución del pipeline de entrenamiento
    y_pred = training_pipeline_from_cos(data)
    print("Modelo entrenado")
    return 'Modelo entrenado'
    
@app.route('/post_desde_front', methods=['POST'])
def predict_route6():
    """
        Función de lanzamiento del pipeline de entrenamiento a partir de datos subidos a la nube.

        Returns:
           dict.  Mensaje de salida (predicción)
    """

    # Obtener los datos pasados por el request
    #data = request.get_json()
    data = request.get_data()
    texto = data.decode("utf-8")
    texto = texto.split(";")
    indice_ciudad = float(texto[2])
    horas_formacion = float(texto[12])
    dato_final = []
    for i in range(0,13):
        if i == 2:
            dato_final.append(indice_ciudad)
        elif i== 12:
            dato_final.append(horas_formacion)
        else:
            dato_final.append(texto[i])
        #print(dato_final[i])
        #print(type(dato_final[i]))
    print(dato_final)
    # Lanzar la ejecución del pipeline de entrenamiento
    #y_pred = training_pipeline_from_cos(data)
    #print("FINAL PROGRAMA")
    y_pred = predict_pipeline(dato_final)
    print(y_pred)
    return "OK"

# main
if __name__ == '__main__':
    # ejecución de la app
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
