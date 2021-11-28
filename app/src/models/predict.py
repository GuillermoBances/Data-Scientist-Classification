from ..data.make_dataset_predict import make_dataset_predict
from app import cos, client
from cloudant.query import Query
import pickle, os


def predict_pipeline(data, model_info_db_name='proyecto-final'):

    """
        Función para gestionar el pipeline completo de inferencia
        del modelo.

        Args:
            path (str):  Ruta hacia los datos.

        Kwargs:
            model_info_db_name (str):  base de datos a usar para almacenar
            la info del modelo.

        Returns:
            list. Lista con las predicciones hechas.
    """

    # obteniendo la información del modelo en producción
    model_info = get_best_model_info(model_info_db_name)
    if model_info == "":
        prediccion = ""
    else:
        # columnas a retirar
        cols_to_remove = model_info['objects']['cols_to_remove']
        # cargando y transformando los datos de entrada
        data_df = make_dataset_predict(data, model_info, cols_to_remove)
        # Descargando el objeto del modelo
        model_name = model_info['name']+'.pkl'
        print('------> Checking if model {} object is already downloaded'.format(model_name))
        modelo_descargado = 0
        arr = os.listdir()
        for i in range(len(arr)):
            if arr[i] == model_name:
                modelo_descargado = 1 
                print('---------> Model already downloaded. Using file from local')
                model = load_model_from_local(arr[i])
        if modelo_descargado == 0:
            print('------> Model not in local, downloading the model {} object from the cloud'.format(model_name))
            model = load_data_from_cos(model_name)
            pickle.dump(model, open(model_name, 'wb'))

        # realizando la inferencia con los datos de entrada
        print("Realizando prediccion")
        prediccion = model.predict(data_df).tolist()

    return prediccion

def predict_pipeline_from_cos(data, model_info_db_name='proyecto-final'):

    """
        Función para gestionar el pipeline completo de inferencia
        del modelo.

        Args:
            path (str):  Ruta hacia los datos.

        Kwargs:
            model_info_db_name (str):  base de datos a usar para almacenar
            la info del modelo.

        Returns:
            list. Lista con las predicciones hechas.
    """

    load_data = load_data_from_cos(data)
    # Carga de la configuración de entrenamiento
    model_info = get_best_model_info(model_info_db_name)
    if model_info == "":
        prediccion = ""
    else:
        # columnas a retirar
        cols_to_remove = model_info['objects']['cols_to_remove']
        # cargando y transformando los datos de entrada
        data_df = make_dataset_predict(load_data, model_info, cols_to_remove)
        # Descargando el objeto del modelo
        model_name = model_info['name']+'.pkl'
        print('------> Checking if model {} object is already downloaded'.format(model_name))
        modelo_descargado = 0
        arr = os.listdir()
        for i in range(len(arr)):
            if arr[i] == model_name:
                modelo_descargado = 1 
                print('---------> Model already downloaded. Using file from local')
                model = load_model_from_local(arr[i])
        if modelo_descargado == 0:
            print('------> Model not in local, downloading the model {} object from the cloud'.format(model_name))
            model = load_data_from_cos(model_name)
            pickle.dump(model, open(model_name, 'wb'))

        # realizando la inferencia con los datos de entrada
        print("Realizando prediccion")
        prediccion = model.predict(data_df).tolist()
    return prediccion


def load_model_from_local(name):
    """
         Función para cargar el modelo en IBM COS

         Args:
             name (str):  Nombre de objeto en COS a cargar.

         Kwargs:
             bucket_name (str):  depósito de IBM COS a usar.

        Returns:
            obj. Objeto descargado.
     """
    return pickle.load(open(name,'rb'))


def get_best_model_info(db_name):
    """
         Función para cargar la info del modelo de IBM Cloudant

         Args:
             db_name (str):  base de datos a usar.

         Kwargs:
             bucket_name (str):  depósito de IBM COS a usar.

        Returns:
            dict. Info del modelo.
     """
    db = client.get_database(db_name)
    query = Query(db, selector={'status': {'$eq': 'en_produccion'}})
    if len(query()['docs']) == 0:
        info = ""
    else:
        info = query()['docs'][0]
    
    return info 

def load_data_from_cos(name, bucket_name='models-uem-pf'):
    """
         Función para cargar el modelo en IBM COS

         Args:
             name (str):  Nombre de objeto en COS a cargar.

         Kwargs:
             bucket_name (str):  depósito de IBM COS a usar.

        Returns:
            obj. Objeto descargado.
     """
    return cos.get_object_in_cos(name, bucket_name)
