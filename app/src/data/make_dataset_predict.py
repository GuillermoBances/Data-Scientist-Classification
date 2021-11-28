import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from app import cos, init_cols_predict, features_num


def make_dataset_predict(data, model_info, cols_to_remove):

    """
        Función que permite crear el dataset usado para el entrenamiento
        del modelo.

        Args:
           data (List):  Lista con la observación llegada por request.
           model_info (dict):  Información del modelo en producción.

        Kwargs:
           model_type (str): tipo de modelo usado.

        Returns:
           DataFrame. Dataset a inferir.
    """

    print('---> Getting data')
    data_df = get_raw_data_from_request(data)
    print('---> Transforming data')
    data_df = transform_data(data_df, model_info, cols_to_remove)
    print('---> Preparing data for training')
    data_df = pre_train_data_prep(data_df, model_info)
    
    return data_df.copy()

def get_raw_data_from_request(data):

    """
        Función para obtener nuevas observaciones desde request

        Args:
           data (List):  Lista con la observación llegada por request.

        Returns:
           DataFrame. Dataset con los datos de entrada.
    """
    if isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=init_cols_predict)
    else:
        data = pd.DataFrame([data], columns=init_cols_predict)
        
    return data


def transform_data(data_df, model_info, cols_to_remove):
    """
        Función que permite realizar las primeras tareas de transformación
        de los datos de entrada.

        Args:
            data_df (DataFrame):  Dataset de entrada.
            model_info (dict):  Información del modelo en producción.
            cols_to_remove (list): Columnas a retirar.

        Returns:
           DataFrame. Dataset transformado.
    """
    print('------> Getting encoded columns from cos')
    # obteniendo las columnas presentes en el entrenamiento desde COS
    enc_key = model_info['objects']['initial_encoders']+'.pkl'
    enc_cols = cos.get_object_in_cos(enc_key)
    # agregando las columnas dummies faltantes en los datos de entrada
    if 'target' in enc_cols:
        enc_cols = enc_cols[enc_cols != 'target']
    data_df = data_df.reindex(columns=enc_cols, fill_value=0)
    print('------> Removing unnecessary columns')
    # eliminando columnas no empleadas en el entrenamiento
    data_df = remove_unwanted_columns(data_df, cols_to_remove)

    return data_df.copy()


def pre_train_data_prep(data_df, model_info):

    """
        Función que realiza las últimas transformaciones sobre los datos
        antes del entrenamiento (imputación de nulos, escalado y label encoding para variables categoricas)

        Args:
            data_df (DataFrame):  Dataset de entrada.
            model_info (dict):  Información del modelo en producción.

        Returns:
            DataFrame. Datasets de salida.
    """
    data_df_num = data_df[features_num]
    data_df_cat = data_df.drop(features_num, axis = 1)

    data_df_num = preprocess_numerical_data(data_df_num,model_info)
    data_df_cat = preprocess_categorical_data(data_df_cat,model_info)
    
    data_df = pd.concat([data_df_num, data_df_cat],axis=1)

    enc_key = model_info['objects']['final_encoders']+'.pkl'
    # obteniendo las columnas presentes en el entrenamiento desde COS
    enc_cols = cos.get_object_in_cos(enc_key)
    # agregando las columnas dummies faltantes en los datos de entrada
    data_df = data_df.reindex(columns=enc_cols, fill_value=0)
    
    return data_df.copy()


def remove_unwanted_columns(df, cols_to_remove):
    """
        Función para quitar variables innecesarias

        Args:
           df (DataFrame):  Dataset.

        Returns:
           DataFrame. Dataset.
    """
    return df.drop(columns=cols_to_remove)


def preprocess_numerical_data(data_df, model_info, bucket_name = 'models-uem-pf'):

    """
        Función para procesado de datos numericos

        Args:
           data_df (DataFrame):  Dataset de entrada.

        Returns:
           DataFrame. Datasets de salida.
    """
    print('------> Getting numerical imputer from cos')
    imputer_int_key = model_info['objects']['imputer_num']+'.pkl'
    data_df_num = input_missing_values(data_df, imputer_int_key)
    #scaler = model_info['objects']['scaler']+'.pkl'
    #scaler_to_fit = cos.get_object_in_cos(scaler, bucket_name)
    #data_df_num = pd.DataFrame(scaler_to_fit.fit_transform(data_df_num), columns=data_df_num.columns)
    return data_df_num.copy()
 

def preprocess_categorical_data(data_df, model_info, bucket_name = 'models-uem-pf'):

    """
        Función para procesado de datos categoricos

        Args:
           data_df (DataFrame):  Dataset de entrada.

        Returns:
           DataFrame. Datasets de salida.
    """
    print('------> Getting categorical imputer from cos')
    imputer_cat_key = model_info['objects']['imputer_cat']+'.pkl'
    data_df_cat = input_missing_values(data_df, imputer_cat_key)
    data_df_cat = label_cat(data_df_cat)
    return data_df_cat.copy()


def input_missing_values(data_df, key, bucket_name = 'models-uem-pf'):

    """
        Función para la imputación de nulos

        Args:
            data_df (DataFrame):  Dataset de entrada.
            key (str):  Nombre del objeto imputador en COS.

        Returns:
            DataFrame. Datasets de salida.
    """

    print('------> Inputing missing values')
    # obtenemos el objeto SimpleImputer desde COS
    imputer = cos.get_object_in_cos(key, bucket_name)
    data_df = pd.DataFrame(imputer.fit_transform(data_df), columns=data_df.columns)

    return data_df.copy()


def label_cat(df):
    """
        Función de label encoder de variables categoricas

        Args:
           df (DataFrame):  Dataset.

        Returns:
           DataFrame. Dataset.
    """
    print('------> Label encoding for categorical data')
    df["tamano_compania"] = pd.DataFrame(df["tamano_compania"].apply(lambda x : '0-10' if x == '<10' else x))
    df["ultimo_nuevo_trabajo"] = pd.DataFrame(df["ultimo_nuevo_trabajo"].apply(lambda x : '4+' if x == '>4' else x))
    bins = pd.DataFrame(df["experiencia"].apply(lambda x : '25' if x == '>20' else x))
    bins = pd.DataFrame(bins["experiencia"].apply(lambda x : '0' if x == '<1' else x))
    bins["experiencia"] = pd.to_numeric(bins["experiencia"])
    labels = ['0-2', '2-5', '5-10', '10-20', '20+']
    df["experiencia"] = pd.cut(bins.experiencia, bins = 5, labels = labels)

    le = LabelEncoder()
    ohe = OneHotEncoder()

    features = df.columns.values
    for i in range(len(features)):
        cat_encoded = le.fit_transform(df[features[i]].astype(str))
        cat_ohe = ohe.fit_transform(cat_encoded.reshape(-1,1))
        df_cat = pd.DataFrame(cat_ohe.toarray(),columns = features[i] + "_" + le.classes_)
        if i == 0:
            data_df = df_cat
        else:
            data_df = pd.concat([data_df, df_cat], axis = 1)


    return data_df



