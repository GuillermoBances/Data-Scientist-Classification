import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from feature_engine.imputation import CategoricalImputer
from sklearn.model_selection import train_test_split
from app import cos, init_cols_train, features_num


def make_dataset_train(data, timestamp, target, cols_to_remove):

    """
        Función que permite crear el dataset usado para el entrenamiento
        del modelo.

        Args:
           data (List):  Lista con la observación llegada por request.
           model_info (dict):  Información del modelo en producción.

        Returns:
           DataFrame. Dataset a inferir.
    """
    print('---> Getting data')
    data_df = get_raw_data_from_request(data)
    print('---> Train / test split')
    train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42)
    print('---> Transforming data')
    train_df, test_df = transform_data(train_df, test_df, timestamp, target, cols_to_remove)
    print('---> Preparing data for training')
    train_df, test_df = pre_train_data_prep(train_df, test_df, timestamp, target)

    return train_df.copy(), test_df.copy()

def get_raw_data_from_request(data):

    """
        Función para obtener nuevas observaciones desde request

        Args:
           data (List):  Lista con la observación llegada por request.

        Returns:
           DataFrame. Dataset con los datos de entrada.
    """
    if isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=init_cols_train)
    else:
        data = pd.DataFrame([data], columns=init_cols_train)
    return data

def transform_data(train_df, test_df, timestamp, target, cols_to_remove):
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

    print('------> Saving initial columns')
    cos.save_object_in_cos(train_df.columns, 'initial_encoded_columns', timestamp)

    print('---------> Removing unnecessary columns')
    # eliminando columnas no empleadas en el entrenamiento
    train_df = remove_unwanted_columns(train_df, cols_to_remove)
    test_df = remove_unwanted_columns(test_df, cols_to_remove)

    return train_df.copy(), test_df.copy()


def pre_train_data_prep(train_df, test_df, timestamp, target):

    """
        Función que realiza las últimas transformaciones sobre los datos
        antes del entrenamiento (imputación de nulos, escalado y label encoding para variables categoricas)

        Args:
            data_df (DataFrame):  Dataset de entrada.
            model_info (dict):  Información del modelo en producción.

        Returns:
            DataFrame. Datasets de salida.
    """

    train_target = pd.DataFrame(train_df[target])
    test_target = pd.DataFrame(test_df[target])
    train_df = train_df.drop(columns=[target], axis = 1)
    test_df = test_df.drop(columns=[target], axis = 1)

    train_df_num = train_df[features_num]
    train_df_cat = train_df.drop(features_num, axis = 1)
    test_df_num = test_df[features_num]
    test_df_cat = test_df.drop(features_num, axis = 1)

    train_df_num = preprocess_numerical_data(train_df_num,timestamp)
    train_df_cat = preprocess_categorical_data(train_df_cat,timestamp)
    test_df_num = preprocess_numerical_data(test_df_num,timestamp)
    test_df_cat = preprocess_categorical_data(test_df_cat,timestamp)

    train_target = train_target.reset_index(drop=True)
    test_target = test_target.reset_index(drop=True)
    train_df = pd.concat([train_df_num, train_df_cat, train_target], axis=1)
    test_df = pd.concat([test_df_num, test_df_cat, test_target], axis=1)

    # agregando las columnas dummies faltantes en los datos de entrada

    print('---------> Saving final columns')
    final_columns = train_df.drop(columns=[target], axis = 1)
    cos.save_object_in_cos(final_columns.columns, 'final_encoded_columns', timestamp)
    
    return train_df.copy(), test_df.copy()


def remove_unwanted_columns(df, cols_to_remove):
    """
        Función para quitar variables innecesarias

        Args:
           df (DataFrame):  Dataset.

        Returns:
           DataFrame. Dataset.
    """
    return df.drop(columns=cols_to_remove)


def preprocess_numerical_data(data_df, timestamp):

    """
        Función para procesado de datos numericos

        Args:
           data_df (DataFrame):  Dataset de entrada.

        Returns:
           DataFrame. Datasets de salida.
    """
    imputer = SimpleImputer(strategy="median") 
    data_df_num = input_missing_values(data_df, imputer)
    print('---------> Saving numerical imputer')
    cos.save_object_in_cos(imputer, 'imputer_num', timestamp)
    return data_df_num.copy()


def preprocess_categorical_data(data_df, timestamp):

    """
        Función para procesado de datos categoricos

        Args:
           data_df (DataFrame):  Dataset de entrada.

        Returns:
           DataFrame. Datasets de salida.
    """
    imputer = CategoricalImputer(imputation_method='frequent')
    data_df_cat = input_missing_values(data_df, imputer)
    print('---------> Saving categorical imputer')
    cos.save_object_in_cos(imputer, 'imputer_cat', timestamp)
    data_df_cat = label_cat(data_df_cat)
    return data_df_cat.copy()


def input_missing_values(data_df, imputer):

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
    print('---------> Label encoding for categorical data')
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

def remove_missing_targets(df, target):
    """
        Función para quitar los valores nulos en la variable objetivo

        Args:
           df (DataFrame):  Dataset.

        Returns:
           DataFrame. Dataset.
    """
    return df[~df[target].isna()].copy()