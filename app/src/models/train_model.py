from ..data.make_dataset_train import make_dataset_train
from ..evaluation.evaluate_model import evaluate_model
from app import ROOT_DIR, cos, client
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from cloudant.query import Query
import time

def training_pipeline(data, model_info_db_name='proyecto-final'):
    """
        Función para gestionar el pipeline completo de entrenamiento
        del modelo.

        Args:
            path (str):  Ruta hacia los datos.

        Kwargs:
            model_info_db_name (str):  base de datos a usar para almacenar
            la info del modelo.
    """

    # Carga de la configuración de entrenamiento
    model_config = load_model_config(model_info_db_name)['model config']
    # variable dependiente a usar
    #target = model_config['target']
    target = 'target'
    # columnas a retirar
    cols_to_remove = model_config['cols_to_remove']

    # timestamp usado para versionar el modelo y los objetos
    ts = time.time()

    # carga y transformación de los datos de train y test
    train_df, test_df = make_dataset_train(data, ts, target, cols_to_remove)

    # separación de variables independientes y dependiente
    y_train = train_df[target]
    X_train = train_df.drop(columns=[target]).copy()
    y_test = test_df[target]
    X_test = test_df.drop(columns=[target]).copy()

    # definición del modelo 
    model = modelo_clf(model_config)

    print('---> Training a model with the following configuration:')
    print(model_config)
    # Ajuste del modelo con los datos de entrenamiento
    model.fit(X_train, y_train)

    # guardado del modelo en IBM COS
    print('------> Saving the model {} object on the cloud'.format('model_'+str(int(ts))))
    save_model(model, 'model',  ts)

    # Evaluación del modelo y recolección de información relevante
    print('---> Evaluating the model')
    metrics_dict = evaluate_model(model, X_test, y_test, ts, model_config, cols_to_remove)

    # Guardado de la info del modelo en BBDD documental
    print('------> Saving the model information on the cloud')
    info_saved_check = save_model_info(model_info_db_name, metrics_dict)

    # Check de guardado de info del modelo
    if info_saved_check:
        print('------> Model info saved SUCCESSFULLY!!')
    else:
        if info_saved_check:
            print('------> ERROR saving the model info!!')

    # selección del mejor modelo para producción
    print('---> Putting best model in production')
    put_best_model_in_production(metrics_dict, model_info_db_name)


def training_pipeline_from_cos(data, model_info_db_name='proyecto-final'):
    """
        Función para gestionar el pipeline completo de entrenamiento
        del modelo.

        Args:
            path (str):  Ruta hacia los datos.

        Kwargs:
            model_info_db_name (str):  base de datos a usar para almacenar
            la info del modelo.
    """
    load_data = load_data_from_cos(data)
    # Carga de la configuración de entrenamiento
    model_config = load_model_config(model_info_db_name)['model config']
    # variable dependiente a usar
    target = 'target'
    # columnas a retirar
    cols_to_remove = model_config['cols_to_remove']

    # timestamp usado para versionar el modelo y los objetos
    ts = time.time()

    # carga y transformación de los datos de train y test
    train_df, test_df = make_dataset_train(load_data, ts, target, cols_to_remove)

    # separación de variables independientes y dependiente
    y_train = train_df[target]
    X_train = train_df.drop(columns=[target]).copy()
    y_test = test_df[target]
    X_test = test_df.drop(columns=[target]).copy()

    # definición del modelo 
    model = modelo_clf(model_config)

    print('---> Training a model with the following configuration:')
    print(model_config)

    # Ajuste del modelo con los datos de entrenamiento
    model.fit(X_train, y_train)

    # guardado del modelo en IBM COS
    print('------> Saving the model {} object on the cloud'.format('model_'+str(int(ts))))
    save_model(model, 'model',  ts)

    # Evaluación del modelo y recolección de información relevante
    print('---> Evaluating the model')
    metrics_dict = evaluate_model(model, X_test, y_test, ts, model_config, cols_to_remove)

    # Guardado de la info del modelo en BBDD documental
    print('------> Saving the model information on the cloud')
    info_saved_check = save_model_info(model_info_db_name, metrics_dict)

    # Check de guardado de info del modelo
    if info_saved_check:
        print('------> Model info saved SUCCESSFULLY!!')
    else:
        if info_saved_check:
            print('------> ERROR saving the model info!!')

    # selección del mejor modelo para producción
    print('---> Putting best model in production')
    put_best_model_in_production(metrics_dict, model_info_db_name)


def modelo_clf(model_config):

    tipo_modelo = model_config["model_name"]
    if tipo_modelo == "RandomForest":
        modelo = RandomForestClassifier(n_estimators = model_config['n_estimators'],
        bootstrap = model_config['bootstrap'],
        max_depth = model_config['max_depth'],
        min_samples_leaf = model_config['min_samples_leaf'],
        min_samples_split = model_config['min_samples_split'],
        max_features = model_config['max_features'],
        random_state=42,
        n_jobs=-1)

    elif tipo_modelo == "SVC":
        modelo = SVC(C = model_config['C'], 
        gamma = model_config['gamma'], 
        kernel = model_config['kernel'],
        probability=True)

    elif tipo_modelo == "KNeighbor":
        modelo = KNeighborsClassifier(algorithm = model_config['algorithm'], 
        leaf_size= model_config['leaf_size'], 
        metric = model_config['metric'], 
        n_neighbors = model_config['n_neighbors'], 
        weights= model_config['weights'],
        n_jobs = -1)

    elif tipo_modelo == "XGBoost":
        modelo = XGBClassifier(learning_rate = model_config['learning_rate'], 
        n_estimators = model_config['n_estimators'], 
        max_depth = model_config['max_depth'], 
        min_child_weight = model_config['min_child_weight'], 
        gamma = model_config['gamma'], 
        subsample = model_config['subsample'], 
        colsample_bytree = model_config['colsample_bytree'],
        objective = model_config['objective'], 
        nthread = model_config['nthread'], 
        scale_pos_weight = model_config['scale_pos_weight'],
        reg_alpha = model_config['reg_alpha'], 
        seed = 42,
        n_jobs = -1)

    return modelo

def save_model(obj, name, timestamp, bucket_name='models-uem-pf'):
    """
        Función para guardar el modelo en IBM COS

        Args:
            obj (sklearn-object): Objeto de modelo entrenado.
            name (str):  Nombre de objeto a usar en el guardado.
            timestamp (float):  Representación temporal en segundos.

        Kwargs:
            bucket_name (str):  depósito de IBM COS a usar.
    """
    cos.save_object_in_cos(obj, name, timestamp)


def save_model_info(db_name, metrics_dict):
    """
        Función para guardar la info del modelo en IBM Cloudant

        Args:
            db_name (str):  Nombre de la base de datos.
            metrics_dict (dict):  Info del modelo.

        Returns:
            boolean. Comprobación de si el documento se ha creado.
    """
    db = client.get_database(db_name)
    client.create_document(db, metrics_dict)

    return metrics_dict['_id'] in db


def put_best_model_in_production(model_metrics, db_name):
    """
        Función para poner el mejor modelo en producción.

        Args:
            model_metrics (dict):  Info del modelo.
            db_name (str):  Nombre de la base de datos.
    """

    # conexión a la base de datos elegida
    db = client.get_database(db_name)
    # consulta para traer el documento con la info del modelo en producción
    query = Query(db, selector={'status': {'$eq': 'en_produccion'}})
    res = query()['docs']
    #  id del modelo en producción
    best_model_id = model_metrics['_id']

    # en caso de que SÍ haya un modelo en producción
    if len(res) != 0:
        # se realiza una comparación entre el modelo entrenado y el modelo en producción
        best_model_id, worse_model_id = get_best_model(model_metrics, res[0])
        # se marca el peor modelo (entre ambos) como "NO en producción"
        worse_model_doc = db[worse_model_id]
        worse_model_doc['status'] = 'none'
        # se actualiza el marcado en la BDD
        worse_model_doc.save()
    else:
        # primer modelo entrenado va a automáticamente a producción
        print('------> FIRST model going in production')

    # se marca el mejor modelo como "SÍ en producción"
    best_model_doc = db[best_model_id]
    best_model_doc['status'] = 'en_produccion'
    # se actualiza el marcado en la BDD
    best_model_doc.save()


def get_best_model(model_metrics1, model_metrics2):
    """
        Función para comparar modelos.

        Args:
            model_metrics1 (dict):  Info del primer modelo.
            model_metrics2 (str):  Info del segundo modelo.

        Returns:
            str, str. Ids del mejor y peor modelo en la comparación.
    """

    # comparación de modelos usando la métrica AUC score.
    auc1 = model_metrics1['model_metrics']['roc_auc_score']
    auc2 = model_metrics2['model_metrics']['roc_auc_score']
    print('------> Model comparison:')
    print('---------> TRAINED model {} with AUC score: {}'.format(model_metrics1['_id'], str(round(auc1, 3))))
    print('---------> CURRENT model in PROD {} with AUC score: {}'.format(model_metrics2['_id'], str(round(auc2, 3))))

    # el orden de la salida debe ser (mejor modelo, peor modelo)
    if auc1 >= auc2:
        print('------> TRAINED model going in production')
        return model_metrics1['_id'], model_metrics2['_id']
    else:
        print('------> NO CHANGE of model in production')
        return model_metrics2['_id'], model_metrics1['_id']


def load_model_config(db_name):
    """
        Función para cargar la info del modelo desde IBM Cloudant.

        Args:
            db_name (str):  Nombre de la base de datos.

        Returns:
            dict. Documento con la configuración del modelo.
    """
    db = client.get_database(db_name)
    query = Query(db, selector={'_id': {'$eq': 'model-config'}})
    return query()['docs'][0]

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
