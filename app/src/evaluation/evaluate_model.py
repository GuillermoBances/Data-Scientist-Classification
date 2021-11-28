from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from datetime import datetime


def evaluate_model(model, X_test, y_test, timestamp, model_config, cols_to_remove):
    """
        Esta función permite realizar una evaluación del modelo entrenado
        y crear un diccionario con toda la información relevante del mismo

        Args:
           model (sklearn-object):  Objecto del modelo entrenado.
           X_test (DataFrame): Variables independientes en test.
           y_test (Series):  Variable dependiente en test.
           timestamp (float):  Representación temporal en segundos.
           model_name (str):  Nombre del modelo

        Returns:
           dict. Diccionario con la info del modelo
    """

    # obtener predicciones usando el modelo entrenado
    y_pred = model.predict(X_test)

    # extraer la importancia de variables
    #feature_importance_values = model.feature_importances_

    # Nombre de variables
    features = list(X_test.columns)
    
    # creación del diccionario de info del modelo
    model_info = {}
    #fi_df = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

    # info general del modelo
    model_info['_id'] = 'model_' + str(int(timestamp))
    model_info['name'] = 'model_' + str(int(timestamp))
    # fecha de entrenamiento (dd/mm/YY-H:M:S)
    model_info['date'] = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    
    # info de los parametros con los que fue entrenado el modelo
    model_info['model_used'] = {}
    model_info['model_used']["model_name"] = model_config["model_name"]
    
    if model_config["model_name"] == "RandomForest":
      model_info['model_used']['n_estimators'] = model_config['n_estimators']
      model_info['model_used']['bootstrap'] = model_config['bootstrap']
      model_info['model_used']['max_depth'] = model_config['max_depth']
      model_info['model_used']['min_samples_leaf'] = model_config['min_samples_leaf']
      model_info['model_used']['min_samples_split'] = model_config['min_samples_split']
      model_info['model_used']['max_features'] = model_config['max_features']

    elif model_config["model_name"] == "SVC":
      model_info['model_used']["C"] = model_config['C'] 
      model_info['model_used']['gamma']  = model_config['gamma'] 
      model_info['model_used']['kernel'] = model_config['kernel']

    elif model_config["model_name"] == "KNeighbor":
      model_info['model_used']['algorithm']  = model_config['algorithm'] 
      model_info['model_used']['leaf_size'] = model_config['leaf_size']
      model_info['model_used']['metric']  = model_config['metric'] 
      model_info['model_used']['n_neighbors'] = model_config['n_neighbors']
      model_info['model_used']['weights']= model_config['weights']

    elif model_config["model_name"] == "XGBoost":
      model_info['model_used']["learning_rate"] = model_config['learning_rate'] 
      model_info['model_used']['n_estimators']  = model_config['n_estimators'] 
      model_info['model_used']['max_depth']  = model_config['max_depth'] 
      model_info['model_used']['min_child_weight'] = model_config['min_child_weight']
      model_info['model_used']['gamma'] = model_config['gamma']
      model_info['model_used']['subsample'] = model_config['subsample']
      model_info['model_used']['colsample_bytree'] = model_config['colsample_bytree']
      model_info['model_used']['objective'] = model_config['objective']
      model_info['model_used']['nthread'] = model_config['nthread']
      model_info['model_used']['scale_pos_weight']= model_config['scale_pos_weight']
      model_info['model_used']['reg_alpha'] = model_config['reg_alpha']

    # objectos usados en el modelo (encoders, imputer)
    model_info['objects'] = {}
    model_info['objects']['cols_to_remove'] = cols_to_remove
    model_info['objects']['initial_encoders'] = 'initial_encoded_columns_'+str(int(timestamp))
    model_info['objects']['final_encoders'] = 'final_encoded_columns_'+str(int(timestamp))
    model_info['objects']['imputer_cat'] = 'imputer_cat_' + str(int(timestamp))
    model_info['objects']['imputer_num'] = 'imputer_num_' + str(int(timestamp))

    # métricas usadas
    model_info['model_metrics'] = {}
    # model_info['model_metrics']['feature_importances'] = dict(zip(fi_df.area, fi_df.importance))
    model_info['model_metrics']['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
    model_info['model_metrics']['accuracy_score'] = accuracy_score(y_test, y_pred)
    model_info['model_metrics']['precision_score'] = precision_score(y_test, y_pred)
    model_info['model_metrics']['recall_score'] = recall_score(y_test, y_pred)
    model_info['model_metrics']['f1_score'] = f1_score(y_test, y_pred)
    model_info['model_metrics']['roc_auc_score'] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    # status del modelo (en producción o no)
    model_info['status'] = "none"

    return model_info



