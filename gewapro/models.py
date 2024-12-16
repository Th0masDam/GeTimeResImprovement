# Model creation functions
import tensorflow as tf
import keras
from typing import Literal
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import numpy as np
from mlflow.models import infer_signature
import mlflow.sklearn
import mlflow.xgboost
import mlflow.keras
from sklearn.exceptions import NotFittedError

def model_type(model) -> str:
    if isinstance(model, MLPRegressor) or str(getattr(model,"loader_module","")) == "mlflow.sklearn":
        return "SKLearnNN"
    elif isinstance(model, xgb.XGBRegressor) or str(getattr(model,"loader_module","")) == "mlflow.xgboost":
        return "XGBTree"
    elif isinstance(model, keras.Sequential) or str(getattr(model,"loader_module","")) == "mlflow.keras":
        return "KerasNN"
    return "Unknown"

def regressor_model(type: Literal["SKLearn","TensorFlow","XGBoost"], pca_components: int, hidden_layers: list[int] = [], activation = "relu", alpha = 1e-4, max_iter=200, **kwargs):
    """Initializes a regressor model for training
    
    See the various documentation pages for model argument inputs:
    - SKLearn: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
    - TensorFlow: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
    - XGBoost: https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor

    Parameters
    ----------
    type: Literal["SKLearn","TensorFlow","XGBoost"]
        The type of model to instantiate
    pca_components: int
        The number of PCA components the model input uses
    hidden_layers: list[int], default []
        Hidden layers to use, only applicable to Neural Network models
    activation: str, default "relu"
        Activation function per layer (when a list of strings) or activation function to apply to all layers (when a string). Only
        available for Neural Network models
    alpha: float, default 1e-4
        The alpha parameter for Neural Network models only
    max_iter: int, default 200
        Max number of training iterations or epochs in case of Neural Networks. For regression trees this is max_depth of the tree
    """
    if type.lower() in ["sklearn", "sk"]:
        return MLPRegressor(hidden_layer_sizes = hidden_layers,
                            activation = activation,
                            solver = "adam",
                            alpha = alpha,
                            max_iter = max_iter,
                            **kwargs)
    elif type.lower() in ["xgboost","xgb"]:
        tree_method=kwargs.pop("tree_method","hist")
        device=kwargs.pop("device","cuda")
        return xgb.XGBRegressor(tree_method=tree_method, device=device, **kwargs)  # XGBRFRegressor (random forest, maybe better?)
    elif type.lower() not in ["tensorflow", "tf"]:
        raise ValueError("model type must be either SKLearn (SK), TensorFlow (TF) or XGBoost (XGB)")

    # Check values for TensorFlow regressor
    if sum([isinstance(l_s, int) for l_s in hidden_layers]) != len(hidden_layers):
        raise ValueError("hidden_layers must be a list of integers")
    elif isinstance(activation, (list, tuple)):
        if len(hidden_layers) != len(activation):
            raise ValueError("Number of activation functions must be equal to the number of hidden layers")
    else:
        if isinstance(activation, str):
            activation = [activation]*len(hidden_layers)
    
    model = keras.Sequential(**kwargs)
    model.add(keras.layers.InputLayer(shape=(pca_components,)))
    for i,layer_size in enumerate(hidden_layers):
        model.add(keras.layers.Dense(layer_size, activation=activation[i]))

    model.add(keras.layers.Dense(1))
    model_name = f"TF_NN_{pca_components}-"+str(hidden_layers)[1:-1].replace(", ","-")+"-1_"+str(activation)[1:-1].replace(", ","-").replace("'","")+f"_{alpha}"
    model.name = model_name
    model._max_iter = max_iter
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer='rmsprop', #keras.optimizers.Adam(),
        # metrics=[keras.metrics.BinaryAccuracy(name="accuracy")]
    )
    return model

def train_model(model, data, labels, label_weights=None, force_train = False):
    """Trains the Keras, SKLearn or XGBoost model and then returns it
    
    - y_weights argument currently only works for Keras models
    - force_train forces training of pre-trained models (otherwise training is skipped)
    """
    if model_type(model) == "KerasNN":
        if model.get_weights()[-1][0] != 0 and not force_train:
            print("KerasNN Sequential model was already trained, skipping training...")
            return model
        print("Training TensorFlow Keras Sequential model...")
        y_weights = label_weights or np.ones(shape=(len(labels),))
        if labels.shape != y_weights.shape:
            raise ValueError(f"Labels and label weights do not have the same dimensions: {labels.shape} \u2260 {y_weights.shape}")
        max_iter = getattr(model,"_max_iter",100)
        model.fit(data, labels, sample_weight=y_weights,epochs=max_iter)
    elif model_type(model) == "SKLearnNN":
        print("Training SKLearn MLPRegressor model...")
        model.fit(data, labels)
    elif model_type(model) == "XGBTree":
        try:
            model.get_booster()
            if not force_train:
                print("XGBRegresser model was already trained, skipping training...")
                return model
        except NotFittedError:
            print("Training XGBoost XGBRegressor model...")
        # dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.NaN)
        model.fit(data, labels)
    return model
    
def predict(model, data):
    """Universal output format for each model type prediction"""
    if model_type(model) in ["SKLearnNN", "XGBTree"]:
        return model.predict(data)
    elif model_type(model) == "KerasNN":
        return model.predict(data).transpose()[0]
    else:
        raise ValueError("model type not recognized")
    
def loggable_model_params(model):
    if model_type(model) == "SKLearnNN":
        return {
            "Hidden layers": str(model.hidden_layer_sizes).replace(",",""), 
            "Activation function": model.activation,
            "Solver": model.solver,
            "Alpha": model.alpha,
            "Max epochs": model.max_iter,
        }
    elif model_type(model) == "XGBTree":
        return {
            "Max tree depth": model.max_depth, 
            "Number of estimators": model.n_estimators, 
            "Max leaves": model.max_leaves or "no maximum",
            "Tree method": model.tree_method,
        }
    elif model_type(model) == "KerasNN":
        layers = [layer.input.shape[1] for layer in model.layers]+[model.layers[-1].output.shape[1]]
        activation = ['']+[layer.get_config()['activation'] for layer in model.layers]
        return {
            "Hidden layers": str(layers[1:-1]).replace(",",""), 
            "Activation function": str(activation[1:-1]).replace(",",""),
            "Solver": model.get_compile_config()['optimizer']['class_name'],
            "Learning rate": model.get_compile_config()['optimizer']['config']['learning_rate'],
            # "Rho Epsilon": model.get_compile_config()['optimizer']['config']['rho'] model.get_compile_config()['optimizer']['config']['epsilon'] 
            "Max epochs": getattr(model, "_max_iter", 100),
        }
    
def loggable_model_metrics(fitted_model, data_test, labels_test):
    if model_type(fitted_model) == "SKLearnNN":
        return {
            "Loss final": fitted_model.loss_,
            "Loss min.": fitted_model.best_loss_,
            "Loss min. epoch": fitted_model.loss_curve_.index(fitted_model.best_loss_),
            "Validation score R2": fitted_model.score(data_test, labels_test),
            "Iterations/epochs": fitted_model.n_iter_,
            "t": fitted_model.t_,
        }
    elif model_type(fitted_model) == "XGBTree":
        return {}
    elif model_type(fitted_model) == "KerasNN":
        return {}
    
def log_model(fitted_model, d_train, predicted_train, PCA_seed):
    if model_type(fitted_model) == "SKLearnNN":
        return mlflow.sklearn.log_model(
            sk_model=fitted_model,
            artifact_path="sk_models",
            signature=infer_signature(d_train, predicted_train),
            input_example=d_train[0],
            registered_model_name="MLPRegressorModel",
            metadata={"PCA random seed": PCA_seed}
        )
    elif model_type(fitted_model) == "XGBTree":
        return mlflow.xgboost.log_model(
            xgb_model=fitted_model,
            artifact_path="xgboost_models",
            signature=infer_signature(d_train, predicted_train),
            input_example=d_train[0],
            registered_model_name="XGBoostedTree",
            metadata={"PCA random seed": PCA_seed},
        )
    elif model_type(fitted_model) == "KerasNN":
        return mlflow.keras.log_model(
            model=fitted_model,
            artifact_path="keras_models",
            signature=infer_signature(d_train, predicted_train),
            input_example=d_train[0],
            registered_model_name=fitted_model.name,
            metadata={"PCA random seed": PCA_seed}
        )