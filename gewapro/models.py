# Model creation functions
import tensorflow as tf
import keras
# from pydantic import BaseModel
from typing import Literal
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA, TruncatedSVD
import numpy as np
from mlflow.models import infer_signature, model
import mlflow.sklearn
import mlflow.xgboost
import mlflow.keras
import mlflow.pyfunc
from sklearn.exceptions import NotFittedError
from functools import partial
from typing import Callable
import warnings
from gewapro.cache import cache
import pandas as pd
import os
from contextlib import contextmanager
from gewapro.util import add_notes

def get_models():
    client = mlflow.MlflowClient()
    print("Searching for models, this may take a minute...")
    data = client.search_registered_models()
    print("Got all models.")
    models = []
    for model in data:
        models.append(model.name)
    return models

@contextmanager
def modify_message(*exceptions: Exception, prepend_msg: str = "", append_msg: str = "", replace: dict[str,str] = {}, notes: dict[str,str] = {}):
    if any(invalid_excepts := [not (isinstance(excep, Exception) or "Exception" in str(excep)) for excep in exceptions]):
        raise ValueError(f"Got non-ExceptionType in exceptions arguments: {[t[0] for t in zip(exceptions,invalid_excepts) if t[1]==1]}")

    def _replace(string: str, replacer: dict[str,str]):
        for replace in replacer:
            string = string.replace(replace, replacer[replace])
        print("returning final string",string)
        return string

    def _add_notes(err: Exception, checker: dict[str,str]):
        for string in checker:
            if string in err.args[0]:
                err.add_note(checker[string])
        return err

    if not exceptions:
        exceptions = Exception
    try:
        yield # You can return something if you want, that gets picked up in the 'as'
    except exceptions as err:
        err.args = ((f"{prepend_msg}{_replace(arg,replace)}{append_msg}" if i == 0 else arg) for i,arg in enumerate(err.args))
        raise _add_notes(err, notes)
    finally:
        pass

VALID_MODEL_TYPES = get_models()


class ModelInfo:
    """Info on a model, instantiated by ``model`` (later retrievable as attribute), or instantiated using the ``from_database`` class method
    
    Attributes: ``pca_method``, ``pca_components``, ``pca_random_seed`` & ``dt_correcting``

    Methods: ``get_and_check_transformer``
    """
    model: mlflow.pyfunc.PyFuncModel
    version: int
    pca_method: PCA|TruncatedSVD
    pca_components: int
    pca_random_seed: int
    dt_correcting: bool

    def __init__(self, model: mlflow.pyfunc.PyFuncModel):
        pca_method = model.metadata.metadata.get("PCA method", "sklearn.decomposition.PCA")
        if pca_method == (pca_method_name := "sklearn.decomposition.PCA"):
            pca_method = PCA
        elif pca_method == (pca_method_name := "sklearn.decomposition.TruncatedSVD"):
            pca_method = TruncatedSVD
        elif pca_method is None:
            pass
        else:
            raise ValueError(f"Unknown PCA decomposition method '{pca_method}' found in model, known methods are 'sklearn.decomposition.PCA' and 'sklearn.decomposition.TruncatedSVD'")
        try:
            dt_correcting = bool(model.metadata.metadata.get("dT correcting", 0))  # Whether the model predicts a correction to dT or whole new value
            pca_random_seed = model.metadata.metadata.get("PCA random seed", None)
            pca_components = int(model.metadata.signature.inputs.inputs[0].shape[-1])
        except Exception as err:
            valerr = ValueError("Failed to create ModelInfo")
            raise err from valerr
        self.model = model
        self.model_name = "<Unknown>"
        self.version = None
        self.pca_method = pca_method
        self._pca_method_name = "'"+pca_method_name+"'" if pca_method else None
        self.pca_components = pca_components
        self.pca_random_seed = pca_random_seed
        self.dt_correcting = dt_correcting
    
    def get_transformer(self) -> PCA|TruncatedSVD:
        """Gets transformer model (PCA or TruncatedSVD) from model parameters"""
        return self.pca_method(self.pca_components, random_state=self.pca_random_seed) if self.pca_method else None

    def check_transformer(self, PCA_fit: PCA|TruncatedSVD|None) -> PCA|TruncatedSVD|None:
        """Checks validity of passed transformer agains model parameters, then returns it"""
        if isinstance(PCA_fit, (PCA,TruncatedSVD)) and not isinstance(PCA_fit, self.pca_method):
            raise ValueError(f"Expected {self.pca_method if self.pca_method else 'no PCA method'}, but got PCA_fit: {PCA_fit}")
        elif isinstance(PCA_fit, (PCA,TruncatedSVD)):
            if not getattr(PCA_fit,"n_samples_",False):
                raise ValueError(f"PCA_fit was not yet fitted, got unfitted {PCA_fit}")
            return PCA_fit
        elif PCA_fit is None and self.pca_method is None:
            return PCA_fit
        else:
            raise ValueError(f"Unknown PCA decomposition method '{PCA_fit}' passed to PCA_fit, known methods are 'sklearn.decomposition.PCA' and 'sklearn.decomposition.TruncatedSVD'")
    
    @classmethod
    def from_database(self, model_name: str, model_version: int) -> "ModelInfo":
        replacer = {f"Model Version (name={model_name}, version={model_version}) not found": "Unknown model version",
                    f"Registered Model with name={model_name} not found": "Invalid model name",
                    "RESOURCE_DOES_NOT_EXIST: ": ""}
        notes = {"Invalid model name": f"Valid model names are: {', '.join(VALID_MODEL_TYPES)}"}
        with modify_message(prepend_msg=f"Failed to load {model_name} v{model_version}: ", replace=replacer, notes=notes):
            regressor = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
        modelinfo = ModelInfo(regressor)
        modelinfo.model_name, modelinfo.version = model_name,model_version
        return modelinfo
    
    def __str__(self):
        title = f"ModelInfo for {self.model_name} {'v'+str(v) if (v:=self.version) else ''}"
        bar = "="*42  # Answer to everything
        model_info = "MLFlow info:\n "+f"{self.model}".rstrip()
        pca_str = f"PCA method: {self._pca_method_name}\nPCA components: {self.pca_components}\nPCA random seed: {self.pca_random_seed}" if self.pca_method else "<No PCA method>"
        dt_correcting_str = f"Is dt correcting: {self.dt_correcting}"
        return "\n".join([title,bar,model_info,pca_str,dt_correcting_str])+"\n"


def print_warnings(warning_format: str = "[WARNING] <TyPe>: <Message>"):
    """Catches all warnings that the function execution raises and prints them according to `warning_format`

    Follows the capitalization of \\<type\\> and \\<message\\>, e.g. "UserWarning: test warning you CANNOT miss!" with format "[\\<TYPE\\>]
    \\<Message\\>" will be printed as "[USERWARNING] Test warning you CANNOT miss!"
    """
    if not isinstance(warning_format, str):
        raise TypeError("warning_format must be a string")
    def warn_decorator(func: Callable, warning_format: str):
        return warn_wrapper(func, warning_format=warning_format)
    return partial(warn_decorator, warning_format=warning_format)

def warn_wrapper(wrapped: Callable, warning_format: str):
    """Warning wrapper for functions, it is advised to use the function decorator ``@print_warnings()`` instead"""
    # Function that replaces type and message in the format with values
    def _replace_from_format(formatter: str, type: str, message: str):
        replacer = {formatter[(sl:=formatter.lower().find("<type>")):sl+6]:str(type),
                    formatter[(sl:=formatter.lower().find("<message>")):sl+9]:str(message)}
        for k,v in replacer.items():
            for stringmethod in ["lower","upper","capitalize"]:
                if k and k[1:-1] == k[1:-1].__getattribute__(stringmethod)():
                    if stringmethod == "capitalize":
                        replacer[k] = v.upper()[:1]+v[1:]
                    else:
                        replacer[k] = v.__getattribute__(stringmethod)()
            formatter = formatter.replace(k,replacer[k]) if k else formatter
        return formatter

    # Create the wrapped function
    def warn_func(*args, **kwargs):
        # Catch all warnings
        with warnings.catch_warnings(record=True) as caught_warnings:
            output = wrapped(*args, **kwargs)
            for warning in caught_warnings:
                # Print each warning as it is caught
                print(_replace_from_format(warning_format,warning.category.__name__,warning.message))
            return output
    return warn_func


def model_type(model) -> str:
    if isinstance(model, MLPRegressor) or str(getattr(model,"loader_module","")) == "mlflow.sklearn":
        return "SKLearnNN"
    elif isinstance(model, xgb.XGBRegressor) or str(getattr(model,"loader_module","")) == "mlflow.xgboost":
        return "XGBTree"
    elif isinstance(model, keras.Sequential) or str(getattr(model,"loader_module","")) == "mlflow.keras":
        return "KerasNN"
    return f"{model.__module__} {model.__class__} {model.__qualname__}"

def regressor_model(type: Literal["SKLearn","TensorFlow","XGBoost"], pca_components: int, hidden_layers: list[int] = [], activation = "relu", alpha = 1e-4, max_iter=200, **kwargs):
    """Initializes a regressor model for training
    
    See the various documentation pages for model argument inputs:
    - SKLearn: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
    - TensorFlow: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
    - XGBoost: https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor

    Arguments
    ---------
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
        Max number of training iterations or epochs in case of Neural Networks.
    """
    if type.lower() in ["sklearn", "sk"]:
        return MLPRegressor(hidden_layer_sizes = hidden_layers,
                            activation = activation,
                            solver = kwargs.pop("solver","adam"),
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

@print_warnings()
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
    if (m_type := model_type(model)) in ["SKLearnNN", "XGBTree"]:
        return model.predict(data)
    elif m_type == "KerasNN":
        return model.predict(data).transpose()[0]
    else:
        raise ValueError(f"Model type not recognized: '{m_type}'")
    
def loggable_model_params(model):
    if (m_type := model_type(model)) == "SKLearnNN":
        return {
            "Hidden layers": str(model.hidden_layer_sizes).replace(",",""), 
            "Activation function": model.activation,
            "Solver": model.solver,
            "Alpha": model.alpha,
            "Max epochs": model.max_iter,
        }
    elif m_type == "XGBTree":
        return {
            "Max tree depth": model.max_depth, 
            "Number of estimators": model.n_estimators, 
            "Max leaves": model.max_leaves or "no maximum",
            "Tree method": model.tree_method,
        }
    elif m_type == "KerasNN":
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
    else:
        raise ValueError(f"Model type not recognized: '{m_type}'")
    
def loggable_model_metrics(fitted_model, data_test, labels_test):
    if (m_type := model_type(fitted_model)) == "SKLearnNN":
        return {
            "Loss final": fitted_model.loss_,
            "Loss min.": fitted_model.best_loss_,
            "Loss min. epoch": fitted_model.loss_curve_.index(fitted_model.best_loss_),
            "Validation score R2": fitted_model.score(data_test, labels_test),
            "Iterations/epochs": fitted_model.n_iter_,
            "t": fitted_model.t_,
        }
    elif m_type == "XGBTree":
        return {}
    elif m_type == "KerasNN":
        return {}
    else:
        raise ValueError(f"Model type not recognized: '{m_type}'")

@print_warnings()
def log_model(fitted_model, d_train: np.ndarray, predicted_train: np.ndarray, PCA_seed: int, PCA_method: str, dT_correcting: bool = False) -> model.ModelInfo:
    """Model type agnostic logging function for MLFlow. Can log keras' Sequential NN model, xgboost's XGBoostedTree and sklearn's MLPRegressorModel
    
    INFO: predicted_train is only used to infer the function signature (so not fully logged)"""
    if (m_type := model_type(fitted_model)) == "SKLearnNN":
        return mlflow.sklearn.log_model(
            sk_model=fitted_model,
            artifact_path="sk_models",
            signature=infer_signature(d_train, predicted_train),
            input_example=d_train[:1],
            registered_model_name="MLPRegressorModel",
            metadata={"PCA random seed": PCA_seed,
                      "PCA method": PCA_method,
                      "dT correcting": dT_correcting}
        )
    elif m_type == "XGBTree":
        return mlflow.xgboost.log_model(
            xgb_model=fitted_model,
            artifact_path="xgboost_models",
            signature=infer_signature(d_train, predicted_train),
            input_example=d_train[:1],
            registered_model_name="XGBoostedTree",
            metadata={"PCA random seed": PCA_seed,
                      "PCA method": PCA_method,
                      "dT correcting": dT_correcting}
        )
    elif m_type == "KerasNN":
        return mlflow.keras.log_model(
            model=fitted_model,
            artifact_path="keras_models",
            signature=infer_signature(d_train, predicted_train),
            input_example=d_train[:1],
            registered_model_name=fitted_model.name,
            metadata={"PCA random seed": PCA_seed,
                      "PCA method": PCA_method,
                      "dT correcting": dT_correcting}
        )
    else:
        raise ValueError(f"Model type not recognized: '{m_type}'")

def _get_model_types(runs: pd.DataFrame):
    found_types = []
    print(runs["tags.mlflow.log-model.history"][0])
    if runs["tags.mlflow.log-model.history"].str.contains('"artifact_path": "xgboost_models"').sum() > 0:
        found_types.append("XGBoostedTree")
    if runs["tags.mlflow.log-model.history"].str.contains('"artifact_path": "sk_models"').sum() > 0:
        found_types.append("MLPRegressorModel")
    return found_types

@cache(cache_dir=os.path.join("data","cache"))
def _get_version_map_for_length(exp_id: list[str], experiment_length: int):
    verbose = 0
    all_runs: pd.DataFrame = mlflow.search_runs(experiment_ids=exp_id,search_all_experiments=True)
    s = all_runs["tags.mlflow.log-model.history"][0]
    mapping = {}
    for model in _get_model_types(all_runs):
        raised, i = 0, 0
        while not raised:
            i += 1
            try:
                mapping[f"{model}_v{i}"] = str(mlflow.pyfunc.load_model(model_uri=f"models:/{model}/{i}").metadata.metadata["PCA random seed"])
            except OSError:
                continue
            except TypeError:
                continue
            except Exception as e:
                if verbose:
                    print(f"{e.__class__.__name__} was raised for {model}, ending mapper loop: {e}")
                raised = 1

    # Create model version series & Remove duplicate indices
    model_version = pd.DataFrame(data={"model_version":list(mapping.keys())},index=list(mapping.values()))
    model_version = model_version[~model_version.index.duplicated(keep='first')]

    # Change index to string type (like in all_runs) and set index of all_runs as column, then later set back to index
    model_version.index = model_version.index.astype('str')
    all_runs["index"] = all_runs.index
    runs_df = all_runs[["index","run_id","params.PCA random seed"]].set_index("params.PCA random seed",drop=True)
    runs_df["model_version"] = model_version["model_version"]
    runs_df = runs_df.set_index("index",drop=True)
    return runs_df

def get_model_version_map(exp_id: list[int]|int):
    if not all([isinstance(i,int) for i in (exp_id if isinstance(exp_id, (list,tuple)) else [exp_id])]):
        raise ValueError("exp_ids must be an integer or list of integers")
    exp_length = len(mlflow.search_runs(experiment_ids=(exp_id := [str(id) for id in exp_id]),search_all_experiments=True))
    return _get_version_map_for_length(exp_id, exp_length)