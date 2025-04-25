# Model creation functions
# import tensorflow as tf
import keras
from datetime import datetime
# from pydantic import BaseModel
from typing import Literal
import xgboost as xgb
from zoneinfo import ZoneInfo
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA, TruncatedSVD
import numpy as np
from mlflow.models import infer_signature, model
import mlflow.sklearn
import mlflow.xgboost
import mlflow.keras
import mlflow.pyfunc
from sklearn.exceptions import NotFittedError
from gewapro.cache import cache
import pandas as pd
import os
from gewapro.util import print_warnings, strictclassmethod, modify_message, pandas_string_rep

@cache("__pycache__",verbose=False)
def _get_models() -> pd.DataFrame:
    client = mlflow.MlflowClient()
    print("[GeWaPro][models.ModelInfo] Searching for valid models, this may take a minute...")
    data = client.search_registered_models()
    print("[GeWaPro][models.ModelInfo] Got all models.")
    models = []
    for model in data:
        models.append(model.name)
    last_updated = datetime.now(tz=ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M:%S+00:00")
    return pd.DataFrame({last_updated:models},index=range(0,len(models)))

def get_models() -> list:
    "Gets a list of all model names of the models that are logged in the mlruns folder"
    model_df = _get_models()
    last_updated = model_df.columns.to_list()[0]
    if (datetime.now(tz=ZoneInfo("UTC")) - pd.to_datetime(last_updated)) > pd.to_timedelta("10s"):
        print(f"[GeWaPro][models.ModelInfo] Got valid model names that were last updated on {last_updated}: {list(model_df[last_updated].values)}")
    else:
        print(f"[GeWaPro][models.ModelInfo] Got valid model names from search: {list(model_df[last_updated].values)}")
    return list(model_df[last_updated].values)

INITIAL_VALID_MODEL_TYPES = get_models()
VALID_MODEL_TYPES_LIST: list = [INITIAL_VALID_MODEL_TYPES]

def update_validity():
    "Updates the currently listed valid models"
    _get_models.clear_cache()
    VALID_MODEL_TYPES_LIST.append(get_models())

class ModelInfo:
    """Info on a model, instantiated by ``model`` (later retrievable as attribute), or instantiated using the ``from_database`` class method
    
    Attributes: ``model``, ``model_name``, ``version``, ``pca_method``, ``pca_components``, ``pca_random_seed`` & ``which``

    Methods: ``get_and_check_transformer``
    """
    model: mlflow.pyfunc.PyFuncModel
    model_name: str
    version: int
    pca_method: PCA|TruncatedSVD
    pca_components: int
    pca_random_seed: int
    which: str

    def __init__(self, model: mlflow.pyfunc.PyFuncModel):
        pca_method = model.metadata.metadata.get("PCA method", "sklearn.decomposition.PCA") if model.metadata.metadata else "sklearn.decomposition.PCA"
        if pca_method == (pca_method_name := "sklearn.decomposition.PCA"):
            pca_method = PCA
        elif pca_method == (pca_method_name := "sklearn.decomposition.TruncatedSVD"):
            pca_method = TruncatedSVD
        elif pca_method is None:
            pass
        else:
            raise ValueError(f"Unknown PCA decomposition method '{pca_method}' found in model, known methods are 'sklearn.decomposition.PCA' and 'sklearn.decomposition.TruncatedSVD'")
        try:
            pca_random_seed = model.metadata.metadata.get("PCA random seed", None) if model.metadata.metadata else None
            which = model.metadata.metadata.get("which", None) if model.metadata.metadata else None
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
        self.which = which
    
    def get_transformer(self) -> PCA|TruncatedSVD:
        """Gets transformer model (PCA or TruncatedSVD) from model parameters"""
        if self.pca_random_seed:
            return self.pca_method(self.pca_components, random_state=self.pca_random_seed) if self.pca_method else None
        raise ValueError("Transformer for this model cannot be gotten, as it has no PCA random seed")

    def check_transformer(self, PCA_fit: PCA|TruncatedSVD|None) -> PCA|TruncatedSVD|None:
        """Checks validity of passed transformer agains model parameters, then returns it"""
        if isinstance(PCA_fit, (PCA,TruncatedSVD)) and not isinstance(PCA_fit, self.pca_method):
            raise ValueError(f"Expected {self.pca_method if self.pca_method else 'no PCA method'}, but got PCA_fit: {PCA_fit}")
        elif isinstance(PCA_fit, (PCA,TruncatedSVD)):
            if (isinstance(PCA_fit,PCA) and not hasattr(PCA_fit,"n_samples_")) or (isinstance(PCA_fit,TruncatedSVD) and not hasattr(PCA_fit,"components_")):
                raise ValueError(f"PCA_fit was not yet fitted, got unfitted {PCA_fit}")
            return PCA_fit
        elif PCA_fit is None and self.pca_method is None:
            return PCA_fit
        else:
            raise ValueError(f"Unknown PCA decomposition method '{PCA_fit}' passed to PCA_fit, known methods are 'sklearn.decomposition.PCA' and 'sklearn.decomposition.TruncatedSVD'")
    
    @strictclassmethod
    def from_database(cls, model_name: str, model_version: int) -> "ModelInfo":
        "Instantiates ModelInfo from model with certain name and version from database"
        replacer = {f"Model Version (name={model_name}, version={model_version}) not found": "Unknown model version",
                    f"Registered Model with name={model_name} not found": "Invalid model name",
                    "RESOURCE_DOES_NOT_EXIST: ": ""}
        notes = {"Invalid model name": f"Valid model names are: {', '.join(VALID_MODEL_TYPES_LIST[-1])}" if len(VALID_MODEL_TYPES_LIST[-1])>0 else "No registered models found, no valid names to display..."}
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
        which_str = f"Which labels have been used: {self.which}"
        return "\n".join([title,bar,model_info,pca_str,which_str])+"\n"


def fitted_PCA(model_version: int, waveforms: pd.DataFrame, model_name: str = "MLPRegressorModel") -> PCA|TruncatedSVD:
    """Gets a fitted_PCA for a certain model version, using the provided waveforms"""
    start = datetime.now()
    print(f"[GeWaPro][models.fitted_PCA] Fitting data for {model_name} v{model_version} on {pandas_string_rep(waveforms)}")
    modelinfo: ModelInfo = ModelInfo.from_database(model_name=model_name,model_version=model_version)
    pca: TruncatedSVD|PCA = modelinfo.get_transformer()
    pca = pca.fit(waveforms.T.values)
    print(f"[GeWaPro][models.fitted_PCA] Fitting finished in",datetime.now()-start)
    return pca

def model_type(model) -> str:
    if isinstance(model, MLPRegressor) or str(getattr(model,"loader_module","")) == "mlflow.sklearn":
        return "SKLearnNN"
    elif isinstance(model, xgb.XGBRegressor) or str(getattr(model,"loader_module","")) == "mlflow.xgboost":
        return "XGBRegressor"
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
            print("[GeWaPro][models.train_model] KerasNN Sequential model was already trained, skipping training...")
            return model
        print("[GeWaPro][models.train_model] Training TensorFlow Keras Sequential model...")
        y_weights = label_weights or np.ones(shape=(len(labels),))
        if labels.shape != y_weights.shape:
            raise ValueError(f"Labels and label weights do not have the same dimensions: {labels.shape} \u2260 {y_weights.shape}")
        max_iter = getattr(model,"_max_iter",100)
        model.fit(data, labels, sample_weight=y_weights,epochs=max_iter)
    elif model_type(model) == "SKLearnNN":
        print("[GeWaPro][models.train_model] Training SKLearn MLPRegressor model...")
        model.fit(data, labels)
    elif model_type(model) == "XGBRegressor":
        try:
            model.get_booster()
            if not force_train:
                print("[GeWaPro][models.train_model] XGBRegresser model was already trained, skipping training...")
                return model
        except NotFittedError:
            print("[GeWaPro][models.train_model] Training XGBoost XGBRegressor model...")
        # dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.NaN)
        model.fit(data, labels)
    return model

def predict(model, data):
    """Universal output format for each model type prediction"""
    if (m_type := model_type(model)) in ["SKLearnNN", "XGBRegressor"]:
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
    elif m_type == "XGBRegressor":
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
    elif m_type == "XGBRegressor":
        return {}
    elif m_type == "KerasNN":
        return {}
    else:
        raise ValueError(f"Model type not recognized: '{m_type}'")

@print_warnings()
def log_model(fitted_model, d_train: np.ndarray, predicted_train: np.ndarray, PCA_seed: int, PCA_method: str, registered_model_name: str = "auto") -> model.ModelInfo:
    """Model type agnostic logging function for MLFlow. Can log keras' Sequential NN model, xgboost's XGBoostedTree and sklearn's MLPRegressorModel
    
    INFO: predicted_train is only used to infer the function signature (so not fully logged)"""
    autoreg = True if registered_model_name == "auto" else False
    if (m_type := model_type(fitted_model)) == "SKLearnNN":
        return mlflow.sklearn.log_model(
            sk_model=fitted_model,
            artifact_path="sk_models",
            signature=infer_signature(d_train, predicted_train),
            input_example=d_train[:1],
            registered_model_name="MLPRegressorModel" if autoreg else registered_model_name,
            metadata={"PCA random seed": PCA_seed,
                      "PCA method": PCA_method}
        )
    elif m_type == "XGBRegressor":
        return mlflow.xgboost.log_model(
            xgb_model=fitted_model,
            artifact_path="xgboost_models",
            signature=infer_signature(d_train, predicted_train),
            input_example=d_train[:1],
            registered_model_name="XGBoostedTree" if autoreg else registered_model_name,
            metadata={"PCA random seed": PCA_seed,
                      "PCA method": PCA_method}
        )
    elif m_type == "KerasNN":
        return mlflow.keras.log_model(
            model=fitted_model,
            artifact_path="keras_models",
            signature=infer_signature(d_train, predicted_train),
            input_example=d_train[:1],
            registered_model_name=fitted_model.name if autoreg else registered_model_name,
            metadata={"PCA random seed": PCA_seed,
                      "PCA method": PCA_method}
        )
    else:
        raise ValueError(f"Model type not recognized: '{m_type}'")

def _get_model_types(runs: pd.DataFrame, verbose: bool = False):
    found_types = []
    print("runs['tags.mlflow.log-model.history'][0]: ", runs["tags.mlflow.log-model.history"][0]) if verbose else None
    if runs["tags.mlflow.log-model.history"].str.contains('"artifact_path": "xgboost_models"').sum() > 0:
        found_types.append("XGBoostedTree")
    if runs["tags.mlflow.log-model.history"].str.contains('"artifact_path": "sk_models"').sum() > 0:
        found_types.append("MLPRegressorModel")
        found_types.append("MLPRegressorModel2")
    return found_types

@cache(cache_dir=os.path.join("data","cache"), ignore_args=["verbose"])
def _get_version_map_for_length(exp_id: list[str], experiment_length: int, verbose:bool=False):
    all_runs: pd.DataFrame = mlflow.search_runs(experiment_ids=exp_id,search_all_experiments=True)
    mapping = {}
    for model in _get_model_types(all_runs):
        raised, i = 0, 0
        print(f"[GeWaPro][models.get_model_version_map] Trying to get mapper for model type {model}...")
        while not raised:
            i += 1
            try:
                mapping[f"{model}_v{i}"] = str(mlflow.pyfunc.load_model(model_uri=f"models:/{model}/{i}").metadata.metadata.get("PCA random seed",np.nan))
            except OSError:
                continue
            except TypeError:
                continue
            except Exception as e:
                print(f"[GeWaPro][models.get_model_version_map] {e.__class__.__name__} was raised for {model}, ending mapper loop: {e}") if verbose else None
                raised = 1

    # Create model version series & Remove duplicate indices
    model_version = pd.DataFrame(data={"model_version":list(mapping.keys()),"run_id":[None]*len(mapping.keys())},index=list(mapping.values()))
    model_version = model_version[~model_version.index.duplicated(keep='first')]
    print("[GeWaPro][models.get_model_version_map] Model version:", pandas_string_rep(model_version,25)) if verbose else None
    display(model_version) if verbose else None

    # Change index to string type (like in all_runs) and set index of all_runs as column, then later set back to index
    model_version.index = model_version.index.astype('str')
    all_runs["index"] = all_runs.index
    all_runs = all_runs.loc[~(all_runs["params.PCA random seed"] == "None")]
    runs_df = all_runs[["index","run_id","params.PCA random seed"]].set_index("params.PCA random seed",drop=True)
    print("[GeWaPro][models.get_model_version_map] runs_df with indexed random seed PCA:", pandas_string_rep(runs_df,25)) if verbose else None
    display(runs_df) if verbose else None
    runs_df["model_version"] = model_version["model_version"]
    print("[GeWaPro][models.get_model_version_map] runs_df after adding model_version column:", pandas_string_rep(runs_df,25)) if verbose else None
    display(runs_df) if verbose else None
    runs_df = runs_df.set_index("index")
    return runs_df

def get_model_version_map(exp_id: list[int]|int):
    """Gets a DataFrame mapper of the MLFlow runs for a model type with columns 'model_version' and 'run_id' ordered by run time"""
    if (exp_id := exp_id if isinstance(exp_id, (list,tuple)) else [exp_id]) and not all([isinstance(i,int) for i in exp_id]):
        raise ValueError("exp_ids must be an integer or list of integers")
    all_runs = mlflow.search_runs(experiment_ids=(exp_id := [str(id) for id in exp_id]),search_all_experiments=True)
    exp_length = len(all_runs)
    if "params.PCA random seed" not in all_runs.columns:
        print(f"[GeWaPro][models.get_model_version_map] No model version map available for experiment(s) {exp_id} of length {exp_length}")
        return pd.DataFrame(columns=["run_id","model_version"]).set_index("run_id")
    return _get_version_map_for_length(exp_id, exp_length).set_index("run_id")