import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from mlflow.data.numpy_dataset import NumpyDataset
import mlflow.sklearn
from mlflow.data import from_numpy
from mlflow.models import infer_signature
import mlflow.pyfunc
import os
from datetime import datetime
import plotly.graph_objects as go
from gewapro.preprocessing import get_waveforms, train_test_split_cond, get_and_smoothen_waveforms
from gewapro.plots import histogram, energy_scatter_plot
import warnings
from typing import Callable, Literal
from gewapro.models import regressor_model, train_model, predict, loggable_model_params, loggable_model_metrics, model_type, log_model
# import tensorflow as tf
# import keras

# class RichFigure(go.Figure):
#     """Plotly graph objects Figure that can store parameters in ``params`` attribute"""
#     _params: dict
#     def __init__(self, *args, params, **kwargs):
#         self._params=params
#         super().__init__(*args, **kwargs)
    
#     @property
#     def params(self):
#         return self._params

class Experiment:
    ...

def run_experiment(data: pd.DataFrame,
                   data_name: str,
                   select_channels: list[int]|int,
                   select_energies: tuple[int,int] = None,
                   include_energy: bool = False,
                   applied_conditions = None,
                   applied_conditions_names: list = None,
                   smoothing_window: int|tuple[int,int] = None,
                   smoothing_energy_range: tuple[int,int] = None,
                   normalize_after_smoothing: bool = False,
                   remove_nan_waveforms: bool = False,
                   pca_components: int = 64,
                   return_regressor: bool = False,
                   **kwargs
                   ):
    """Runs & logs an MLFlow experiment with the given parameters, returns a figure as output
    
    Extra args for NN:
    hidden_layers: list[int] = [16],
    alpha: float = 1e-4,
    max_iterations: int = 2000,
    """
    if applied_conditions and not isinstance(applied_conditions_names, (list, tuple)) and not applied_conditions_names:
        raise ValueError("If conditions are applied to the data, names of those must be given as a non-empty list in applied_conditions_names")
    start_time = datetime.now()
    # Get kwargs
    m_type = kwargs.pop("model_type","SKLearn")
    hidden_layers = kwargs.pop("hidden_layers", [16])
    alpha = kwargs.pop("alpha", 1e-4)
    activation =  kwargs.pop("activation","relu"),
    solver = kwargs.pop("solver","adam"),
    max_iterations = kwargs.pop("max_iterations", 2000)

    # Define useful functions for later use
    x_to_t = lambda x: 160-(x*4)
    t_to_x = lambda t: (160-t)/4

    # Get all data
    if "raw" not in data_name:
        if smoothing_window:
            print(f"[MLFlow run] Smoothing waveforms within energy range {smoothing_energy_range} over window(s) {smoothing_window}, normalize: {normalize_after_smoothing}")
            data_df = get_and_smoothen_waveforms(source_data_path = os.path.join("data",data_name),
                                                include_energy = include_energy,
                                                select_channels = select_channels,
                                                select_energies = select_energies,
                                                smoothing_window = smoothing_window,
                                                apply_to_energies = smoothing_energy_range,
                                                normalize = normalize_after_smoothing)
        else:
            data_df = get_waveforms(source_data=data, get_indices_map=False, include_energy=include_energy, select_energies=select_energies, select_channels=select_channels)
        av_len = None
    else:
        data_df = data
        av_len = "unknown"
    
    # Check and handle empty data
    if data_df.isnull().values.any() and not remove_nan_waveforms:
        print("[WARNING] Waveform data contains NaNs, this will cause errors later. Pass remove_nan_waveforms=True or remove the following waveforms manually:")
        display(data_df.loc[:, data_df.isna().any()])
    elif data_df.isnull().values.any():
        remove_cols = [*data_df.loc[:, data_df.isna().any()].columns]
        print("[WARNING] Waveform data contains NaNs! Removing columns: "+" & ".join(remove_cols).replace(" &",",",len(remove_cols)-1))
        data_df = data_df.drop(columns=remove_cols)
    
    # Get labels
    available_length = av_len or get_waveforms(source_data=data, get_indices_map=False, select_channels=select_channels).shape[1]
    data = data_df.values.transpose()
    labels_t = np.array([float([s[s.find("]")+1:s.find("dT")] for s in [col.replace(" ","")]][0]) for col in data_df.columns])
    labels_x = t_to_x(labels_t)
    wave_i = np.array([int(col[col.find("[")+1:col.find("]")]) for col in data_df.columns])

    # Transform the data (PCA)
    PCA_seed = round((592138171 * (datetime.now().timestamp()*9732103 % 38045729)) % 3244034593)
    model = PCA(pca_components, random_state=PCA_seed)
    data_trans = model.fit_transform(data)
    pca_var_ratio = model.explained_variance_ratio_
    # print("pca_var_ratio:",pca_var_ratio)

    # Create a conditioned train-test split on the data, with data not passing the condition added to the testing set
    d_train, d_test, l_train, l_test, l_train_t, l_test_t, wi_train, wi_test = train_test_split_cond(data_trans, labels_x, labels_t, wave_i, test_size=0.5, 
                                                                                                     condition=applied_conditions, random_state=42, 
                                                                                                     add_removed_to_test=True)
    print("[MLFlow run] Divided data in train, test sets:",l_train.shape, l_test.shape," -> total set of",l_train.shape[0]+l_test.shape[0],"/ available",available_length)
    
    # Create datasets for logging
    source_path = os.path.join(os.path.abspath("./data"),data_name)
    dataset_train: NumpyDataset = from_numpy(d_train, source=source_path, name=data_name+" train", targets=l_train)
    dataset_test: NumpyDataset = from_numpy(d_test, source=source_path, name=data_name+" test", targets=l_test)

    # Create a regressor model
    regr = regressor_model(m_type,
                           pca_components=pca_components,
                           hidden_layers=hidden_layers,
                           activation=activation,
                           alpha=alpha,
                           max_iter=max_iterations,                           
                           **kwargs)

    # Create a name for the experiment and start it
    name_signature = data_name[-15:][data_name[-16:].find("-"):-4]
    channels = [select_channels] if isinstance(select_channels, int) else select_channels
    experiment_name = f"{model_type(regr)}, Na22 {'th.'+name_signature[-2:] if 'ecf' in name_signature else name_signature} Ch{channels[0] if len(channels) == 1 else channels}"
    prepend_run_name = "XGBoostedTree" if model_type(regr) == "XGBoostTree" else f"{model_type(regr)}{str(hidden_layers).replace(',','')}"
    run_name = f"{prepend_run_name}_{(l_train.shape[0]+l_test.shape[0])/(np.nan if isinstance(av_len, str) else available_length):.2%}_{datetime.now().strftime("%Y%m%d_%H%M%S")[2:]}"
    experiment = mlflow.set_experiment(experiment_name)
    
    print(f"[MLFlow run] Finished setup of experiment '{experiment_name}', run '{run_name}' in {datetime.now()-start_time}. Starting model fitting (timed in MLFlow)...")
    with warnings.catch_warnings(action="ignore",record=True) as caught_warnings:
        # warnings.simplefilter("ignore")  # Catch all warnings
        with mlflow.start_run(run_name=run_name) as mlflow_run:
            # Creating and training the model, clock starts ticking here
            print("Fitting model...")
            now = datetime.now()
            train_model(regr, d_train, l_train)
            print("Finished fitting model in",datetime.now()-now)

            # Combining labels and creating prediction Series
            s_labels_t = pd.Series(np.append(l_train_t,l_test_t))
            shift = -round(s_labels_t.mode().iloc[0])
            s_labels_t.name = f"Initial data: dT {'-' if shift < 0 else '+'} {abs(shift)} ns"
            predicted_train = predict(regr, d_train)
            pred_s_train = pd.Series(l_train_t - x_to_t(predicted_train),name="dT_act - dT_pred (train)")
            pred_s_test = pd.Series(l_test_t - x_to_t(predict(regr, d_test)),name="dT_act - dT_pred (test)")

            # Add histogram with predicted vs actual data
            fig_hist = histogram(pd.concat([s_labels_t+shift,pred_s_train,pred_s_test], axis=1), [-30,30,0.25], title="Arrival Time Histogram", xaxis_title="Time (ns)", yaxis_title="Prevalence")
            print(f"[MLFlow run] Created histogram with params: {fig_hist._params}")
            fwhm_train = 2*np.sqrt(2*np.log(2)) * fig_hist._params[pred_s_train.name+" Gaussian"]["sigma"]
            fwhm_test = 2*np.sqrt(2*np.log(2)) * fig_hist._params[pred_s_test.name+" Gaussian"]["sigma"]
            overtraining_factor = fwhm_test / fwhm_train
            # Log all parameters and metrics
            mlflow.set_experiment_tag("BaseModel","SKLearn Neural Network MLPRegressor")
            normalize_str = f" normalized to {normalize_after_smoothing}" if (isinstance(normalize_after_smoothing, int) and normalize_after_smoothing) else ""
            smooth_str = ("single " if isinstance(smoothing_window,int) else "double ")+f"{smoothing_window}".strip("()").replace(", "," - ")+normalize_str
            mlflow.log_params({
                "Channels used": str(select_channels).strip("[]").replace(",",""),
                "Energy range used": str(select_energies).strip("[]").replace(","," - ")+" eV",
                "Energy included for training": include_energy,
                "Applied conditions": str(applied_conditions_names).replace(",","") if applied_conditions else "-",
                "Used conditionally removed data in test set": True if applied_conditions else "-",
                "Train - Test set shapes": f'{l_train_t.shape} - {l_test_t.shape}'.replace("(","[").replace(",)","]").replace(",","").replace(")","]"),
                "Waveforms used": f'{l_train_t.shape[0]+l_test_t.shape[0]} / available {available_length}',
                "Waveform smoothing": smooth_str if smoothing_window else "None",
                "Smoothing energy range": f"{smoothing_energy_range}".strip("()[]").replace(",","-") if smoothing_energy_range else "-",
                "PCA components": pca_components,
                "PCA random seed": PCA_seed,
                "PCA explained variance": pca_var_ratio,
            } | loggable_model_params(regr))
            # mlflow.log_figure(fig_hist, "PredictionHistogram.html")
            if model_type(regr) == "SKLearnNN":
                fig = go.Figure(go.Scatter(y=regr.loss_curve_,name="Loss Curve"))
                fig.add_trace(go.Scatter(x=[regr.loss_curve_.index(regr.best_loss_)],y=[regr.best_loss_],name="Loss minimum"))
                fig.update_layout(title='Loss curve plot',
                                xaxis_title="Epoch",
                                yaxis_title="Loss")
                mlflow.log_figure(fig, "LossCurve.html")
            mlflow.log_metrics({
                "FWHM Train": fwhm_train,
                "FWHM Test": fwhm_test,
                "Overtraining factor": overtraining_factor,
                "Train mean":pred_s_train.mean(),
                "Train RMS": pred_s_train.std(),
                "Test mean": pred_s_test.mean(),
                "Test RMS": pred_s_test.std()
            } | loggable_model_metrics(regr, d_test, l_test))
            mlflow.log_input(dataset_train, context="training")
            mlflow.log_input(dataset_test, context="testing")

            # Log the model
            model_info = log_model(regr, d_train, predicted_train, PCA_seed)
            # Print each warning as it is caught
            for warning in caught_warnings:
                print(f"[WARNING] {warning.category}: {warning.message}")
            print(f"[MLFlow run] Run '{run_name}' finished. Run ID:", mlflow_run.info.run_id)
    if return_regressor:
        return fig_hist, regr
    return fig_hist

def predict_from_model(on_data: str,
                       energy_range: tuple[int,int],
                       model_version: int,
                       data_dict: dict,
                       model_name: str = "MLPRegressorModel",
                       verbose: bool = True,
                       PCA_transform_on: str = "self",
                       plot_type: Literal["Histogram","EnergyScatter"] = "Histogram",
                       custom_func: Callable = None,
                       **kwargs):
    """Fetches fitted model (with name ``model_name`` & version ``model_version``) and predicts ``on_data`` provided from the ``data_dict`` within ``energy_range``
    
    - If ``PCA_transform_on`` is not given, performs the PCA transform fit on the prediction DataFrame (``data_dict[on_data]``) itself.
    - The ``custom_func`` argument takes the untransformed input dataframe and adds its output to the prediction series.
    """
    x_to_t = lambda x: 160-(x*4)
    t_to_x = lambda t: (160-t)/4
    func = custom_func or (lambda on_data_df: np.zeros(on_data_df.iloc[0].transpose().shape))

    # Get regressor and data in right format
    regressor = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    pca_random_seed = regressor.metadata.metadata["PCA random seed"]
    pca_components = regressor.metadata.signature.inputs.inputs[0].shape[-1]
    dfs: dict[str,pd.DataFrame] = {}
    for string in [on_data,PCA_transform_on]:
        if "raw" in string:
            print("Found 'raw' string in data name, so using the data directly (no waveform getting)...")
            dfs[string] = data_dict[string]
        elif string != "self":
            dfs[string] = get_waveforms(select_energies=energy_range,include_energy=kwargs.pop("include_energy",False),source_data=data_dict[string])
        elif string == "self" and string != on_data:
            dfs[string] = dfs[on_data]

    # # Check if on_data is a subset of PCA_transform_on:
    # if not set(dfs[on_data].columns) <= set(dfs[PCA_transform_on].columns):
    #     raise ValueError("The data to perform the PCA transform on must always contain all the waveforms from the prediction data set")

    # Get labels
    try:
        labels_t = np.array([float([s[s.find("]")+1:s.find("dT")] for s in [col.replace(" ","")]][0]) for col in dfs[on_data].columns])
        s_labels_t = pd.Series(labels_t, name="labels dT")
        s_labels_E = pd.Series(np.array([float([s[s.find(",")+1:s.find("eV")] for s in [col.replace(" ","")]][0]) for col in dfs[on_data].columns]), name="Initial data")
    except Exception as e:
        e.add_note(f"Failed to create labels for columns: {list(dfs[on_data].columns)[:3]}...")
        raise e
    labels_x = t_to_x(s_labels_t.values)
    print(f"Predicting data ({on_data}) for regressor v{model_version} has shape:", dfs[on_data].shape, "with",pca_components,"PCA components and energy range",energy_range) if verbose else None

    # Transform the data
    model = PCA(pca_components, random_state=pca_random_seed)
    print("Random state of PCA:",model.random_state) if verbose else None
    data_trans = model.fit_transform(dfs[PCA_transform_on].values.transpose())
    if PCA_transform_on != "self":
        data_trans = model.transform(dfs[on_data].values.transpose())

    # Combining labels and creating prediction Series
    shift = -round(s_labels_t.mode().iloc[0])
    s_labels_t.name = f"Initial data: dT {'-' if shift < 0 else '+'} {abs(shift)} ns"
    # display(pd.concat([pd.Series(regressor.predict(predict_on_data_trans),name="predicted x"),pd.Series(labels_x,name="labels x")],axis=1))
    if str(regressor.loader_module) in ["mlflow.sklearn", "mlflow.xgboost"]: #regr.predict(d_test).transpose()[0]
        predicted_s = pd.Series(labels_t - x_to_t(regressor.predict(data_trans) + func(dfs[on_data])),name="dT_act - dT_pred")
    else:
        predicted_s = pd.Series(labels_t - x_to_t(regressor.predict(data_trans).transpose()[0] + func(dfs[on_data])),name="dT_act - dT_pred")
    hist_kwargs = {"bins":[-30,30,0.25],"title":f"Prediction Histogram {model_name} v{model_version} on '{on_data}' E-range {energy_range}"} | kwargs
    scatter_kwargs = {"energy_map":s_labels_E} | kwargs
    if kwargs.get("debug",False):
        import plotly.express as px
        display(pd.concat([s_labels_t+shift,predicted_s], axis=1))
        return px.scatter(pd.concat([s_labels_t+shift,predicted_s], axis=1))
    if plot_type.lower() == "histogram":
        return histogram(pd.concat([s_labels_t+shift,predicted_s], axis=1), **hist_kwargs)
    elif plot_type.lower() == "energyscatter":
        return energy_scatter_plot(pd.concat([s_labels_t+shift,predicted_s], axis=1), **scatter_kwargs)
    else:
        raise ValueError("plot_type must be either 'Histogram' or 'EnergyScatter'")
