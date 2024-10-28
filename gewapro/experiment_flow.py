import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from mlflow.data.numpy_dataset import NumpyDataset
import mlflow.sklearn
from mlflow.data import from_numpy
from mlflow.models import infer_signature
import os
from datetime import datetime
import plotly.graph_objects as go
from gewapro.preprocessing import get_waveforms, train_test_split_cond, get_and_smoothen_waveforms
from gewapro.plots import histogram
import warnings

# class RichFigure(go.Figure):
#     """Plotly graph objects Figure that can store parameters in ``params`` attribute"""
#     _params: dict
#     def __init__(self, *args, params, **kwargs):
#         self._params=params
#         super().__init__(*args, **kwargs)
    
#     @property
#     def params(self):
#         return self._params

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
                   hidden_layers: list[int] = [16],
                   alpha: float = 1e-4,
                   max_iterations: int = 2000,
                   ):
    """Runs & logs an MLFlow experiment with the given parameters, returns a figure as output"""
    if applied_conditions and not isinstance(applied_conditions_names, (list, tuple)) and not applied_conditions_names:
        raise ValueError("If conditions are applied to the data, names of those must be given as a non-empty list in applied_conditions_names")
    start_time = datetime.now()
    # Define useful functions for later use
    x_to_t = lambda x: 160-(x*4)
    t_to_x = lambda t: (160-t)/4

    # Get all data and label arrays
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
    if data_df.isnull().values.any() and not remove_nan_waveforms:
        print("[WARNING] Waveform data contains NaNs, this will cause errors later. Pass remove_nan_waveforms=True or remove the following waveforms manually:")
        display(data_df.loc[:, data_df.isna().any()])
    elif data_df.isnull().values.any():
        remove_cols = [*data_df.loc[:, data_df.isna().any()].columns]
        print("[WARNING] Waveform data contains NaNs! Removing columns: "+" & ".join(remove_cols).replace(" &",",",len(remove_cols)-1))
        data_df = data_df.drop(columns=remove_cols)
    display(data_df.head())
    available_length = get_waveforms(source_data=data, get_indices_map=False, select_channels=select_channels).shape[1]
    data = data_df.values.transpose()
    labels_t = np.array([float([s[s.find("]")+1:s.find("dT")] for s in [col.replace(" ","")]][0]) for col in data_df.columns])
    labels_x = t_to_x(labels_t)
    wave_i = np.array([int(col[col.find("[")+1:col.find("]")]) for col in data_df.columns])

    # Transform the data
    model = PCA(pca_components)
    data_trans = model.fit_transform(data)
    pca_var_ratio = model.explained_variance_ratio_
    print("pca_var_ratio:",pca_var_ratio)

    # Create a conditioned train-test split on the data, with data not passing the condition added to the testing set
    d_train, d_test, l_train, l_test, l_train_t, l_test_t, wi_train, wi_test = train_test_split_cond(data_trans, labels_x, labels_t, wave_i, test_size=0.5, 
                                                                                                     condition=applied_conditions, random_state=42, 
                                                                                                     add_removed_to_test=True)
    print("[MLFlow run] Divided data in train, test sets:",l_train.shape, l_test.shape," -> total set of",l_train.shape[0]+l_test.shape[0],"/ available",available_length)
    
    # Create datasets for logging
    source_path = os.path.join(os.path.abspath("./data"),data_name)
    dataset_train: NumpyDataset = from_numpy(d_train, source=source_path, name=data_name+" train", targets=l_train)
    dataset_test: NumpyDataset = from_numpy(d_test, source=source_path, name=data_name+" test", targets=l_test)

    # Create a name for the experiment and start it
    name_signature = data_name[-15:][data_name[-16:].find("-"):-4]
    channels = [select_channels] if isinstance(select_channels, int) else select_channels
    experiment_name = f"Sklearn NN, Na22 {'th.'+name_signature[-2:] if 'ecf' in name_signature else name_signature} Ch{channels[0] if len(channels) == 1 else channels}"
    run_name = f"NN{str(hidden_layers).replace(',',' ')}_{(l_train.shape[0]+l_test.shape[0])/available_length:.2%}_{datetime.now().strftime("%Y%m%d_%H%M%S")[2:]}"
    experiment = mlflow.set_experiment(experiment_name)
    
    print(f"[MLFlow run] Finished setup of experiment '{experiment_name}', run '{run_name}' in {datetime.now()-start_time}. Starting model fitting (timed in MLFlow)...")
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("ignore")  # Catch all warnings
        with mlflow.start_run(run_name=run_name) as mlflow_run:
            # Creating and training the model, clock starts ticking here
            regr = MLPRegressor(hidden_layer_sizes = hidden_layers,
                                activation = "relu",
                                solver = "adam",
                                alpha = alpha,
                                max_iter = max_iterations)
            regr.fit(d_train, l_train)

            # Combining labels and creating prediction Series
            s_labels_t = pd.Series(np.append(l_train_t,l_test_t))
            shift = -round(s_labels_t.mean())
            s_labels_t.name = f"Initial data: dT {'-' if shift < 0 else '+'} {abs(shift)} ns"
            predicted_train = regr.predict(d_train)
            pred_s_train = pd.Series(l_train_t - x_to_t(predicted_train),name="dT_act - dT_pred (train)")
            pred_s_test = pd.Series(l_test_t - x_to_t(regr.predict(d_test)),name="dT_act - dT_pred (test)")

            # Add histogram with predicted vs actual data
            fig_hist = histogram(pd.concat([s_labels_t+shift,pred_s_train,pred_s_test], axis=1), [-30,30,0.25], title="Arrival Time Histogram", xaxis_title="Time (ns)", yaxis_title="Prevalence")
            print(f"[MLFlow run] Created histogram with params: {fig_hist._params}")
            fwhm_train = 2*np.sqrt(2*np.log(2)) * fig_hist._params[pred_s_train.name+" Gaussian"]["sigma"]
            fwhm_test = 2*np.sqrt(2*np.log(2)) * fig_hist._params[pred_s_test.name+" Gaussian"]["sigma"]

            # Log all parameters and metrics
            mlflow.set_experiment_tag("BaseModel","SKLearn Neural Network MLPRegressor")
            normalize_str = f" normalized to {normalize_after_smoothing}" if (isinstance(normalize_after_smoothing, int) and normalize_after_smoothing) else ""
            smooth_str = ("single " if isinstance(smoothing_window,int) else "double ")+f"{smoothing_window}".strip("()").replace(", "," - ")+normalize_str
            mlflow.log_params({
                "Channels used": str(select_channels).strip("[]").replace(","," "),
                "Energy range used": str(select_energies).strip("[]").replace(","," - ")+" eV",
                "Energy included for training": include_energy,
                "Applied conditions": str(applied_conditions_names).replace(",","  ") if applied_conditions else "-",
                "Used conditionally removed data in test set": True if applied_conditions else "-",
                "Train - Test set shapes": f'{l_train_t.shape} - {l_test_t.shape}'.replace("(","[").replace(",)","]").replace(",","").replace(")","]"),
                "Waveforms used": f'{l_train_t.shape[0]+l_test_t.shape[0]} / available {available_length}',
                "Waveform smoothing": smooth_str if smoothing_window else "None",
                "Smoothing energy range": f"{smoothing_energy_range}".strip("()[]").replace(",","-") if smoothing_energy_range else "-",
                "PCA components": pca_components,
                "PCA explained variance": pca_var_ratio,
                "Hidden layers": str(regr.hidden_layer_sizes).replace(","," "), 
                "Activation function": regr.activation,
                "Solver": regr.solver,
                "Alpha": regr.alpha,
                "Max epochs": regr.max_iter,
            })
            mlflow.log_figure(fig_hist, "PredictionHistogram.html")
            fig = go.Figure(go.Scatter(y=regr.loss_curve_,name="Loss Curve"))
            fig.add_trace(go.Scatter(x=[regr.loss_curve_.index(regr.best_loss_)],y=[regr.best_loss_],name="Loss minimum"))
            fig.update_layout(title='Loss curve plot',
                            xaxis_title="Epoch",
                            yaxis_title="Loss")
            mlflow.log_figure(fig, "LossCurve.html")
            mlflow.log_metrics({
                "FWHM Train": fwhm_train,
                "FWHM Test": fwhm_test,
                "Loss final": regr.loss_,
                "Loss min.": regr.best_loss_,
                "Loss min. epoch": regr.loss_curve_.index(regr.best_loss_),
                "Validation score R2": regr.score(d_test, l_test),
                "Iterations/epochs": regr.n_iter_,
                "t": regr.t_,
                "Train mean":pred_s_train.mean(),
                "Train RMS": pred_s_train.std(),
                "Test mean": pred_s_test.mean(),
                "Test RMS": pred_s_test.std()
            })
            mlflow.log_input(dataset_train, context="training")
            mlflow.log_input(dataset_test, context="testing")

            # Log the model
            model_info = mlflow.sklearn.log_model(
                sk_model=regr,
                artifact_path="sk_models",
                signature=infer_signature(d_train, predicted_train),
                input_example=d_train,
                registered_model_name="MLPRegressorModel",
            )
            # Print each warning as it is caught
            for warning in caught_warnings:
                print(f"[WARNING] {warning.category}: {warning.message}")
            print(f"[MLFlow run] Run '{run_name}' finished. Run ID:", mlflow_run.info.run_id)
    return fig_hist