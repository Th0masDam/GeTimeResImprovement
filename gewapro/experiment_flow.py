import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA, TruncatedSVD
from mlflow.data.numpy_dataset import NumpyDataset
import mlflow.sklearn
from mlflow.data import from_numpy
from mlflow.models import infer_signature
import mlflow.pyfunc
import os
from logging import getLogger
from datetime import datetime
import plotly.graph_objects as go
from gewapro.preprocessing import get_waveforms, train_test_split_cond, get_and_smoothen_waveforms
from gewapro.plotting import histogram, energy_scatter_plot
from typing import Callable, Literal
from gewapro.models import regressor_model, train_model, predict, loggable_model_params, loggable_model_metrics, model_type, log_model, get_model_name
from gewapro.util import name_to_vals, warn_wrapper, add_notes
from_numpy = warn_wrapper(from_numpy, "[WARNING] <Message>")

def sort_test_ranges(test_set_ranges: list[int]) -> list[tuple[int,int]]:
    if not (isinstance(test_set_ranges, list) and all([isinstance(x,int) for x in test_set_ranges])):
        raise ValueError("Test_set_ranges must be a list of integer energy bounds (energy in arb. units)")
    return sorted([(x,y) for x,y in list(zip(test_set_ranges[:-1],test_set_ranges[1:])) if x<y])

def create_test_sets(test_set_ranges: list[int],
                     source_data: pd.DataFrame,
                     include_energy: bool = False,
                     select_channels: list = [],
                     limit: Literal["smallest"]|None = "smallest") -> dict[str, pd.DataFrame]:
    """Creates test sets from source_data within test_set_ranges, which are equal in size if limit=='smallest'"""
    test_ranges = sort_test_ranges(test_set_ranges)
    test_sets = {f"{test_range}": get_waveforms(source_data=source_data,
                                                get_indices_map=False,
                                                include_energy=include_energy,
                                                select_energies=test_range,
                                                select_channels=select_channels) for test_range in test_ranges}
    if limit:
        set_limit = min(*[len(df.columns) for df in test_sets.values()])
        if isinstance(limit, int):
            set_limit = min(set_limit, limit)
        return {k: df.iloc[:,:set_limit] for k,df in test_sets.items()}
    else:
        return test_sets

def get_test_set_results(test_sets: dict[str, pd.DataFrame], pca_model: PCA, regressor, which="Tfit") -> tuple[dict[str,float],dict[str,go.Figure]]:
    """Gets test set results for a given dict of test_sets, a fitted PCA model & a regressor"""
    x_to_t = lambda x: 160-(x*4)
    uniform_metrics = {}
    uniformity_artifacts = {}
    for k,df in test_sets.items():
        data_trans = pca_model.transform(df.T.values) if pca_model else df.T.values
        s_labels_t = pd.Series(np.array([name_to_vals(col)[which] for col in df.columns]))
        s_labels_t.name = f"Initial data: "+which
        pred_s_test = pd.Series(s_labels_t - x_to_t(predict(regressor, data_trans)), name=f"{which} - Tpred (test, E\u2208{k})")
        hist = histogram(pd.concat([s_labels_t,pred_s_test], axis=1), [-30,30,0.25], title="Arrival Time Histogram", xaxis_title="Time (ns)", yaxis_title="Prevalence")
        uniformity_artifacts[f"PredictionHistogramE{k[1:-1].replace(', ','-')}"] = hist
        uniform_metrics[f"Uniform test FWHM E{k[1:-1].replace(', ','-')}"] = 2*np.sqrt(2*np.log(2)) * hist._params[pred_s_test.name+" Gaussian"]["sigma"]
    return uniform_metrics, uniformity_artifacts

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
                   pca_method: PCA|TruncatedSVD = PCA,
                   return_regressor: bool = False,
                   uniform_test_set: list[int] = [],
                   test_size: float = 0.5,
                   show_progress_bar: bool = False,
                   log_level: Literal["WARNING","INFO","DEBUG","ERROR"] = "WARNING",
                   which: str = "Tfit",
                   registered_model_name: str = "auto",
                   **kwargs
                   ):
    """Runs & logs an MLFlow experiment with the given parameters, returns a figure as output

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame with source data. So not after processing with get_waveforms().
    data_name: str
        Name of the data set, used for logging (and your own administration). If this name contains 'raw',
        it willassume get_waveforms() has already been run on the data and does not do further processing.
    select_channels: list[int]|int
        Channels to select. Note that when the source data contains only channel A and you pass here channel
        B,there will not be any waveform data and the experiment will fail.
    select_energies: tuple[int,int], default None
        Energies to select from the source data.
    include_energy: bool, default False
        Whether to include the energy column as a predictor (feature).
    model_type: Literal["SKLearn","TensorFlow","XGBoost"], default "SKLearn"
        The type of model to instantiate, SKLearn Neural Network, TensorFlow Neural Network, or XGBoost
        Tree regressor.
    applied_conditions, default None
        Conditions applied to the selection of the source data.
    applied_conditions_names: list, default None
        Name of the conditions, used for logging (and your own administration).
    smoothing_window: int|tuple[int,int], default None
        Not recommended. Smoothing to use.
    smoothing_energy_range: tuple[int,int], default None
        Not recommended. Smoothing energy range to use.
    normalize_after_smoothing: bool, default False
        Not recommended. Whether to normalize again after smoothing.
    remove_nan_waveforms: bool, default False
        Whether to remove completely empty waveforms.
    pca_components: int, default 64
        Number of PCA components.
    pca_method: PCA|TruncatedSVD, default PCA
        PCA method.
    return_regressor: bool, default False
        Whether to return the finally fitted regressor together with the results plot.
    uniform_test_set: list[int], default []
        List of uniform test sets to include in the model evaluation. E.g. [0,500,1500,400,1000] will create sets
        of ranges 0 - 500, 500 - 1500 and 400 - 1000, with equal number of waveforms in each set.
    test_size: float, default 0.5
        Testing set size, by default 50%.
    show_progress_bar: bool, default False
        Whether to show the annoying MLFlow progress bar in the output.
    log_level: Literal["WARNING","INFO","DEBUG","ERROR"], default "WARNING"
        Log level of the output during the experiment.
    which: str, default "Tfit"
        Which label to train the model on.
    
    Extra args for NN
    -----------------
    hidden_layers: list[int] = [16],
    alpha: float = 1e-4,
    max_iterations: int = 2000,
    """
    if applied_conditions and (not isinstance(applied_conditions_names, (list, tuple)) or not applied_conditions_names):
        raise ValueError("If conditions are applied to the data, names of those must be given as a non-empty list in applied_conditions_names")
    start_time = datetime.now()
    # Remove annoying INFOs and progress bars
    os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = "True" if show_progress_bar else "False"
    logger = getLogger("mlflow")
    logger.setLevel(log_level)
    # Get kwargs
    exp_name = kwargs.pop("experiment_name","")
    m_type = kwargs.pop("model_type","SKLearn")
    hidden_layers = kwargs.pop("hidden_layers", [16])
    alpha = kwargs.pop("alpha", 1e-4)
    activation =  kwargs.pop("activation", "relu")
    # solver = kwargs.pop("solver","adam")
    max_iterations = kwargs.pop("max_iterations", 2000)

    # Define useful functions for later use
    x_to_t = lambda x: 160-(x*4)
    t_to_x = lambda t: (160-t)/4

    # Get all data
    if "raw" not in data_name:
        if smoothing_window:
            print(f"[GeWaPro][experiment_flow.run_experiment] Smoothing waveforms within energy range {smoothing_energy_range} over window(s) {smoothing_window}, normalize: {normalize_after_smoothing}")
            data_df = get_and_smoothen_waveforms(source_data_path = data,
                                                include_energy = include_energy,
                                                select_channels = select_channels,
                                                select_energies = select_energies,
                                                smoothing_window = smoothing_window,
                                                apply_to_energies = smoothing_energy_range,
                                                normalize = normalize_after_smoothing)
        else:
            data_df = get_waveforms(source_data=data, get_indices_map=False, include_energy=include_energy, select_energies=select_energies, select_channels=select_channels)
            if uniform_test_set:
                test_ranges = sort_test_ranges(uniform_test_set)
                test_sets = create_test_sets(test_set_ranges=uniform_test_set,
                                             source_data=data,
                                             include_energy=include_energy,
                                             select_channels=select_channels,
                                             limit="smallest")
                print(f"[GeWaPro][experiment_flow.run_experiment] Created uniform test sets of length {len(list(test_sets.values())[0].columns)} in energy (arb. unit) ranges:",test_ranges)
        av_len = None
    else:
        if uniform_test_set:
            raise ValueError("uniform_test_set can only be used without raw waveform data")
        data_df = data
        av_len = "unknown"
    
    # Check and handle empty data
    if data_df.isnull().values.any() and not remove_nan_waveforms:
        print("[WARNING][GeWaPro][experiment_flow.run_experiment] Waveform data contains NaNs, this will cause errors later. Pass remove_nan_waveforms=True or remove the following waveforms manually:")
        display(data_df.loc[:, data_df.isna().any()])
    elif data_df.isnull().values.any():
        remove_cols = [*data_df.loc[:, data_df.isna().any()].columns]
        print("[WARNING][GeWaPro][experiment_flow.run_experiment] Waveform data contains NaNs! Removing columns: "+" & ".join(remove_cols).replace(" &",",",len(remove_cols)-1))
        data_df = data_df.drop(columns=remove_cols)

    # Get labels
    available_length = av_len or get_waveforms(source_data=data, get_indices_map=False, select_channels=select_channels).shape[1]
    labels_t = np.array(cols_list := [name_to_vals(col)[which] for col in data_df.columns])
    if "-" in cols_list:
        raise add_notes(ValueError(f"No '{which}' column found in data, failed to create training labels"), f"Either set the 'which' parameter in run_experiment(...) to an existing column, or add the '{which}' column to your input data.")
    labels_x = t_to_x(labels_t)
    wave_i = np.array([int(col[col.find("[")+1:col.find("]")]) for col in data_df.columns])

    # Transform the data (PCA)
    if pca_components is None or pca_components == len(data_df): # Leave data as is when PCA components is None or data length
        print("[GeWaPro][experiment_flow.run_experiment] Got PCA components equal to data dimension, skipping transform...")
        data_trans = data_df.T.values
        PCA_seed = pca_method_str = model = None
    else:                              # Otherwise raise error (components > data length) or fit and transform
        try:
            pca_method_str = pca_method.__module__[:pca_method.__module__.rfind(".")+1]+pca_method.__qualname__
            PCA_seed = round((592138171 * (datetime.now().timestamp()*9732103 % 38045729)) % 3244034593)
            model: PCA = pca_method(pca_components, random_state=PCA_seed)
            data_trans = model.fit_transform(data_df.T.values)
            pca_var_ratio = model.explained_variance_ratio_
        except Exception as e:
            raise ValueError(f"Failed to fit and transform waveform data with PCA method '{pca_method_str}': {e}") from e

    # Create a conditioned train-test split on the data, with data not passing the condition added to the testing set
    d_train, d_test, l_train, l_test, l_train_t, l_test_t, wi_train, wi_test = train_test_split_cond(data_trans, labels_x, labels_t, wave_i, test_size=test_size, 
                                                                                                     condition=applied_conditions, random_state=42, 
                                                                                                     add_removed_to_test=True)
    print("[GeWaPro][experiment_flow.run_experiment] Divided data in train, test sets:",l_train.shape, l_test.shape," -> total set of",l_train.shape[0]+l_test.shape[0],"/ available",available_length)
    
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
    experiment_name = exp_name or f"{model_type(regr)}, Na22 {'th.'+name_signature[-2:] if 'ecf' in name_signature else name_signature} Ch{channels[0] if len(channels) == 1 else channels}"
    prepend_run_name = "XGBoostedTree" if model_type(regr) == "XGBoostTree" else f"{model_type(regr)}{str(hidden_layers).replace(',','')}"
    run_name = f"{prepend_run_name}_{(l_train.shape[0]+l_test.shape[0])/(np.nan if isinstance(av_len, str) else available_length):.2%}_{datetime.now().strftime("%Y%m%d_%H%M%S")[2:]}"
    experiment = mlflow.set_experiment(experiment_name)
    
    print(f"[GeWaPro][experiment_flow.run_experiment] Finished setup of experiment '{experiment_name}', run '{run_name}' in {datetime.now()-start_time}. Starting model fitting (timed in MLFlow)...")

    with mlflow.start_run(run_name=run_name) as mlflow_run:
        # Creating and training the model, clock starts ticking here
        print("[GeWaPro][experiment_flow.run_experiment] Fitting model...")
        now = datetime.now()
        regr = train_model(regr, d_train, l_train)
        print("[GeWaPro][experiment_flow.run_experiment] Finished fitting model in",datetime.now()-now)

        # Combining labels and creating prediction Series
        s_labels_t = pd.Series(np.append(l_train_t,l_test_t))
        shift = -round(s_labels_t.apply(lambda x: round(x)).mode().iloc[0])
        s_labels_t.name = f"Initial data: {which} {'-' if shift < 0 else '+'} {abs(shift)} ns"
        predicted_train = predict(regr, d_train)
        pred_s_train = pd.Series(l_train_t - x_to_t(predicted_train),name=f"{which} - Tpred (train)")
        pred_s_test = pd.Series(l_test_t - x_to_t(predict(regr, d_test)),name=f"{which} - Tpred (test)")

        # Add histogram with predicted vs actual data
        fig_hist = histogram(pd.concat([s_labels_t+shift,pred_s_train,pred_s_test], axis=1), [-30,30,0.25], title="Arrival Time Histogram", xaxis_title="Time (ns)", yaxis_title="Prevalence")
        print("[GeWaPro][experiment_flow.run_experiment] Created histogram with params: "+str(fig_hist._params).replace('\n','').replace('       '," ").replace("  "," "))
        fwhm_train = 2*np.sqrt(2*np.log(2)) * fig_hist._params[pred_s_train.name+" Gaussian"]["sigma"]
        fwhm_test = 2*np.sqrt(2*np.log(2)) * fig_hist._params[pred_s_test.name+" Gaussian"]["sigma"]
        overtraining_factor = fwhm_test / fwhm_train

        # Get fwhm from the uniform test sets
        uniform_metrics, uniformity_artifacts = get_test_set_results(test_sets=test_sets, pca_model=model, regressor=regr, which=which)
        print(f"[GeWaPro][experiment_flow.run_experiment] Got FWHMs for test set bounds {test_ranges}: {list(uniform_metrics.values())}")

        # Log all parameters, figures, metrics and inputs
        mlflow.set_experiment_tag("BaseModel","SKLearn Neural Network MLPRegressor")
        normalize_str = f" normalized to {normalize_after_smoothing}" if (isinstance(normalize_after_smoothing, int) and normalize_after_smoothing) else ""
        smooth_str = ("single " if isinstance(smoothing_window,int) else "double ")+f"{smoothing_window}".strip("()").replace(", "," - ")+normalize_str
        mlflow.log_params({
            "Model name": registered_model_name if registered_model_name != "auto" else get_model_name(regr),
            "Channels used": str(select_channels).strip("[]").replace(",",""),
            "Energy range used": str(select_energies).strip("[]").replace(", "," - "),
            "Energy included for training": include_energy,
            "Applied conditions": str(applied_conditions_names).replace(",","") if applied_conditions else "-",
            "Used conditionally removed data in test set": True if applied_conditions else "-",
            "Train - Test set shapes": f'{l_train_t.shape} - {l_test_t.shape}'.replace("(","[").replace(",)","]").replace(",","").replace(")","]"),
            "Waveforms used": f'{l_train_t.shape[0]+l_test_t.shape[0]} / available {available_length}',
            "Waveform smoothing": smooth_str if smoothing_window else "None",
            "Smoothing energy range": f"{smoothing_energy_range}".strip("()[]").replace(",","-") if smoothing_energy_range else "-",
            "PCA method": pca_method_str,
            "PCA components": pca_components,
            "PCA random seed": PCA_seed,
            # "PCA explained variance": pca_var_ratio,
            "Tref": which,
        } | loggable_model_params(regr))
        print(f"[GeWaPro][experiment_flow.run_experiment] Logged parameters")
        mlflow.log_figure(fig_hist, "PredictionHistogram.html")
        for figname,figure in uniformity_artifacts.items():
            mlflow.log_figure(figure, figname+".html")
        if model_type(regr) == "SKLearnNN":
            fig = go.Figure(go.Scatter(y=regr.loss_curve_,name="Loss Curve"))
            fig.add_trace(go.Scatter(x=[regr.loss_curve_.index(regr.best_loss_)],y=[regr.best_loss_],name="Loss minimum"))
            fig.update_layout(title='Loss curve plot',
                            xaxis_title="Epoch",
                            yaxis_title="Loss")
            mlflow.log_figure(fig, "LossCurve.html")
        print(f"[GeWaPro][experiment_flow.run_experiment] Logged figures")
        mlflow.log_metrics({
            "FWHM Train": fwhm_train,
            "FWHM Test": fwhm_test,
            "Overtraining factor": overtraining_factor,
            "Train mean":pred_s_train.mean(),
            "Train RMS": pred_s_train.std(),
            "Test mean": pred_s_test.mean(),
            "Test RMS": pred_s_test.std()
        } | loggable_model_metrics(regr, d_test, l_test) | uniform_metrics)
        print(f"[GeWaPro][experiment_flow.run_experiment] Logged metrics")
        mlflow.log_input(dataset_train, context="training")
        mlflow.log_input(dataset_test, context="testing")
        print(f"[GeWaPro][experiment_flow.run_experiment] Logged inputs")

        # Log the model and finish run
        now = datetime.now()
        model_info = log_model(fitted_model=regr,
                               d_train=d_train,
                               predicted_train=predicted_train,
                               PCA_seed=PCA_seed,
                               PCA_method=pca_method_str,
                               run_id=mlflow_run.info.run_id,
                               registered_model_name=registered_model_name)
        print(f"[GeWaPro][experiment_flow.run_experiment] Logged model in {datetime.now()-now}. Run '{run_name}' finished (run ID: {mlflow_run.info.run_id})")
    if return_regressor:
        return fig_hist, regr
    return fig_hist
