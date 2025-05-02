import os
import pandas as pd
import numpy as np
import warnings as warns
from datetime import datetime
from typing import Callable, Literal
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.stats import kstest, norm
import plotly.graph_objects as go
import plotly.express as px
import mlflow.pyfunc
from gewapro.models import ModelInfo, predict
from gewapro.functions import gaussian, gaussian_arr, inverse_quadratic, inverse_quadratic_arr
from sklearn.decomposition import PCA, TruncatedSVD
from gewapro.preprocessing import get_waveforms
from gewapro.cache import cache
from gewapro.util import stats, join_strings, pandas_string_rep, add_notes, name_to_vals

DEFAULT_HIST_FITS = {"Gaussian": gaussian, "Inv. Quadratic": inverse_quadratic}
PLOT_SETTINGS: dict[str,str|dict] = {"show_dt":True, "show_pred":True, "which": "Tfit", "default_plot_mode": "Bar", "show_progess_bar": False, "histogram_fits": DEFAULT_HIST_FITS}
os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = "False"

def settings(**new_settings):
    """Update or show the plot settings. Use `settings.reset()` to reset to initial defaults."""
    global PLOT_SETTINGS
    try:
        updated_settings = {setting:value for setting,value in new_settings.items() if PLOT_SETTINGS[setting] != value}
        if updated_settings.get("default_plot_mode","bar") not in ["Bar", "Line", "bar", "line"]:
            raise ValueError(f"default_plot_mode must be 'Bar' or 'Line', got '{updated_settings['default_plot_mode']}'")
        if updated_settings.get("show_progess_bar", False) not in [True, False]:
            raise ValueError("show_progess_bar must be bool")
        if isinstance((new_hist_fits:=updated_settings.get("histogram_fits", {})), dict):
            if any(key in new_hist_fits.keys() for key in ["Gaussian", "Inv. Quadratic"]):
                raise ValueError("Defaults 'Gaussian' & 'Inv. Quadratic' cannot be changed or removed from histogram_fits")
            if invalid_hist_fits := [f"'{n}' ({type(c)})" for n,c in new_hist_fits.items() if not callable(c)]:
                raise ValueError(f"All new custom fit methods must be a callable function, but got invalid {', '.join(invalid_hist_fits)}")
            updated_settings["histogram_fits"] = (new_hist_fits | DEFAULT_HIST_FITS)
        else:
            raise ValueError("histogram_fits must be dict")
    except KeyError as e:
        raise add_notes(ValueError("Could not update settings due to invalid setting: "+str(e)), f"Valid settings are: "+", ".join(list(PLOT_SETTINGS.keys()))) from e
    if new_settings.get("show_progess_bar", False) is True:
        os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = "True"
    elif new_settings.get("show_progess_bar", True) is False:
        os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = "False"
    PLOT_SETTINGS |= new_settings
    if updated_settings:
        if new_hist_fits:
            print("[WARNING][GeWaPro][plotting.settings] Got new functions for histogram fitting, note these may still not work since formats (function signatures) were not checked for validity")
        else:
            updated_settings.pop("histogram_fits")
        print("[GeWaPro][plotting.settings] Updated plot settings: "+", ".join([f"{s} = {v}" for s,v in updated_settings.items()]))
    else:
        print("[GeWaPro][plotting.settings] Settings not changed: "+", ".join([f"{s} = {v}" for s,v in PLOT_SETTINGS.items()]))
    return PLOT_SETTINGS

def _reset_plot_settings():
    """Resets the plot settings to the initial defaults"""
    global PLOT_SETTINGS
    PLOT_SETTINGS = {"show_dt":True, "show_pred":True, "which": "Tfit", "default_plot_mode": "Bar", "show_progess_bar": False, "histogram_fits": DEFAULT_HIST_FITS}
    return PLOT_SETTINGS

settings.reset = _reset_plot_settings
"""Resets the plot settings to the initial defaults"""

def histogram(data: pd.DataFrame|pd.Series,
              bins: list[int] = None,
              warnings: Literal["raise","print","ignore"] = "print",
              add_fits: list[str] = ["Gaussian"],
              mode: Literal["Bar","Line"] = "DEFAULT",
              **options):
    """Returns a histogram with one (Series) or more (DataFrame) traces with fitted Gaussians, with ``options`` kwargs passed to the traces"""
    if not isinstance(warnings, str):
        raise ValueError("warnings must be a string")
    if len(data) == 0 and warnings != "raise":
        if warnings != "ignore":
            print("[WARNING][GeWaPro][plotting.histogram] Got empty data, nothing to plot")
        return go.Figure()
    elif len(data) == 0:
        raise ValueError("Got empty data, nothing to plot")
    if isinstance(data, pd.DataFrame):
        pass
    elif isinstance(data, pd.Series) or isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    else:
        raise ValueError("Data must be a pandas DataFrame or a pandas Series")
    if bins and len(bins) == 3 and all([isinstance(i,(float,int)) for i in bins]) and bins[1] > bins[0] and bins[2] < (bins[1] - bins[0]):
        pass
    else:
        if warnings == "raise" and bins is not None:
            raise ValueError(f"Invalid bins provided: {bins}. Bins must be a 3-long number list [start,end,step] where start must be after end and step must be smaller than start-end interval")
        bins = [data.min().min(), data.max().max(), (data.max().max() - data.min().min()) // 100]
        if warnings != "ignore":
            print(f"[WARNING][GeWaPro][plotting.histogram] No (valid) bins provided, assuming start {bins[0]:.0f}, end {bins[1]:.0f} and step {bins[2]}")
    if not add_fits:
        add_fits = []
    elif not isinstance(add_fits, list) or any([fit_name not in PLOT_SETTINGS['histogram_fits'] for fit_name in add_fits]):
        raise ValueError(f"add_fits must be a list containing {join_strings(PLOT_SETTINGS['histogram_fits'])}")
    mode = PLOT_SETTINGS["default_plot_mode"] if mode.lower() == "default" else mode
    fig = go.Figure()
    options = {"xaxis_range":[bins[0],bins[1]]} | options
    title = options.pop("title","Histogram")
    trace_name = options.pop("trace_name",None)
    colors = options.pop("colors", px.colors.qualitative.Plotly)
    layout_options = {k.replace("layout_",""):v for k,v in options.items() if "axis" in k or k.startswith("layout_")}
    fit_options = {k.replace("fit_",""):v for k,v in options.items() if k.startswith("fit_")}
    options = {k:v for k,v in options.items() if not ("axis" in k or k.startswith("layout_") or k.startswith("fit_"))}
    trace_names = options.pop("trace_names", [trace_name]*len(data.columns))
    params = {}
    data_min, data_max = (data.min().min() // bins[2] ) * bins[2], (data.max().max() // bins[2] ) * bins[2] + bins[2]
    for i,col in enumerate(data.columns):
        trace_data = data[col].values
        x_data = np.arange(data_min+bins[2]/2, data_max, bins[2])
        x_data_s = x_data - bins[2]/2
        x_data_e = x_data + bins[2]/2
        # nearest_x_bins = (bins[0] - data_min) // bins[2], (bins[1] - data_min) // bins[2]
        # print("bin bounds:", bins[0:1], "nearest_x_bins:", nearest_x_bins, "\n", x_data[nearest_x_bins[0]], x_data[nearest_x_bins[1]])
        data_count = np.array([1.0]*len(data))
        y_data = np.array([data_count[(trace_data >= x-bins[2]/2) & (trace_data < x+bins[2]/2)].sum() for x in x_data])
        if mode.lower() == "bar":
            fig.add_trace(go.Bar(x=x_data,
                                 y=y_data,
                                 marker={"line":{"width":0}},
                                 width=bins[2],
                                 name=trace_names[i] or col,
                                 marker_color=colors[i],
                                 customdata=pd.DataFrame(data=[x_data_s,x_data_e]).T,
                                 hovertemplate ='<b>[%{customdata[0]:.2f},%{customdata[1]:.2f}): %{y:.0f}x</b>',
                                **options,
                                ))
        elif mode.lower() == "line":
            fig.add_trace(go.Scatter(x=x_data,
                                     y=y_data,
                                     marker={"line":{"width":0}},
                                     mode="lines",
                                     line_shape="hvh",
                                     name=trace_names[i] or col,
                                     line_color=colors[i],
                                     marker_color=colors[i],
                                     customdata=pd.DataFrame(data=[x_data_s,x_data_e]).T,
                                     hovertemplate ='<b>[%{customdata[0]:.2f},%{customdata[1]:.2f}): %{y:.0f}x</b>',
                                    **options,
                                ))
        else:
            raise ValueError(f"Got invalid mode (must be either \"Line\" or \"Bar\"): {mode}")
        for fit_name,fit_func in PLOT_SETTINGS["histogram_fits"].items():
            if fit_name not in add_fits:
                continue
            try:
                p0 = [y_data.max(), x_data[list(y_data).index(y_data.max())], 10]
                # y_max = y_data[list(x_data).index(bins[0]):list(x_data).index(bins[1])].max()
                # p0 = [y_max, x_data[list(y_data).index(y_max)], 10]
                with warns.catch_warnings(record=True) as w:
                    warns.simplefilter("always")
                    popt, pcov = curve_fit(fit_func, x_data, y_data, p0 = p0)
                    if len(w) > 0 and issubclass(w[-1].category, OptimizeWarning) and warnings != "ignore":
                        print(f"[WARNING][GeWaPro][plotting.histogram] {fit_name} fit optimization failed for initial guess [A,\u03BC,\u03C3] = {p0}: OptimizeWarning: {w[-1].message}")
                min_tb,max_tb = popt[1]-5*popt[2], popt[1]+5*popt[2]
                ksh = kstest(trace_data[(trace_data > min_tb) & (trace_data < max_tb)],cdf=norm.cdf,args=(popt[1],0.5*popt[2])).pvalue
                ks1 = kstest(trace_data[(trace_data > min_tb) & (trace_data < max_tb)],cdf=norm.cdf,args=(popt[1],popt[2])).pvalue
                ksd = kstest(trace_data[(trace_data >min_tb) & (trace_data < max_tb)],cdf=norm.cdf,args=(popt[1],2*popt[2])).pvalue
                params[f"{trace_names[i] or col} {fit_name}"] = dict(
                    zip(fit_func.__code__.co_varnames[1:fit_func.__code__.co_argcount+fit_func.__code__.co_kwonlyargcount], popt, strict=True)
                ) | {"Covariance": pcov, "GoodnessOfFit": 0 if ((nom:=ksd*ksh) <= 0) else 1/np.log((ks1*ks1)/nom)}
                fig.add_trace(go.Scatter(x=x_data,
                                         y=fit_func(x_data, *popt),
                                         line_color=fig.data[2*i].marker.color,
                                         line_dash="dot",
                                         line_shape="spline",
                                         name=f"{fit_name} fit",
                                         hovertemplate = "<b>(%{x},%{y})</b>"+
                                             f'<br>\u03BC: {popt[1]:.4f}<br>\u03C3: {popt[2]:.4f}'+   # mu (popt[1]), sigma (popt[2])
                                             f'<br>Ampl.: {popt[0]/(popt[2]*np.sqrt(2*np.pi)):.2f}<br><b>FWHM: {popt[2]*2*np.sqrt(2*np.log(2)):.4f}</b>',
                                         **fit_options
                                        ))
            except Exception as e:
                if warnings == "print":
                    print(f"[WARNING][GeWaPro][plotting.histogram] {fit_name} fitting failed: {e.__class__.__name__}: {e}")
                elif not warnings in ["ignore","print"]:
                    raise e

    fig.update_layout(barmode='overlay', title=title, **layout_options)
    # Reduce opacity to see all histograms
    fig.update_traces(opacity=0.75)
    fig._params = params
    return fig

def energy_scatter_plot(data: pd.DataFrame|pd.Series,
                        energy_map: pd.Series,
                        warnings: Literal["raise","print","ignore"] = "print",
                        **options):
    """Returns a scatter plot of dT vs Energy with one (Series) or more (DataFrame) traces, with ``options`` kwargs passed to the traces
    
    - ``energy_map`` must be provided to map the integer indices to corresponding energy
    """
    if len(data) == 0 and warnings != "raise":
        if warnings != "ignore":
            print("[WARNING][GeWaPro][plotting.energy_scatter_plot] Got empty data, nothing to plot")
            return go.Figure()
    elif len(data) == 0:
        raise ValueError("Got empty data, nothing to plot")
    if isinstance(data, pd.DataFrame):
        pass
    elif isinstance(data, pd.Series) or isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    else:
        raise ValueError("Data must be a pandas DataFrame or a pandas Series")
    fig = go.Figure()
    title = options.pop("title","Tref - Tpred vs Energy (arb. units)")
    trace_name = options.pop("trace_name",None)
    colors = options.pop("colors", px.colors.qualitative.Plotly)
    layout_options = {k.replace("layout_",""):v for k,v in options.items() if "axis" in k or k.startswith("layout_")}
    fit_options = {k.replace("fit_",""):v for k,v in options.items() if k.startswith("fit_")}
    options = {k:v for k,v in options.items() if not ("axis" in k or k.startswith("layout_") or k.startswith("fit_"))}
    trace_names = options.pop("trace_names", [trace_name]*len(data.columns))
    for i,col in enumerate(data.columns):
        fig.add_trace(go.Scatter(x=energy_map.values,
                                 y=data[col].values,
                                 name=trace_names[i] or col,
                                 mode='markers',
                                 marker_color=colors[i],
                                 hovertemplate ='<b>%{x:.0f}: %{y:.0f} ns</b>',
                                **options,
                                 ))
        # CREATING A MEAN ROLLING AVERAGE CURVE OVER THE DATA POINTS
        # for fit_name,fit_func in {"Gaussian":gaussian}.items():#,"Inv. Quadratic":inv_quadratic_function}.items():
        #     try:
        #         p0 = [y_data.max(), x_data[list(y_data).index(y_data.max())], 10]
        #         # y_max = y_data[list(x_data).index(bins[0]):list(x_data).index(bins[1])].max()
        #         # p0 = [y_max, x_data[list(y_data).index(y_max)], 10]
        #         with warns.catch_warnings(record=True) as w:
        #             warns.simplefilter("always")
        #             popt, _ = curve_fit(fit_func, x_data, y_data, p0 = p0)
        #             if len(w) > 0 and issubclass(w[-1].category, OptimizeWarning) and warnings != "ignore":
        #                 print(f"[WARNING] {fit_name} fit optimization failed for initial guess [A,\u03BC,\u03C3] = {p0}: OptimizeWarning: {w[-1].message}")
        #     except Exception as e:
        #         if warnings == "print":
        #             print(f"[WARNING] {fit_name} fitting failed: {e.__class__.__name__}: {e}")
        #         elif not warnings in ["ignore","print"]:
        #             raise e
        #     params[f"{trace_names[i] or col} {fit_name}"] = dict(
        #         zip(fit_func.__code__.co_varnames[1:fit_func.__code__.co_argcount+fit_func.__code__.co_kwonlyargcount], popt, strict=True)
        #     )
        #     fig.add_trace(go.Scatter(x=x_data,
        #                              y=fit_func(x_data, *popt),
        #                              line_color=fig.data[2*i].marker.color,
        #                              name=f"{fit_name} fit",
        #                              hovertemplate = "<b>(%{x},%{y})</b>"+
        #                                 f'<br>\u03BC: {popt[1]:.4f}<br>\u03C3: {popt[2]:.4f}'+
        #                                 f'<br>Ampl.: {popt[0]/(popt[2]*np.sqrt(2*np.pi)):.2f}<br><b>FWHM: {popt[2]*fhwm:.4f}</b>',
        #                             **fit_options
        #                             ))
    
    fig.update_layout(barmode='overlay', title=title, **layout_options)
    # Reduce opacity to see all scatters
    fig.update_traces(opacity=0.75)
    return fig

def plot_predictions(on_data: str,
                     energy_range: tuple[int,int],
                     model_version: int,
                     data_dict: dict[str,pd.DataFrame],
                     model_name: str = "MLPRegressorModel",
                     select_channels = 0,
                     verbose: bool = False,
                     PCA_fit: str|PCA|TruncatedSVD = "self",
                     plot: Literal["Histogram","EnergyScatter",False] = "Histogram",
                     custom_func: Callable = None,
                     which: Literal["T0","Tfit"] = None,  #Default Tfit
                     shift: float = 1.0,
                    **kwargs) -> go.Figure:
    """Fetches fitted model (with name ``model_name`` & version ``model_version``) and predicts ``on_data`` provided from the ``data_dict`` within ``energy_range``
    
    - If ``PCA_fit`` is not given, performs the PCA transform fit on the prediction DataFrame (``data_dict[on_data]``) itself. If given as a string,
    performs the PCA fitting on the DataFrame indexed by this string in the data_dict (``data_dict[PCA_fit]``). If given as a fitted PCA or TruncatedSVD
    instance, uses this instance directly to perform the transformation on (recommended; fastest for time-consuming fitting).
    - The ``custom_func`` argument takes the untransformed input dataframe and adds its output to the prediction series.
    - The ``shift`` argument determines if and how far the initial data is shifted. If 0, no shift is applied; if 1, shifts fully towards the mode;
    if 0.5, shifts halfway towards the mode.
    - The ``which`` argument gets Tfit or T0 from model by default, then uses which setting if not found in model metadata.
    - Extra keyword arguments ``show_dt`` & ``show_pred`` may be given to modify what traces are shown/hidden
    - Any remaining keyword arguments passable to Histogram or EnergyScatter will be passed there (e.g. ``bins``, ``mode``, ...)
    """
    x_to_t = lambda x: 160-(x*4)
    t_to_x = lambda t: (160-t)/4
    func = custom_func or (lambda on_data_df: np.zeros(on_data_df.iloc[0].transpose().shape))

    # Get regressor and PCA data in right format
    modelinfo: ModelInfo = ModelInfo.from_database(model_name, model_version)
    regressor, dt_correction_mode, pca_method, pca_components = modelinfo.model, getattr(modelinfo, "dt_correcting", False), modelinfo.pca_method, modelinfo.pca_components
    if isinstance(PCA_fit, str):
        pca_model = modelinfo.get_transformer()
    else:
        pca_model = modelinfo.check_transformer(PCA_fit)
    global PLOT_SETTINGS
    which = ({"which":modelinfo.which or PLOT_SETTINGS["which"]} | ({"which":which} if which else {}))["which"]
    dfs: dict[str,pd.DataFrame] = {}
    include_energy = kwargs.pop("include_energy",False)
    show_pred = kwargs.pop("show_pred",PLOT_SETTINGS["show_pred"])
    for string in ([on_data,PCA_fit] if isinstance(PCA_fit, str) else [on_data]):
        if "raw" in string:
            print("[GeWaPro][plotting.plot_predictions] Found 'raw' string in data name, so using the data directly (no waveform getting)...")
            dfs[string] = data_dict[string]
        elif string == "self" and string != on_data:
            dfs[string] = dfs[on_data]
        else:
            dfs[string] = get_waveforms(select_energies=energy_range if string==on_data else (),
                                        select_channels=select_channels if string==on_data else [],
                                        include_energy=include_energy,
                                        source_data=data_dict[string])

    # # Check if on_data is a subset of PCA_fit:
    # if not set(dfs[on_data].columns) <= set(dfs[PCA_fit].columns):
    #     raise ValueError("The data to perform the PCA transform on must always contain all the waveforms from the prediction data set")

    # Get labels
    get_dt_labels = False
    if kwargs.pop("show_dt",PLOT_SETTINGS["show_dt"]):
        try:
            get_dt_labels = False if any(["-" == name_to_vals(col)["dT"] for col in dfs[on_data].columns]) else True
        except:
            pass
    if dt_correction_mode and not get_dt_labels:
        raise add_notes(ValueError(f"Found no dT labels in the given data, while this is required for dT correcting model \"{model_name} v{model_version}\". Try a different model, or add the labels to the data"),"Provided data: "+pandas_string_rep(dfs[on_data]))
    try:
        labels_t = np.array([name_to_vals(col)[which] for col in dfs[on_data].columns])
        if get_dt_labels:
            labels_dt = np.array([name_to_vals(col)["dT"] for col in dfs[on_data].columns])
            s_labels_ref_dt = pd.Series(labels_dt,name=f"dT")
        else:
            labels_dt = 0*labels_t
        s_labels_t = pd.Series(labels_t, name=f"labels {which}")
        s_labels_E = pd.Series(np.array([name_to_vals(col)["E"] for col in dfs[on_data].columns]), name="Initial data")
    except Exception as e:
        e.add_note(f"Failed to create labels for columns: {list(dfs[on_data].columns)[:3]}...")
        raise e
    labels_x = t_to_x(s_labels_t.values)
    print(f"[GeWaPro][plotting.plot_predictions] Predicting data ({on_data}) for regressor v{model_version} has shape:", dfs[on_data].shape, "with",pca_components,f"{pca_method.__name__} components (random state {pca_model.random_state}) and energy range",energy_range) if verbose else None

    # (Fit and) transform the data
    time_before_pca = datetime.now()
    if isinstance(PCA_fit, str):
        data_trans = pca_model.fit(dfs[PCA_fit].T.values) if pca_model else dfs[PCA_fit].T.values
    data_trans = pca_model.transform(dfs[on_data].T.values) if pca_model else dfs[on_data].T.values
    print("[GeWaPro][plotting.plot_predictions] Finished PCA fitting and transformation, took",datetime.now() - time_before_pca) if verbose else None

    # Combining labels and creating prediction Series
    shift = -round(shift*s_labels_t.apply(lambda x: round(x)).mode().iloc[0]) if shift else 0
    s_labels_t.name = f"Initial data: {which}"+(f"{' -' if shift < 0 else ' +'} {abs(shift)} ns" if shift else "")
    # display(pd.concat([pd.Series(regressor.predict(predict_on_data_trans),name="predicted x"),pd.Series(labels_x,name="labels x")],axis=1))
    time_before_pred = datetime.now()
    predicted_s_r = pd.Series(x_to_t(predict(regressor,data_trans) + func(dfs[on_data])) + (dt_correction_mode*labels_dt),name="Tpred")
    predicted_s = predicted_s_r - labels_t
    predicted_s.name = "Tpred - Tref"
    print("[GeWaPro][plotting.plot_predictions] Finished prediction of regressor, took",datetime.now() - time_before_pred) if verbose else None
    if dt_correction_mode:
        predicted_s.name = f"dT + Tcorr - {which}" #predicted_s.name + " (dT + Tcorr - Tref)"

    # Handling the creation and plotting of df_plot
    add_df = kwargs.pop("add_df",False)
    hist_kwargs = {"bins":[-30,30,0.25],"title":f"Prediction Histogram {model_name} v{model_version} on '{on_data}' E-range {energy_range}"} | kwargs
    scatter_kwargs = {"energy_map":s_labels_E} | kwargs
    df_plot = pd.concat([s_labels_t+shift,predicted_s]+([predicted_s_r] if show_pred else [])+([s_labels_ref_dt] if get_dt_labels else []), axis=1)
    print("[GeWaPro][plotting.plot_predictions] Plot columns:",df_plot.columns) if verbose else None
    if plot is False:
        return df_plot
    elif isinstance(plot, str) and plot.lower() == "histogram":
        hist = histogram(df_plot, **hist_kwargs)
        if add_df is True:
            hist._df = df_plot
        return hist
    elif isinstance(plot, str) and plot.lower() == "energyscatter":
        esp = energy_scatter_plot(df_plot, **scatter_kwargs)
        if add_df is True:
            esp._df = df_plot
        return esp
    else:
        raise ValueError("'plot' must be either 'Histogram', 'EnergyScatter' or False")

@cache(os.path.join("data","cache"), ignore_args=["data_dict","PCA_fit","verbose"])
def _fwhm_energy_df(on_data: str,
                   start: int,
                   end: int,
                   step: int,
                   model_version: int,
                   data_dict: dict[str,pd.DataFrame],
                   model_name: str = "MLPRegressorModel",
                   select_channels = [],
                   PCA_fit: str|PCA|TruncatedSVD = "self",
                   which: Literal["T0","Tfit"] = None,
                   verbose: bool = False):
    """Creates an energy DataFrame that can be used to plot FWHMs for different energy bins. It is strongly recommended to use fitted PCA models to increase computing time"""
    ranges = _get_ranges(start, end, step)
    def rename(string: str):
        """Removes the shift and pretext from the column name"""
        if ":" in string:
            if "-" in string:
                return string[string.find(":")+2:string.find("-")-1]
            elif "+" in string:
                return string[string.find(":")+2:string.find("+")-1]
            return string[string.find(":")+2:]
        return string
    
    for e_range in ranges:
        fig_pred = plot_predictions(on_data=on_data,
                                    energy_range=e_range,
                                    model_version=model_version,
                                    data_dict=data_dict,
                                    model_name=model_name,
                                    select_channels=select_channels,
                                    PCA_fit=PCA_fit,
                                    verbose=verbose,
                                    add_df=True,
                                    which=which)
        results_df = stats(fig_pred._df)
        results_df.loc["FWHM",:] = [2*np.sqrt(2*np.log(2)) * fig_pred._params[col+" Gaussian"]["sigma"] for col in results_df.columns]
        results_df.loc["FWHM SD",:] = [2*np.sqrt(2*np.log(2)) * dict(zip((pars := fig_pred._params[col+" Gaussian"]).keys(), np.sqrt(np.diag(pars["Covariance"]))))["sigma"] for col in results_df.columns]
        results_df.loc["FWHM GoF%",:] = [fig_pred._params[col+" Gaussian"]["GoodnessOfFit"] for col in results_df.columns]
        results_df.loc["FWHM GoF",:] = results_df.loc["FWHM GoF%",:] * results_df.loc["FWHM",:]
        if e_range == ranges[0]:
            multi_index = pd.MultiIndex.from_product([[rename(col) for col in results_df.columns],[(i[1]+i[0])/2 for i in ranges]], names=["Series","Energy range"])
            df_fig = pd.DataFrame(index=multi_index,columns=results_df.index)
        df_fig.loc[(slice(None),(e_range[1]+e_range[0])/2),:] = results_df.T.rename(index={col:rename(col) for col in results_df.columns}).values
    return df_fig

def _get_ranges(start: int, end: int, step: int, /):
    if not all(ints := [isinstance(i, int) for i in [start, end , step]]):
        raise ValueError(f"Invalid value(s) for {join_strings([k for k,v in zip(["start", "end", "step"],ints) if not v], 0, "&")}; should be integer")
    elif (end - start) % step != 0:
        raise ValueError("end must be an integer amount of steps from start")
    return [(i-step/2,i+step/2) for i in [start+i*step for i in range(0,(end - start) // step + 1)]]

def _get_counts_for_bins(start: int, end: int, step: int, data_dict: dict, source_data: str, select_channels: list[int]):
    ranges = _get_ranges(start, end, step)
    data_plot = get_waveforms(source_data=data_dict[source_data], select_channels=select_channels)
    s_labels_E = pd.Series(np.array([name_to_vals(col)["E"] for col in data_plot.columns]))
    return {f"{i:.0f}-{j:.0f}":((s_labels_E >= i) & (s_labels_E < j)).sum() for i,j in ranges}