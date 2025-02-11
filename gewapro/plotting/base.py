import os
import pandas as pd
import numpy as np
import warnings as warns
from typing import Callable, Literal
from scipy.optimize import OptimizeWarning, curve_fit
import plotly.graph_objects as go
import plotly.express as px
import mlflow.pyfunc
from gewapro.functions import gaussian, gaussian_arr, inverse_quadratic, inverse_quadratic_arr
from sklearn.decomposition import PCA, TruncatedSVD
from gewapro.preprocessing import get_waveforms
from gewapro.cache import cache
from gewapro.util import stats, join_strings

def histogram(data: pd.DataFrame|pd.Series,
              bins: list[int] = None,
              warnings: Literal["raise","print","ignore"] = "print",
              add_fits: list[str] = ["Gaussian"],
              **options):
    """Returns a histogram with one (Series) or more (DataFrame) traces with fitted Gaussians, with ``options`` kwargs passed to the traces"""
    if not isinstance(warnings, str):
        raise ValueError("warnings must be a string")
    if len(data) == 0 and warnings != "raise":
        if warnings != "ignore":
            print("[WARNING] Got empty data, nothing to plot")
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
            print(f"[WARNING] No (valid) bins provided, assuming start {bins[0]:.0f}, end {bins[1]:.0f} and step {bins[2]}")
    if not add_fits:
        add_fits = []
    elif not isinstance(add_fits, list) or any([fit_name not in ["Gaussian","Inv. Quadratic"] for fit_name in add_fits]):
        raise ValueError("add_fits must be a list containing 'Gaussian' and/or 'Inv. Quadratic'")
    fig = go.Figure()
    options = {"xaxis_range":[bins[0],bins[1]]} | options
    title = options.pop("title","Histogram")
    trace_name = options.pop("trace_name",None)
    colors = options.pop("colors", px.colors.qualitative.Plotly)
    layout_options = {k.replace("layout_",""):v for k,v in options.items() if "axis" in k or k.startswith("layout_")}
    fit_options = {k.replace("fit_",""):v for k,v in options.items() if k.startswith("fit_")}
    options = {k:v for k,v in options.items() if not ("axis" in k or k.startswith("layout_") or k.startswith("fit_"))}
    trace_names = options.pop("trace_names", [trace_name]*len(data.columns))
    fhwm = 2*np.sqrt(2*np.log(2))
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
        for fit_name,fit_func in {"Gaussian":gaussian,"Inv. Quadratic":inverse_quadratic}.items():
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
                        print(f"[WARNING] {fit_name} fit optimization failed for initial guess [A,\u03BC,\u03C3] = {p0}: OptimizeWarning: {w[-1].message}")
            except Exception as e:
                if warnings == "print":
                    print(f"[WARNING] {fit_name} fitting failed: {e.__class__.__name__}: {e}")
                elif not warnings in ["ignore","print"]:
                    raise e
            params[f"{trace_names[i] or col} {fit_name}"] = dict(
                zip(fit_func.__code__.co_varnames[1:fit_func.__code__.co_argcount+fit_func.__code__.co_kwonlyargcount], popt, strict=True)
            ) | {"Covariance": pcov}
            fig.add_trace(go.Scatter(x=x_data,
                                     y=fit_func(x_data, *popt),
                                     line_color=fig.data[2*i].marker.color,
                                     name=f"{fit_name} fit",
                                     hovertemplate = "<b>(%{x},%{y})</b>"+
                                        f'<br>\u03BC: {popt[1]:.4f}<br>\u03C3: {popt[2]:.4f}'+
                                        f'<br>Ampl.: {popt[0]/(popt[2]*np.sqrt(2*np.pi)):.2f}<br><b>FWHM: {popt[2]*fhwm:.4f}</b>',
                                    **fit_options
                                    ))

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
            print("[WARNING] Got empty data, nothing to plot")
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
                     PCA_transform_on: str = "self",
                     plot: Literal["Histogram","EnergyScatter",False] = "Histogram",
                     custom_func: Callable = None,
                    **kwargs) -> go.Figure:
    """Fetches fitted model (with name ``model_name`` & version ``model_version``) and predicts ``on_data`` provided from the ``data_dict`` within ``energy_range``
    
    - If ``PCA_transform_on`` is not given, performs the PCA transform fit on the prediction DataFrame (``data_dict[on_data]``) itself.
    - The ``custom_func`` argument takes the untransformed input dataframe and adds its output to the prediction series.
    """
    x_to_t = lambda x: 160-(x*4)
    t_to_x = lambda t: (160-t)/4
    func = custom_func or (lambda on_data_df: np.zeros(on_data_df.iloc[0].transpose().shape))

    # Get regressor and data in right format
    regressor = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    pca_method = regressor.metadata.metadata.get("PCA method", "sklearn.decomposition.PCA")
    if pca_method == "sklearn.decomposition.PCA":
        pca_method = PCA
    elif pca_method == "sklearn.decomposition.TruncatedSVD":
        pca_method = TruncatedSVD
    else:
        raise ValueError(f"Unknown PCA decomposition method '{pca_method}', known methods are 'sklearn.decomposition.PCA' and 'sklearn.decomposition.TruncatedSVD'")
    pca_random_seed = regressor.metadata.metadata["PCA random seed"]
    pca_components = regressor.metadata.signature.inputs.inputs[0].shape[-1]
    dfs: dict[str,pd.DataFrame] = {}
    include_energy = kwargs.pop("include_energy",False)
    for string in [on_data,PCA_transform_on]:
        if "raw" in string:
            print("Found 'raw' string in data name, so using the data directly (no waveform getting)...")
            dfs[string] = data_dict[string]
        elif string == "self" and string != on_data:
            dfs[string] = dfs[on_data]
        else:
            dfs[string] = get_waveforms(select_energies=energy_range if string==on_data else (),
                                        select_channels=select_channels if string==on_data else [],
                                        include_energy=include_energy,
                                        source_data=data_dict[string])

    # # Check if on_data is a subset of PCA_transform_on:
    # if not set(dfs[on_data].columns) <= set(dfs[PCA_transform_on].columns):
    #     raise ValueError("The data to perform the PCA transform on must always contain all the waveforms from the prediction data set")

    # Get labels
    try:
        labels_t = np.array([float([s[s.find("Tref")+4:s.find(",dT")] for s in [col.replace(" ","")]][0]) for col in dfs[on_data].columns])
        s_labels_t = pd.Series(labels_t, name="labels Tref")
        s_labels_E = pd.Series(np.array([float([s[s.find("E")+1:] for s in [col.replace(" ","")]][0]) for col in dfs[on_data].columns]), name="Initial data")
    except Exception as e:
        e.add_note(f"Failed to create labels for columns: {list(dfs[on_data].columns)[:3]}...")
        raise e
    labels_x = t_to_x(s_labels_t.values)
    print(f"Predicting data ({on_data}) for regressor v{model_version} has shape:", dfs[on_data].shape, "with",pca_components,f"{pca_method.__name__} components and energy range",energy_range) if verbose else None

    # Transform the data
    model: PCA = pca_method(pca_components, random_state=pca_random_seed)
    print("Random state of PCA:",model.random_state) if verbose else None
    data_trans = model.fit_transform(dfs[PCA_transform_on].values.transpose())
    if PCA_transform_on != "self":
        data_trans = model.transform(dfs[on_data].values.transpose())

    # Combining labels and creating prediction Series
    shift = -round(s_labels_t.apply(lambda x: round(x)).mode().iloc[0])
    s_labels_t.name = f"Initial data: Tref {'-' if shift < 0 else '+'} {abs(shift)} ns"
    # display(pd.concat([pd.Series(regressor.predict(predict_on_data_trans),name="predicted x"),pd.Series(labels_x,name="labels x")],axis=1))
    if str(regressor.loader_module) in ["mlflow.sklearn", "mlflow.xgboost"]: #regr.predict(d_test).transpose()[0]
        predicted_s = pd.Series(labels_t - x_to_t(regressor.predict(data_trans) + func(dfs[on_data])),name="Tref - Tpred")
    else:
        predicted_s = pd.Series(labels_t - x_to_t(regressor.predict(data_trans).transpose()[0] + func(dfs[on_data])),name="Tref - Tpred")
    add_df = kwargs.pop("add_df",False)
    hist_kwargs = {"bins":[-30,30,0.25],"title":f"Prediction Histogram {model_name} v{model_version} on '{on_data}' E-range {energy_range}"} | kwargs
    scatter_kwargs = {"energy_map":s_labels_E} | kwargs
    df_plot = pd.concat([s_labels_t+shift,predicted_s], axis=1)
    print(df_plot) if verbose else None
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

@cache(os.path.join("data","cache"), ignore_args=["data_dict","verbose"])
def _fwhm_energy_df(on_data: str,
                   start: int,
                   end: int,
                   step: int,
                   model_version: int,
                   data_dict: dict[str,pd.DataFrame],
                   model_name: str = "MLPRegressorModel",
                   select_channels = 0,
                   PCA_transform_on: str = "self",
                   verbose: bool = False):
    """Creates an energy DataFrame that can be used to ..."""
    ranges = _get_ranges(start, end, step)
    rename = lambda string: string[string.find(":")+2:string.find("-")-1] if ":" in string else string
    for e_range in ranges:
        fig_pred = plot_predictions(on_data=on_data,
                                    energy_range=e_range,
                                    model_version=model_version,
                                    data_dict=data_dict,
                                    model_name=model_name,
                                    select_channels=select_channels,
                                    PCA_transform_on=PCA_transform_on,
                                    verbose=verbose,
                                    add_df=True)
        results_df = stats(fig_pred._df)
        results_df.loc["FWHM",:] = [2*np.sqrt(2*np.log(2)) * fig_pred._params[col+" Gaussian"]["sigma"] for col in results_df.columns]
        results_df.loc["FWHM sd",:] = [2*np.sqrt(2*np.log(2)) * dict(zip((pars := fig_pred._params[col+" Gaussian"]).keys(), np.sqrt(np.diag(pars["Covariance"]))))["sigma"] for col in results_df.columns]
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
    s_labels_E = pd.Series(np.array([float([s[s.find("E")+1:] for s in [col.replace(" ","")]][0]) for col in data_plot.columns]))
    return {f"{i:.0f}-{j:.0f}":((s_labels_E >= i) & (s_labels_E < j)).sum() for i,j in ranges}