import os
import pandas as pd
import numpy as np
from typing import Callable
import plotly.graph_objects as go
import plotly.express as px
import mlflow.pyfunc
from sklearn.decomposition import PCA, TruncatedSVD
from gewapro.preprocessing import get_waveforms
from gewapro.plotting.base import histogram, _fwhm_energy_df, _get_counts_for_bins
from gewapro.models import get_model_version_map
from gewapro.util import add_notes, stats, join_strings, correct_energy, _validate_a_b, invert_start_end_step, get_len

def plot_transform(transformed_data, labels, n_components, projection: int = 0):
    if n_components == 2:
        px.scatter(transformed_data[:, 0],
                    transformed_data[:, 1],
                    color=labels).show()
    elif n_components == 3:
        px.scatter_3d(x=transformed_data[:, 0],
                      y=transformed_data[:, 1],
                      z=transformed_data[:, 2],
                      color=labels).show()
    elif n_components == 4:
        px.scatter_3d(x=transformed_data[:, projection%4],
                      y=transformed_data[:, (1+projection)%4],
                      z=transformed_data[:, (2+projection)%4],
                      color=transformed_data[:, (3+projection)%4]).show()
    else:
        print("[WARNING] Plotting failed: Number of components must be smaller than 5 to plot")

def energy_histogram(source_data: str,
                     data_dict: dict[str,pd.DataFrame],
                     select_channels: list[int]|int = [],
                     select_energies: tuple[int, int] = (),
                     **kwargs):
    """Creates energy histogram from ``source_data`` provided from the ``data_dict`` with ``select_energies`` and ``select_channels``
    
    Actually gets all energies and zooms in on ``select_energies``.
    Use ``correct_energy`` argument to modify the energy axis (assumes keV after correction)"""
    data_plot = get_waveforms(source_data=data_dict[source_data], select_channels=select_channels)
    s_labels_E = pd.Series(np.array([float([s[s.find("E")+1:] for s in [col.replace(" ","")]][0]) for col in data_plot.columns]), name="Initial data")
    if correct_E := kwargs.pop("correct_energy", False):
        s_labels_E = correct_energy(correct_E, s_labels_E)
    title = f"Energy Histogram on '{source_data}'"
    if select_channels:
        title += f" (channel {select_channels})"
    kwargs["add_fits"] = kwargs.pop("add_fits", False)
    if select_energies:
        kwargs["xaxis_range"] = select_energies
        kwargs["bins"] = kwargs.pop("bins", [s_labels_E.min(), s_labels_E.max(), (select_energies[1] - select_energies[0]) // 100])
    xaxis_title = kwargs.pop("xaxis_title", "Energy (keV)" if correct_E else "Energy (arb. unit)")
    yaxis_title = kwargs.pop("yaxis_title","Prevalence")
    return histogram(s_labels_E, title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title, **kwargs)

# def _display_runs(exp_ids, x, y, show_original_FWHM, **kwargs):
#     all_runs = mlflow.search_runs(experiment_ids=exp_ids,search_all_experiments=True)
#     extra_vars = {k:v for k in ["color","facet_col","facet_row"] if (v:=kwargs.pop(k, ""))}
#     box_kwargs = extra_vars | ({"hover_data":h} if (h:=kwargs.pop("hover_data",[])) else {})
#     title = title or f"Boxplot of {y} vs "+" vs ".join([x]+[v for v in extra_vars.values() if v])
#     rename_cols = {}
#     for c in [x,y]+list(extra_vars.values()):
#         rename_cols[c] = [col for col in all_runs.columns if c in col]
#         if len(rename_cols[c]) > 1:
#             raise ValueError(f"Provided column \"{c}\" matches multiple columns from experiment results: \"{'\", \"'.join(rename_cols[c])}\"")
#         elif len(rename_cols[c]) < 1:
#             if close_cols:=[col for col in all_runs.columns if (c[0:3].lower() in col.lower() or c[-4:-1].lower() in col.lower())]:
#                 raise add_notes(ValueError(f"No match in experiment columns for provided column \"{c}\""), f"Did you mean {join_strings(close_cols, 6)}?")
#             else:
#                 raise add_notes(ValueError(f"No match in experiment columns for provided column \"{c}\""), f"Possible columns are {join_strings(all_runs.columns, 10,'&')}")
#     if show_original_FWHM:
#         if "FWHM" in y:
#             try:
#                 html = mlflow.artifacts.download_artifacts(run_id=all_runs["run_id"].iloc[0],artifact_path=f"PredictionHistogram{y[y.find('E'):]}.html")
#                 with open(html,'r') as f:
#                     data = f.read()
#                     idx = data.find("FWHM")
#                     fwhm_string = data[idx:idx+20]
#                     original_fwhm = float(fwhm_string[6:fwhm_string.find("\\")])
#                 fwhm_line_kwargs = {"annotation_text":f"Original FWHM: {original_fwhm}", "annotation_position":"bottom left","annotation_font_color":"grey", "line_color":"grey", "line_width":1}
#                 fwhm_line_kwargs |= {k:v for k in fwhm_line_kwargs.keys() if (v:=kwargs.pop(k, ""))}
#             except Exception as err:
#                 show_original_FWHM = False
#                 print(f"[WARNING] Failed to add original FWHM to graph: {err}")
#     rename_cols = {v[0]: k for k, v in rename_cols.items()}
#     display_runs = all_runs[list(rename_cols.keys())].rename(columns=rename_cols)
#     print("Original FWHM:",original_fwhm)
#     return display_runs

def boxplot(exp_id: list[int]|int, x: str, y:str|Callable[[pd.DataFrame],pd.Series], boxmean: bool = True, show_original_FWHM: bool = True, title: str = "", ignore_vals: dict[str,any] = {}, **kwargs) -> go.Figure:
    """
    Function that loads and displays results data in a box plot, on certain parameters/metrics

    Arguments
    ---------
    exp_ids : list[str|int]
        List of experiment ids of the experiments to fetch
    x : str
        Column name to use on the x axis
    y : str|Callable[[pd.DataFrame],pd.Series]
        Column name to use on the y axis, or custom function that takes other columns and returns a single new column
    boxmean : bool, default True
        Whether to show the mean of the boxes
    show_original_FWHM : bool, default True
        Whether to show the FWHM of the original data as a line in the plot
    title : str, optional
        Title of the plot, default: "Boxplot of <x> vs <y> vs ..."
    ignore_vals : dict[str,Any]
        Dict of column names with corresponding (list of) value(s) to ignore
    kwargs: dict
        May be color (color of boxplots as extra dimension), facet_col, facet_row, hover_name, height, etc.
    """
    # Get the correct kwargs dicts
    if not all([isinstance(i,int) for i in (exp_id if isinstance(exp_id, (list,tuple)) else [exp_id])]):
        raise ValueError("exp_ids must be an integer or list of integers")
    all_runs = mlflow.search_runs(experiment_ids=[str(id) for id in exp_id],search_all_experiments=True)
    extra_vars = {k:v for k in ["color","facet_col","facet_row"] if (v:=kwargs.pop(k, ""))}
    hover_kwargs = {} | {k:h for k,h in {k:kwargs.pop(k,[]) for k in ["hover_data","hover_name"]}.items() if h}
    box_kwargs = extra_vars | hover_kwargs
    title: str = title or f"Boxplot of {y} vs "+" vs ".join([x]+[v for v in extra_vars.values() if v])

    # Check if y value is Callable, if so then add it as a column and set name of y as y variable
    if callable(y_func := y):
        if not (pd.DataFrame in y.__annotations__.values() and y.__annotations__.get("return",None) == pd.Series and len(y.__annotations__) == 2):
            raise ValueError("A custom metric function must take a pandas DataFrame as input and return a pandas Series")
        try:
            y = kwargs.pop("custom_metric_name", y_func.__name__)
            all_runs[y] = y_func(all_runs)
        except Exception as e:
            e.add_note("Note that the custom metric function must take a pandas DataFrame as input and return a pandas Series")
            raise RuntimeError(f"Failed to create custom metric column with '{y_func.__name__}': {e}")
        title = title.replace(str(y_func),y)

    # Rename all columns with the searched versions
    rename_cols = {}
    for c in [x,y]+list(extra_vars.values()):
        rename_cols[c] = [col for col in all_runs.columns if c in col]
        if len(rename_cols[c]) > 1:
            raise ValueError(f"Provided column \"{c}\" matches multiple columns from experiment results: \"{'\", \"'.join(rename_cols[c])}\"")
        elif len(rename_cols[c]) < 1:
            if close_cols:=[col for col in all_runs.columns if (c[:3].lower() in col.lower() or c[-4:-1].lower() in col.lower())]:
                raise add_notes(ValueError(f"No match in experiment columns for provided column \"{c}\""), f"Did you mean {join_strings(close_cols, 6)}?")
            else:
                raise add_notes(ValueError(f"No match in experiment columns for provided column \"{c}\""), f"Possible columns are {join_strings(all_runs.columns, 10,'&')}")

    # Retrieve the original FWHM to show in the plot
    if show_original_FWHM:
        if "FWHM" in y:
            e = ""
            if callable(y_func):
                try:
                    original_fwhm = float(y_func.FWHM)
                except Exception as e:
                    show_original_FWHM = False
            else:
                try:
                    html = mlflow.artifacts.download_artifacts(run_id=all_runs["run_id"].iloc[0],artifact_path=f"PredictionHistogram{y[y.find('E'):]}.html")
                    with open(html,'r') as f:
                        data = f.read()
                        fwhm_string = data[(idx := data.find("FWHM")):idx+20]
                        original_fwhm = float(fwhm_string[6:fwhm_string.find("\\")])
                except Exception as e:
                    show_original_FWHM = False
            fwhm_line_kwargs = {"annotation_text":f"Original FWHM: {original_fwhm}", "annotation_position":"bottom left","annotation_font_color":"grey", "line_color":"grey", "line_width":1}
            fwhm_line_kwargs |= {k:v for k in fwhm_line_kwargs.keys() if (v:=kwargs.pop(k, ""))}
            if not show_original_FWHM:
                print(f"[WARNING] Failed to add original FWHM to graph due to {e.__qualname__}: {e}")

    # Rename the columns in the 
    rename_cols = {v[0]: k for k, v in rename_cols.items()}
    display_runs = all_runs[list(rename_cols.keys())].rename(columns=rename_cols)
    if ignore_vals and not all(boollist := [k in [x,y]+list(extra_vars.values()) for k in ignore_vals.keys()]):
        raise ValueError(f"Got unknown column(s) {join_strings([k for k,v in zip(ignore_vals.keys(),boollist) if not v], 0, '&')} in \"ignore_vals\"")
    elif ignore_vals:
        for k,v in ignore_vals.items():
            if not isinstance(v, list):
                v = [v]
            for ignore_val in v:
                if np.sum(display_runs[k].values == ignore_val) == 0:
                    raise add_notes(ValueError(f"Got non-existent value to ignore in column \"{k}\": {ignore_val}"), f"Valid ({display_runs[k].dtype}) values are {join_strings(set(display_runs[k].values), 0, '&')}" if k != y else "")
            for ignore_val in v:
                display_runs = display_runs.loc[display_runs[k].values != ignore_val]
    display_runs["model_version"] = get_model_version_map(exp_id)["model_version"]
    box = px.box(display_runs, x=x, y=y, points="all", **box_kwargs).update_traces(boxmean=boxmean).update_layout(
        title=title, **kwargs)
    if show_original_FWHM:
        box.add_hline(y=original_fwhm,**fwhm_line_kwargs)
    return box

def energy_line_plot(on_data: str,
                     start: int,
                     end: int,
                     step: int,
                     model_version: int,
                     data_dict: dict[str,pd.DataFrame],
                     model_name: str = "MLPRegressorModel",
                     select_channels = 0,
                     PCA_fit: str|PCA|TruncatedSVD = "self",
                     verbose: bool = False,
                    **options) -> go.Figure:
    """Create a plot of FWHM as a function of energy for a specific trained model"""
    hex_to_rgba = lambda hex, opacity: f"rgba{tuple([int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]+[opacity])}".replace(" ","")
    colors = options.pop("colors", px.colors.qualitative.Plotly)
    opacity = options.pop("opacity", 0.2)
    y_col = options.pop("y_col", "FWHM")
    y_sd: str = options.pop("y_sd", "FWHM GoF")
    y_sd_name = options.pop("y_sd_name", y_sd.replace(y_col,"").lstrip())
    y_pos = options.pop("y_pos", y_sd)
    y_neg = options.pop("y_neg", y_sd)
    title = options.pop("title",f"{y_col} vs Energy on '{on_data}'")
    hist_limit = options.pop("hist_limit",250)
    options["xaxis_title"] = options.pop("xaxis_title", "Energy [keV]" if options.get("correct_energy",False) else "Energy (arb. unit)")
    options["yaxis_title"] = options.pop("yaxis_title","FWHM [ns]")
    # Change the start, end ,step if energy is to be corrected
    if correct_E := options.pop("correct_energy", False):
        (a,b),len_ = _validate_a_b(correct_E),get_len(start,end,step)
        start,end,step = invert_start_end_step(start, end, step, a, b)
        print(f"[WARNING] Got new start, end & step to correct energy axis: {start}, {end}, {step} ({len_} periods, {start} + {len_}*{step} = {len_*step+start})")
    # Raise error (with specific note) if bins do not have enough values
    if (invalid_bins := np.array([[k,v] for k,v in _get_counts_for_bins(start, end, step, data_dict, on_data, select_channels).items() if v<hist_limit])).size > 0:
        err = ValueError(F"Got bins with too little observations (less than hist_limit of {hist_limit}): {join_strings([f"{k} ({v})" for k,v in invalid_bins],7,"&","")}")
        old_bins = correct_energy(correct_E, np.array([[int(k[:k.find("-")]),int(k[k.find("-")+1:])] for k in invalid_bins[:,0]])).round().astype(int) if correct_E else []
        raise add_notes(err, f"In bin units passed to the function: {join_strings([f"{k[0]}-{k[1]} ({v})" for k,v in zip(old_bins,invalid_bins[:,1])],7,"&","")}" if correct_E else "")
    df_fig = _fwhm_energy_df(on_data, start, end, step, model_version, data_dict, model_name, select_channels, PCA_fit, verbose)
    if verbose:
        display(df_fig)
    # Reindex if correct energy is applied
    if correct_E:
        df_fig.index = pd.MultiIndex.from_tuples([(i[0], a*i[1]+b) for i in df_fig.index],names=df_fig.index.names)
    fig = go.Figure(layout={"title":title}|options)
    for i,ix in enumerate(dict.fromkeys(df_fig.index.get_level_values('Series'))): # Enumerate over the ordered columns
        fig.add_traces([go.Scatter(x = df_fig.loc[ix, :].index, y = df_fig.loc[ix, y_col] + df_fig.loc[ix, y_pos],
                                mode = 'lines', line_color = 'rgba(0,0,0,0)', showlegend = False), # Top range
                        go.Scatter(x = df_fig.loc[ix, :].index, y = df_fig.loc[ix, y_col] - df_fig.loc[ix, y_neg],
                                mode = 'lines', line_color = 'rgba(0,0,0,0)', name = y_sd_name,
                                fill='tonexty', fillcolor = hex_to_rgba(colors[i], opacity)), # Bottom range
                        go.Scatter(x = df_fig.loc[ix, :].index, y = df_fig.loc[ix, y_col],
                                mode = 'lines', name = ix, line_color = colors[i])]) # Line
    if options.pop("add_df",False) is True:
        fig._df = df_fig
        return fig
    return fig

# Add function attributes
energy_line_plot.cache_info = _fwhm_energy_df.cache_info #cache_info(function_to_inspect=_fwhm_energy_df, cache_dir=cache_dir, ignore_args=ignore_args, pre_cache=pre_cache, post_cache=post_cache)
"""Returns info on the cached function's cache settings & cache size"""
energy_line_plot.clear_cache = _fwhm_energy_df.clear_cache #(function_to_clear=_fwhm_energy_df, cache_dir=cache_dir)
"""Clear the cache of this function"""

def rmse_energy_line_plot(on_data: str,
                          start: int,
                          end: int,
                          step: int,
                          model_version: int,
                          data_dict: dict[str,pd.DataFrame],
                          model_name: str = "MLPRegressorModel",
                          select_channels = 0,
                          PCA_transform_on: str = "self",
                          verbose: bool = False,
                         **options) -> go.Figure:
    """Create a plot of RMSE as a function of energy for a specific trained model"""
    options["y_col"] = "RMSE"
    options["y_pos"] = "RMSE+"
    options["y_neg"] = "RMSE-"
    options["yaxis_title"] = options.get("yaxis_title","RMSE")
    return energy_line_plot(on_data, start, end, step, model_version, data_dict, model_name, select_channels, PCA_transform_on, verbose, **options)
