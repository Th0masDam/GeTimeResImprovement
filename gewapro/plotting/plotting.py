import os
import pandas as pd
import numpy as np
from typing import Callable, Literal, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import mlflow.pyfunc
from sklearn.decomposition import PCA, TruncatedSVD
from gewapro.preprocessing import get_waveforms
from gewapro.plotting.base import histogram, _fwhm_energy_df, _get_counts_for_bins, settings
from gewapro.models import get_model_version_map
from gewapro.util import add_notes, stats, join_strings, correct_energy, _validate_a_b, invert_start_end_step, get_len, sort, isort
from gewapro.functions import combine_and, rmse
from gewapro.cache import cache
from itertools import product

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
        print("[WARNING][GeWaPro][plotting.plot_transform] Plotting failed: Number of components must be smaller than 5 to plot")

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
    title = kwargs.pop("title", f"Energy Histogram on '{source_data}'")
    if select_channels:
        title += f" (channel {select_channels})"
    kwargs["add_fits"] = kwargs.pop("add_fits", False)
    if select_energies:
        kwargs["xaxis_range"] = select_energies
        kwargs["bins"] = kwargs.pop("bins", [s_labels_E.min(), s_labels_E.max(), (select_energies[1] - select_energies[0]) // 100])
    xaxis_title = kwargs.pop("xaxis_title", "Energy [keV]" if correct_E else "Energy [arb. unit]")
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
#                 print(f"[WARNING][GeWaPro][plotting.display_runs] Failed to add original FWHM to graph: {err}")
#     rename_cols = {v[0]: k for k, v in rename_cols.items()}
#     display_runs = all_runs[list(rename_cols.keys())].rename(columns=rename_cols)
#     print("[GeWaPro][plotting.display_runs] Original FWHM:",original_fwhm)
#     return display_runs

def box_plot(exp_id: list[int]|int,
             x: str,
             y:str|Callable[[pd.DataFrame],pd.Series],
             units: tuple[str,str] = ("",""),
             boxmean: bool = True,
             show_original_FWHM: bool|float = True,
             title: str = "",
             ignore_vals: dict[str,Any]|dict[tuple[str,...],tuple[list[Any]|Any]] = {},
             load_cols: list[str]|str = [],
             sort_by: list[str]|str = [],
            **kwargs) -> go.Figure:
    """
    Function that loads and displays results data in a box plot, on certain parameters/metrics

    Arguments
    ---------
    exp_ids : list[str|int]
        List of experiment ids of the experiments to fetch
    x : str
        Column name to use on the x axis
    y : str|Callable[[pd.DataFrame], pd.Series]
        Column name to use on the y axis, or custom function that takes other columns and returns a single new column
    units : tuple
        The units of the x and y-axis, added if given.
    boxmean : bool, default True
        Whether to show the mean of the boxes
    show_original_FWHM : bool|float, default True
        Whether to show (when given as bool) the FWHM (value set by float) of the original data as a line in the plot
    title : str, optional
        Title of the plot, default: "Boxplot of \\<x\\> vs \\<y\\> vs ..."
    ignore_vals : dict[str, Any] | dict[tuple[str, ...], tuple[list[Any] | Any]]
        Dict of column names with corresponding (list of) value(s) to ignore. Will be strict AND rules when column
        names & value names are given as a set of tuples, e.g. `{("col1","col2"): ([0,1],[2])}` only excludes data
        where (`col1` is `0` or `1`) AND (`col2` is `2`), while `{"col1": [0, 1], "col2": [2]}` excludes all data
        where either `col1` is `0` or `col1` is `1` or `col2` is `2`.
    load_cols : list[str]|str
        Column names that are not to be plotted but should be loaded (e.g. for sorting with sort_by or ignoring values with ignore_vals)
    sort_by: list[str]|str
        Sort the axes by the column name(s) passed to this argument, at most two, e.g. ["PCA components","Hidden layers"]
    kwargs: dict
        May be color (color of boxplots as extra dimension), facet_col, facet_row, hover_name, height, etc.
    """
    # Get the correct kwargs dicts
    if (exp_id := exp_id if isinstance(exp_id, (list,tuple)) else [exp_id]) and not all([isinstance(i,int) for i in exp_id]):
        raise ValueError("exp_id must be an integer or list of integers")
    all_runs = mlflow.search_runs(experiment_ids=[str(id) for id in exp_id],search_all_experiments=True)
    extra_vars = {k:v for k in ["color","facet_col","facet_row"] if (v:=kwargs.pop(k, ""))}
    load_cols = [load_cols] if isinstance(load_cols, str) else load_cols
    if not isinstance(units, tuple) or not (all(isinstance(u, str) for u in units) and len(units) == 2):
        raise ValueError("units must be a two-long tuple of unit strings")
    if not isinstance(load_cols, list):
        raise ValueError("load_cols must be a string (column name) or list of strings (list of column names)")
    if sort_by and not (isinstance(sort_by, str) or (isinstance(sort_by, (list,tuple)) and len(sort_by) <= 2 and all(isinstance(sort_v,str) for sort_v in sort_by))):
        raise ValueError("The sort_by argument must be a single column string or a list of at most two column strings")
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
    rename_cols,add_unit = {},{k:f"{k} [{u}]" for k,u in zip([x,y],units) if u}
    for c in [x,y,"run_id"]+list(extra_vars.values())+load_cols:
        rename_cols[add_unit.get(c, c)] = (matched_cols := [col for col in all_runs.columns if c in col])
        if len(matched_cols) > 1:
            raise ValueError(f"Provided column \"{c}\" matches multiple columns from experiment results: \"{'\", \"'.join(rename_cols[c])}\"")
        elif len(matched_cols) < 1:
            if close_cols:=[col for col in all_runs.columns if (c[:3].lower() in col.lower() or c[-4:-1].lower() in col.lower())]:
                raise add_notes(ValueError(f"No match in experiment columns for provided column \"{c}\""), f"Did you mean {join_strings(close_cols, 6)}?")
            else:
                raise add_notes(ValueError(f"No match in experiment columns for provided column \"{c}\""), f"Possible columns are {join_strings(all_runs.columns, 10,'&')}")

    # Retrieve the original FWHM to show in the plot
    if show_original_FWHM:
        if isinstance(show_original_FWHM, float):
            original_fwhm = show_original_FWHM
        elif "FWHM" in y:
            excep_to_raise,original_fwhm = NotImplementedError("Unexpected error occured, this should never be raised..."), 0
            if callable(y_func):
                try:
                    original_fwhm = float(y_func.FWHM)
                except Exception as e:
                    excep_to_raise = e
                    show_original_FWHM = False
            else:
                try:
                    if "E" in y:
                        html = mlflow.artifacts.download_artifacts(run_id=all_runs["run_id"].iloc[0],artifact_path=f"PredictionHistogram{y[y.find('E'):]}.html")
                    else:
                        html = mlflow.artifacts.download_artifacts(run_id=all_runs["run_id"].iloc[0],artifact_path=f"PredictionHistogram.html")
                    with open(html,'r') as f:
                        data = f.read()
                        fwhm_string = data[(idx := data.find("FWHM")):idx+20]
                        original_fwhm = float(fwhm_string[6:fwhm_string.find("\\")])
                except Exception as e:
                    excep_to_raise = e
                    show_original_FWHM = False
            if not show_original_FWHM:
                print(f"[WARNING][GeWaPro][plotting.box_plot] Failed to add original FWHM to graph due to {excep_to_raise.__class__.__qualname__}: {excep_to_raise}")
        fwhm_line_kwargs = {"annotation_text":f"Original FWHM: {original_fwhm}" if original_fwhm else "", "annotation_position":"bottom left","annotation_font_color":"grey", "line_color":"grey", "line_width":1}
        fwhm_line_kwargs |= {k:v for k in fwhm_line_kwargs.keys() if (v:=kwargs.pop(k, ""))}

    # Rename the columns in the plotting dataframe
    rename_cols = {v[0]: k for k, v in rename_cols.items()} # e.g. {"Independent PCA components":"PCA comp",...}
    display_runs = all_runs[list(rename_cols.keys())].rename(columns=rename_cols)

    # Process ignored values
    if ignore_vals and all(boollist := [k in rename_cols.values() for k in ignore_vals.keys()]):
        pass
    elif all(isinstance(k,tuple) for k in ignore_vals) and all(boollist2 := [(t in rename_cols.values()) for k in ignore_vals.keys() for t in k]):
        pass
    else:
        if all(isinstance(k,tuple) for k in ignore_vals):
            raise ValueError(f"Got unknown column(s) {join_strings([k for k,v in zip([t for k in ignore_vals.keys() for t in k],boollist2) if not v], 0, '&')} in \"ignore_vals\"")
        raise ValueError(f"Got unknown column(s) {join_strings([k for k,v in zip(ignore_vals.keys(),boollist) if not v], 0, '&')} in \"ignore_vals\"")
    for k,v in ignore_vals.items():
        if isinstance(k,tuple) and not isinstance(v, (list,tuple)):
            raise add_notes(ValueError(f"Got invalid value to ignore in columns \"{join_strings(k, 0, '&')}\": {v} is not a list or tuple"),"Columns tuple and ignore values list/tuple should have the same amount of items; each column must correspond to a value to ignore")
        elif not isinstance(v, (list,tuple)):
            v = [v]
        if isinstance(k,str):
            for ignore_val in v:
                if np.sum(display_runs[k].values == ignore_val) == 0:
                    raise add_notes(ValueError(f"Got non-existent value to ignore in column \"{k}\": {ignore_val}"), f"Valid ({display_runs[k].dtype}) values are {join_strings(set(display_runs[k].values), 0, '&')}" if k != y else "")
            for ignore_val in v:
                display_runs = display_runs.loc[display_runs[k].values != ignore_val]
        elif isinstance(k,tuple) and all(isinstance(s,str) for s in k):
            if len(k) != len(v):
                raise add_notes(ValueError(f"Got invalid value to ignore in columns \"{join_strings(k, 0, '&')}\": {v}"),"Columns tuple and ignore values list should have the same amount of items; each column must correspond to a value to ignore")
            product_ignorables = list(product(*[v_ if isinstance(v_,(list,tuple)) else [v_] for v_ in v]))
            for ignore_tup in product_ignorables:
                for i,ignore_val in enumerate(ignore_tup):
                    if np.sum(display_runs[k[i]].values == ignore_val) == 0:
                        raise add_notes(ValueError(f"Got non-existent value to ignore in column \"{k[i]}\": {ignore_val}"), f"Valid ({display_runs[k[i]].dtype}) values are {join_strings(set(display_runs[k[i]].values), 0, '&')}" if k != y else "")
                display_runs = display_runs.loc[~combine_and(*[(display_runs[k[i]].values == v) for i,v in enumerate(ignore_tup)])]

    # Add model version map
    display_runs = display_runs.set_index("run_id")
    display_runs["model_version"] = get_model_version_map(exp_id)["model_version"]
    display_runs["model_version"] = display_runs["model_version"].fillna("UNKNOWN MODEL & VERSION")

    # Sort the columns if wanted
    if (sort_by:=[sort_by] if isinstance(sort_by, str) else sort_by):
        if any(inv_sort := [sort_v not in display_runs.columns for sort_v in sort_by]):
            raise add_notes(ValueError(f"Did not find sort_by string(s) in the column names: {join_strings([k for k,v in zip(sort_by,inv_sort) if v], 0, '&')}"), f"Valid column names are {join_strings(display_runs.columns, 0, '&')}")
        sorted_index = isort(display_runs[sort_by[0]].to_list())  # single sort
        if len(sort_by) == 2:  # double sort
            sorted_index = [lsi for v in sort(display_runs.groupby([sort_by[0]]).groups).values() for lsi in list(v[isort(display_runs.loc[v,sort_by[1]].to_list())])]
        display_runs = display_runs.reindex(labels=sorted_index)

    # Create box plots, add FWHM line if wanted and return it
    box = px.box(display_runs, x=add_unit.get(x, x), y=add_unit.get(y, y), points="all", **box_kwargs).update_traces(boxmean=boxmean).update_layout(
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
                     select_channels = [],
                     PCA_fit: str|PCA|TruncatedSVD = "self",
                     which: Literal["T0","Tfit"] = None,
                     verbose: bool = False,
                    **options) -> go.Figure:
    """Create a plot of FWHM as a function of energy for a specific trained model. It is strongly recommended to use fitted PCA models to increase computing time"""
    hex_to_rgba = lambda hex, opacity: f"rgba{tuple([int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]+[opacity])}".replace(" ","")
    colors = options.pop("colors", px.colors.qualitative.Plotly)
    line_shape = options.pop("line_shape","linear")
    opacity = options.pop("opacity", 0.2)
    y_col = options.pop("y_col", "FWHM")
    y_sd: str = options.pop("y_sd", "FWHM GoF")
    if y_sd:
        y_sd_name = options.pop("y_sd_name", y_sd.replace(y_col,"").lstrip())
        y_pos = options.pop("y_pos", y_sd)
        y_neg = options.pop("y_neg", y_sd)
    title = options.pop("title",f"{y_col} vs Energy of {model_name} v{model_version} on '{on_data}' Ch{select_channels}")
    hist_limit = options.pop("hist_limit",250)
    options["xaxis_title"] = options.pop("xaxis_title", "Energy [keV]" if options.get("correct_energy",False) else "Energy (arb. unit)")
    options["yaxis_title"] = options.pop("yaxis_title","FWHM [ns]")
    # Change the start, end ,step if energy is to be corrected
    if correct_E := options.pop("correct_energy", False):
        (a,b),len_ = _validate_a_b(correct_E),get_len(start,end,step)
        start,end,step = invert_start_end_step(start, end, step, a, b)
        print(f"[WARNING][GeWaPro][plotting.energy_line_plot] Got new start, end & step to correct energy axis: {start}, {end}, {step} ({len_} periods, {start} + {len_}*{step} = {len_*step+start})")
    # Raise error (with specific note) if bins do not have enough values
    if (invalid_bins := np.array([[k,v] for k,v in _get_counts_for_bins(start, end, step, data_dict, on_data, select_channels).items() if v<hist_limit])).size > 0:
        err = ValueError(F"Got bins with too little observations (less than hist_limit of {hist_limit}): {join_strings([f"{k} ({v})" for k,v in invalid_bins],7,"&","")}")
        old_bins = correct_energy(correct_E, np.array([[int(k[:k.find("-")]),int(k[k.find("-")+1:])] for k in invalid_bins[:,0]])).round().astype(int) if correct_E else []
        raise add_notes(err, f"In bin units passed to the function: {join_strings([f"{k[0]}-{k[1]} ({v})" for k,v in zip(old_bins,invalid_bins[:,1])],7,"&","")}" if correct_E else "")
    df_fig = _fwhm_energy_df(on_data, start, end, step, model_version, data_dict, model_name, select_channels, PCA_fit, which, verbose)
    if verbose:
        display(df_fig)
    # Reindex if correct energy is applied
    if correct_E:
        df_fig.index = pd.MultiIndex.from_tuples([(i[0], a*i[1]+b) for i in df_fig.index],names=df_fig.index.names)
    fig = go.Figure(layout={"title":title}|options)
    for i,ix in enumerate(dict.fromkeys(df_fig.index.get_level_values('Series'))): # Enumerate over the ordered columns
        shaded_traces = []
        if y_sd:
            shaded_traces = [go.Scatter(x = df_fig.loc[ix, :].index, y = df_fig.loc[ix, y_col] + df_fig.loc[ix, y_pos],
                                        mode = 'lines', line_shape=line_shape, line_color = 'rgba(0,0,0,0)', showlegend = False),  # Top range
                             go.Scatter(x = df_fig.loc[ix, :].index, y = df_fig.loc[ix, y_col] - df_fig.loc[ix, y_neg],
                                        mode = 'lines', line_shape=line_shape, line_color = 'rgba(0,0,0,0)', name = y_sd_name,
                                        fill='tonexty', fillcolor = hex_to_rgba(colors[i], opacity))]       # Bottom range
        fig.add_traces(shaded_traces+[go.Scatter(x = df_fig.loc[ix, :].index, y = df_fig.loc[ix, y_col],         # Line
                                                 mode = 'lines', line_shape=line_shape, name = ix, line_color = colors[i])])
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


def add_energy_histogram(fig_eline: go.Figure, fig_ehist: go.Figure, **layout_kwargs):
    """Adds energy histogram to an energy line plot"""
    yaxis2_title = layout_kwargs.pop("yaxis2_title", "FWHM [ns]")
    for trace in fig_eline.data:
        y2_max = 30
        if trace.name == "dT - Tref":
            y2_max = round(np.max(trace.y)+0.6)
    fig_comb = make_subplots(specs=[[{"secondary_y": True}]]).update_layout(fig_ehist.layout).update_yaxes(title_text=yaxis2_title, range=[0,y2_max], secondary_y=True)
    fig_eline.update_traces(yaxis="y2")
    fig_comb.add_traces(fig_eline.data+fig_ehist.data)
    return fig_comb.update_layout(**layout_kwargs) if layout_kwargs else fig_comb

def combine_line_plots(initial_plot: go.Figure, extra_figs: list[go.Figure], trace_names: str|list[str] = "Model#", **options) -> go.Figure:
    """Combines multiple energy line plot predictions into one plot with default trace names ``'[Model0] Tpred - Tfit'``, ``'[Model1] Tpred - Tfit'``, ...
    
    NOTE: for this to work, all plots must have the same data supplied (same set of channels),
    otherwise no meaningful comparison can be made. Tolerance for this can be made more lenient
    using the ``equality_tolerance`` options kwarg (default 0.000001)
    """
    initial_tfit: go.Scatter = [*initial_plot.select_traces({"name":"Tfit"})][0]
    tfit_x,tfit_y = np.array(initial_tfit.x, dtype=float),np.array(initial_tfit.y, dtype=float)
    initial_tpred: go.Scatter = [*initial_plot.select_traces({"name":"Tpred - Tref"})][0]
    if isinstance(trace_names, str):
        trace_names = [trace_names.replace("#",f"{i}") for i in range(len(extra_figs)+1)]
    elif isinstance(trace_names, list|tuple):
        trace_names = [str(figname) for figname in trace_names]+([f"Model{i}" for i in range(len(trace_names),len(extra_figs)+1)] if len(trace_names) <= len(extra_figs) else [])
    initial_tpred.name = f"[{trace_names[0]}] Tpred - Tref"
    traces_to_add = []
    colors = options.pop("colors", px.colors.qualitative.Plotly)
    equality_tolerance = options.pop("equality_tolerance", 1e-6)
    if equality_tolerance > 0.1:
        raise ValueError(f"equality_tolerance may not be more than 0.1, got {equality_tolerance}")
    for i,fig in enumerate(extra_figs):
        try:
            traces: dict[str,go.Scatter] = {getattr(trace, "name", ""):trace for trace in [*fig.select_traces()] if getattr(trace, "name", "") in ["Tfit", "Tpred - Tref"]}
            trace_x,trace_y = np.array(traces["Tfit"].x, dtype=float),np.array(traces["Tfit"].y, dtype=float)
        except Exception as e:
            raise RuntimeError(f"Failed to create trace '{trace_names[i+1]}' from Figure data in extra_figs[{i}]...") from e
        if "Tpred - Tref" not in traces or "Tfit" not in traces:
            raise ValueError("Figs should all contain a 'Tfit' and a 'Tpred - Tref' column")
        try:
            if not (equal_x := all(np.isclose(trace_x,tfit_x,atol=equality_tolerance))) or not (equal_y := all(np.isclose(trace_y,tfit_y,atol=equality_tolerance))):
                invalid_str = "' & '".join([k for k,v in zip(['x','y'],[equal_x,equal_y]) if not v])
                print(f"Exception raised '{invalid_str}' not equal:")
                if not equal_x:
                    print(f"{traces["Tfit"].x} =/= \n{initial_tfit.x}")
                    print(np.isclose(traces["Tfit"].x,initial_tfit.x))
                if not equal_y:
                    print(f"{traces["Tfit"].y} =/= \n{initial_tfit.y}")
                    print(np.isclose(traces["Tfit"].y,initial_tfit.y))
                raise ValueError("Tfit data were not equal")
        except ValueError as e:
            note = str(e)
            if "Tfit data were not equal" in str(e):
                note = f"Leniency for np.isclose is currently {equality_tolerance}, try stretching it with the 'equality_tolerance' options kwarg"
            raise add_notes(ValueError("Combining plots only works for figures that all have exactly the same initial Tfit data (e.g. equal channels)"),note) from e
        except Exception as e:
            invalid_str = "' & '".join([k for k,v in zip(['x','y'],[equal_x,equal_y]) if not v])
            print(f"Exception raised '{invalid_str}' not equal:")
            if not equal_x:
                print(f"{traces["Tfit"].x} =/= \n{initial_tfit.x}")
                print(np.isclose(traces["Tfit"].x,initial_tfit.x))
            if not equal_y:
                print(f"{traces["Tfit"].y} =/= \n{initial_tfit.y}")
                print(np.isclose(traces["Tfit"].y,initial_tfit.y))
            raise e
        traces["Tpred - Tref"].name = f"[{trace_names[i+1]}] Tpred - Tref"
        traces["Tpred - Tref"].line["color"] = colors[i+2]
        traces_to_add.append(traces["Tpred - Tref"])
    initial_plot.add_traces(traces_to_add)
    return initial_plot


def combined_channel_line_plot(on_data: dict[int,str],
                               model_versions: list[int],
                               model_names: dict[int,str],
                               combine_channels: list[int],
                               fitted_pcas: dict[int,str],
                               data: dict,
                               energy_corrections: dict[int,tuple[int,int]],
                               trace_names: list[str]|str = "Model#",
                               start_end_step: tuple[int,int,int] = (100, 1300, 50),
                               hist_limit: int = 700,
                               y_sd: str = None,
                               add_energy_hist: bool = True,
                               to_table: bool = False,
                               verbose: bool = False,
                              **options):
    """Returns a line plot averaged over all given channels"""
    hex_to_rgba = lambda hex, opacity: f"rgba{tuple([int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]+[opacity])}".replace(" ","")
    colors = options.pop("colors", px.colors.qualitative.Plotly)
    line_shape = options.pop("line_shape","linear")
    opacity = options.pop("opacity", 0.2)
    equality_tolerance = options.pop("equality_tolerance", 1e-6)
    hist_name = options.pop("hist_name","Counts")
    layout_options = {k.replace("layout_",""):v for k,v in options.items() if "axis" in k or k.startswith("layout_")}
    layout_options["xaxis_title"] = layout_options.pop("xaxis_title","Energy [keV]")
    layout_options["yaxis_title"] = layout_options.pop("yaxis_title","Prevalence")
    layout_options["title"] = options.pop("title",f"Predictions of NNs on channels {combine_channels} (trained on channel TCh)")

    def extract_dfs(df: pd.DataFrame, histogram_column_name: str = "Energy_hist") -> tuple[pd.Series,pd.DataFrame]:
        is_energy_hist = np.array([df.index[:i+1].is_monotonic_increasing for i,_ in enumerate(df.index)])
        return df[is_energy_hist][histogram_column_name],df[~is_energy_hist][[col for col in df.columns if col != histogram_column_name]]
    series_hist,df_fig = extract_dfs(_get_combined_channel_and_histogram_df(on_data=on_data,
                                                                            model_versions=model_versions,
                                                                            model_names=model_names,
                                                                            trace_names=trace_names,
                                                                            combine_channels=combine_channels,
                                                                            fitted_pcas=fitted_pcas,
                                                                            data=data,
                                                                            energy_corrections=energy_corrections,
                                                                            start_end_step=start_end_step,
                                                                            hist_limit=hist_limit,
                                                                            y_sd=y_sd,
                                                                            verbose=verbose,
                                                                            equality_tolerance=equality_tolerance))
    if verbose:
        display(df_fig)
        display(series_hist)
    if to_table:
        return df_fig[[col for col in df_fig.columns if not (col.endswith("-") or col.endswith("+"))]]
    
    fig = go.Figure()
    base_cols = [col for col in df_fig.columns if not (col.endswith(" SD") or col.endswith(" SD+") or col.endswith(" SD-"))]
    for i,colbase in enumerate(base_cols): # Enumerate over the ordered base column keys
        shaded_traces = [go.Scatter(x = df_fig.index, y = df_fig[colbase] + df_fig[colbase+" SD+"],
                                    mode = 'lines', line_shape=line_shape, line_color = 'rgba(0,0,0,0)', showlegend = False),  # Top range
                        go.Scatter(x = df_fig.index, y = df_fig[colbase] - df_fig[colbase+" SD-"],
                                    mode = 'lines', line_shape=line_shape, line_color = 'rgba(0,0,0,0)', name = "Standard deviation",
                                    fill='tonexty', fillcolor = hex_to_rgba(colors[i], opacity))]       # Bottom range
        fig.add_traces(shaded_traces+[go.Scatter(x = df_fig.index, y = df_fig[colbase],         # Line
                                                 mode = 'lines', line_shape=line_shape, name = colbase, line_color = colors[i])])

    if options.pop("add_df",False) is True:
        fig._dfs = [df_fig,series_hist]
    if add_energy_hist:
        bin_width = series_hist.index[1] - series_hist.index[0]
        options_2 = {}
        options_2["yaxis2_title"]=options.pop("yaxis2_title","FWHM [ns]")
        fig_ehist = go.Figure(go.Scatter(x=series_hist.index,
                                         y=series_hist.values,
                                         marker={"line":{"width":0}},
                                         mode="lines",
                                         line_shape="hvh",
                                         name=hist_name,
                                         line_color= 'rgba(0,0,0,0.5)',
                                         marker_color= 'rgba(0,0,0,0.5)',
                                         customdata=pd.DataFrame(data=[series_hist-bin_width/2,series_hist+bin_width/2]).T,
                                         hovertemplate ='<b>[%{customdata[0]:.2f},%{customdata[1]:.2f}): %{y:.0f}x</b>',
                                         **options,
                                        ))
        fig = add_energy_histogram(fig, fig_ehist, **options_2) # includes yaxis2_title
        layout_options["xaxis_range"] = layout_options.pop("xaxis_range",[0,df_fig.index.max()])
        fig.update_layout(**layout_options)
        return fig
    return fig.update_layout(**layout_options)


@cache(os.path.join("data","cache"), ignore_args=["data","fitted_pcas","verbose","equality_tolerance"])
def _get_combined_channel_and_histogram_df(on_data: dict[int,str],
                                           model_versions: list[int],
                                           model_names: dict[int,str],
                                           trace_names: list[str],
                                           combine_channels: list[int],
                                           fitted_pcas: dict[int,PCA],
                                           data: dict,
                                           energy_corrections: dict[int,tuple[int,int]],
                                           start_end_step: tuple[int,int,int] = (100, 1300, 50),
                                           hist_limit: int = 700,
                                           y_sd: str = None,
                                           verbose: bool = False,
                                           equality_tolerance: float = 1e-6,
                                          ):
    channel_dict = {i:None for i in combine_channels}
    for ch in combine_channels:
        fig_elines = []
        for model_v in model_versions:
            fig_eline = energy_line_plot(on_data[ch],
                                         start_end_step[0],
                                         start_end_step[1],
                                         start_end_step[2],
                                         model_v,
                                         data,
                                         model_names[model_v],
                                         PCA_fit=fitted_pcas[model_v],
                                         correct_energy=energy_corrections[ch],
                                         hist_limit=hist_limit,
                                         verbose=verbose,
                                         y_sd=y_sd)
            fig_elines.append(fig_eline)
        combined_eline = combine_line_plots(fig_elines[0], fig_elines[1:], trace_names=trace_names, equality_tolerance=equality_tolerance)
        fig_ehist = energy_histogram(on_data[ch], data, select_energies=(0,start_end_step[1]), bins=[0,1400,2], correct_energy=energy_corrections[ch])
        channel_dict[ch] = combined_eline,fig_ehist

    # Check if index is same all around
    x_arrays = [np.round(trace.x, -1) for i in combine_channels for trace in channel_dict[i][0].data]
    if not all((x_arrays[0] == x_arrays[i]).all() for i in combine_channels[1:]):
        raise ValueError("All x-indices must be equal for all traces up to 10 ns, this was not the case")

    # Create list of data for each trace name
    traces_y = {trace.name:[trace.y] for trace in channel_dict[combine_channels[0]][0].data}
    for i in combine_channels[1:]:
        for k in traces_y.keys():
            traces_y[k] += [trace.y for trace in channel_dict[i][0].data if trace.name == k]
    traces_y = {k:np.array(v) for k,v in traces_y.items()}
    print(traces_y) if verbose else None

    # Create the df
    dict_fig = {}
    for k in traces_y:
        dict_fig |= {k:[x for x in traces_y[k].mean(axis=0)],
                     k+" SD": [rmse(traces_y[k][:,i]) for i in range(len(x_arrays[0]))],
                     k+" SD+":[rmse(traces_y[k][:,i],"+") for i in range(len(x_arrays[0]))],
                     k+" SD-":[rmse(traces_y[k][:,i],"-") for i in range(len(x_arrays[0]))]}
    df_fig = pd.DataFrame(data = dict_fig, index = x_arrays[0])
    hist_data = [pd.DataFrame(data={f"{i}":channel_dict[i][1].data[0].y},index=channel_dict[i][1].data[0].x) for i in combine_channels]
    df_hist = pd.concat(hist_data, axis=1).fillna(0)
    series_hist = df_hist.reindex(sorted(df_hist.index)).sum(axis=1)
    df_final = pd.concat([pd.DataFrame({"Energy_hist":series_hist}),df_fig])
    return df_final