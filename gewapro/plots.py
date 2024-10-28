import inspect
import pandas as pd
import numpy as np
import warnings as warns
from typing import Callable, Literal
from scipy.optimize import OptimizeWarning, curve_fit
import plotly.graph_objects as go
import plotly.express as px
from gewapro.functions import gaussian, gaussian_arr, inverse_quadratic, inverse_quadratic_arr

def _pred_trace(results_df: pd.DataFrame, drop_rows: list = [], pred_func: Callable = None, return_variance: bool= False):
    if pred_func is None:
        raise ValueError("Prediction function must be defined")
    df = results_df.drop(index=drop_rows)
    df["xT_pred"] = pred_func(df["a"], df["x0"])
    variance = ((df["xT_pred"] - df["xT"])*(df["xT_pred"] - df["xT"])).sum() / len(df)
    trace = go.Scatter(x=df["xT_pred"],
                       y=df["xT"],
                       mode='markers',
                       # customdata=results_df[["rel_x0_cutoff_var","x_cutoff","xT"]],
                       hovertemplate = '<i>%{text}</i><br>'+
                                      '<b>xT_pred</b>: %{x:.6f}<br>'+
                                      '<b>xT</b>: %{y:.6f}',
                       text = ['Waveform {}'.format(tracename) for tracename in df.index])
    if return_variance:
        return trace, variance
    return trace

def pred_fig(results_df: pd.DataFrame, drop_rows: list = [], pred_func: Callable = None):
    trace, var = _pred_trace(results_df, drop_rows, pred_func, return_variance=True)
    fig = go.Figure(data=trace)
    df = results_df.drop(index=drop_rows)
    min_,max_ = df["xT"].min(),df["xT"].max()
    fig.update_layout(title='Predicted and actual xT',
                      xaxis_title="xT_pred",
                      yaxis_title="xT",
                      shapes = [{'type': 'line', 'yref': 'y', 'xref': 'x', 'y0': min_, 'y1': max_, 'x0': min_, 'x1': max_}])
    print("Variance:", var)
    return fig

def _corr_trace(results_df: pd.DataFrame, drop_rows: list = [], c_min_max: tuple = (None, None)):
    return go.Scatter(x=results_df["a"].drop(drop_rows),
                      y=(results_df["xT"]-results_df["x0"]).drop(drop_rows),
                      mode='markers',
                      marker=dict(cmax=c_min_max[1] or results_df["x0_cutoff_var"].max(),
                                  cmin=c_min_max[0] or results_df["x0_cutoff_var"].min(),
                                  color=results_df["x0_cutoff_var"],
                                  colorbar=dict(title="Parabolic<br>fit error"),
                                  colorscale="Viridis"),
                      customdata=results_df[["x0_cutoff_var","xT","x_cutoff"]],
                      hovertemplate = '<i>%{text}</i><br>'+
                                      '<b>a</b>: %{x:.6f} <br>'+
                                      '<b>xT - x0</b>: %{y}<br>'+
                                      '<b>Par. fit error</b>: %{customdata[0]:.8f}<br>'+
                                      '<b>(xT, x_cutoff)</b>: (%{customdata[1]:.2f}, %{customdata[2]:.0f})<br>',
                                    #   '<b>Final slope (10)</b>: (%{customdata[3]:.5f})',
                      text = ['Waveform {}'.format(tracename) for tracename in results_df.index])
    
def corr_fig(results_df: pd.DataFrame, drop_rows: list = [], c_min_max: tuple = (0, 3e-5), **kwargs):
    fig = go.Figure(data=_corr_trace(results_df, drop_rows, c_min_max))
    fig.update_layout(title='Correlation of fitted parabola and gamma arrival time',
                      xaxis_title=kwargs.pop("xaxis_title","a"),
                      yaxis_title=kwargs.pop("yaxis_title","xT - x0"),
                     **kwargs)
    return fig

def _mlp_reg_trace(data: np.array,
                  labels: np.array,
                  regressor,
                  return_mse: bool= False,
                  wave_indices = None,
                  regr_wrapper: Callable = None,
                  color: np.ndarray = None,
                  c_min_max = (None,None),
                  colorbar_name: str = ""):
    """Returns a plottable MLP regression trace"""
    predictions = regressor.predict(data)
    if isinstance(regr_wrapper, Callable):
        x_data = regr_wrapper(predictions)
    else:
        x_data = predictions
    
    mse = np.sum(np.square(x_data - labels) / len(labels))
    text = ['Waveform' for i in range(len(labels))]
    if wave_indices is not None:
        text = ['Waveform {}'.format(i) for i in wave_indices]
    xy_hovertext = '<b>xT_pred</b>: %{x:.6f}<br><b>xT</b>: %{y:.6f}' if regr_wrapper is None else '<b>x</b>: %{x:.6f}<br><b>y</b>: %{y:.6f}'
    if color is not None:
        xy_hovertext += "<br><b>Color</b>: %{customdata:.6f}"
        extra_colors = dict(marker=dict(cmax=c_min_max[1] or color.max(),
                                        cmin=c_min_max[0] or color.min(),
                                        color=color,
                                        colorbar=dict(title=colorbar_name or "Color"),
                                        colorscale="Viridis"),
                            customdata=color)
    trace = go.Scatter(x=x_data,
                       y=labels,
                       mode='markers',
                       **extra_colors,
                       hovertemplate = '<i>%{text}</i><br>'+xy_hovertext,
                       text = text)
    if return_mse:
        return trace, mse
    return trace

def mlp_reg_fig(data: np.array, labels: np.array, regressor, wave_indices: np.array = None, regr_wrapper: Callable = None, **kwargs):
    color, c_min_max, cname = kwargs.pop("color",None), kwargs.pop("c_min_max",(None,None)), kwargs.pop("colorbar_name","Color")
    trace, mse = _mlp_reg_trace(data, labels, regressor, True, wave_indices, regr_wrapper, color, c_min_max, cname)
    fig = go.Figure(data=trace)
    min_,max_ = np.min(trace.x),np.max(trace.x)
    print_ = f"MSE (r^2/N): {mse:.3f} (RMSE: {np.sqrt(mse):.3f}, FWHM: {2*np.sqrt(2*np.log(2)*mse):.3f})"
    fig.update_layout(title='Predicted and actual xT ('+print_.replace(" (R",", R"),
                      xaxis_title=kwargs.pop("xaxis_title","xT_pred"),
                      yaxis_title=kwargs.pop("yaxis_title","xT"),
                      shapes = [{'type': 'line', 'yref': 'y', 'xref': 'x', 'y0': min_, 'y1': max_, 'x0': min_, 'x1': max_}],
                     **kwargs)
    print(print_)
    return fig


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

def histogram(data: pd.DataFrame|pd.Series,
              bins: list = None,
              warnings: Literal["raise","print","ignore"] = "print",
              **options):
    """Returns a histogram with one (Series) or more (DataFrame) traces with fitted Gaussians, with ``options`` kwargs passed to the traces"""
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
    if bins and len(bins) == 3:
        pass
    else:
        bins = [data.min().min(), data.max().max(), (data.max().max() - data.min().min()) // 100]
        if warnings != "ignore":
            print(f"[WARNING] No bins provided, assuming start {bins[0]}, end {bins[1]} and step {bins[2]}")
    fig = go.Figure()
    options = {"xaxis_range":[bins[0],bins[1]]} | options
    title = options.pop("title","Histogram")
    trace_name = options.pop("trace_name",None)
    colors = options.pop("colors", px.colors.qualitative.Plotly)
    layout_options = {k:v for k,v in options.items() if "axis" in k}
    for k in layout_options.keys():
        options.pop(k)
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
        for fit_name,fit_func in {"Gaussian":gaussian}.items():#,"Inv. Quadratic":inv_quadratic_function}.items():
            try:
                p0 = [y_data.max(), x_data[list(y_data).index(y_data.max())], 10]
                # y_max = y_data[list(x_data).index(bins[0]):list(x_data).index(bins[1])].max()
                # p0 = [y_max, x_data[list(y_data).index(y_max)], 10]
                with warns.catch_warnings(record=True) as w:
                    warns.simplefilter("always")
                    popt, _ = curve_fit(fit_func, x_data, y_data, p0 = p0)
                    if len(w) > 0 and issubclass(w[-1].category, OptimizeWarning) and warnings != "ignore":
                        print(f"[WARNING] {fit_name} fit optimization failed for initial guess [A,\u03BC,\u03C3] = {p0}: OptimizeWarning: {w[-1].message}")
            except Exception as e:
                if warnings == "print":
                    print(f"[WARNING] {fit_name} fitting failed: {e.__class__.__name__}: {e}")
                elif not warnings in ["ignore","print"]:
                    raise e
            params[f"{trace_names[i] or col} {fit_name}"] = dict(
                zip(fit_func.__code__.co_varnames[1:fit_func.__code__.co_argcount+fit_func.__code__.co_kwonlyargcount], popt, strict=True)
            )
            fig.add_trace(go.Scatter(x=x_data,
                                     y=fit_func(x_data, *popt),
                                     line_color=fig.data[2*i].marker.color,
                                     name=f"{fit_name} fit",
                                     hovertemplate = "<b>(%{x},%{y})</b>"+
                                        f'<br>\u03BC: {popt[1]:.4f}<br>\u03C3: {popt[2]:.4f}'+
                                        f'<br>Ampl.: {popt[0]/(popt[2]*np.sqrt(2*np.pi)):.2f}<br><b>FWHM: {popt[2]*fhwm:.4f}</b>',
                                    ))

    fig.update_layout(barmode='overlay', title=title, **layout_options)
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    fig._params = params
    return fig