import pandas as pd
import numpy as np
from typing import Callable
import plotly.graph_objects as go

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