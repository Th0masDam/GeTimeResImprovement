import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import numba as nb
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from typing import Callable

def get_waveforms(*indices, source_data: pd.DataFrame = None, get_indices_map: bool = False):
    """Gets plottable waveforms dataframe using .iplot(). Indices may be start,stop for slicing, or list of indices"""
    if len(indices) == 1 and isinstance(indices[0], (list, int)):
        indices = [indices[0]] if isinstance(indices[0], int) else indices[0]
    elif len(indices) == 2:
        indices = range(indices[0],indices[1])
    else:
        raise ValueError("Invalid indices provided. May be start,stop args for slicing, or list of indices")
    data_vals = source_data.values
    data = {f"[{i}] {data_vals[i][2]} dT, {data_vals[i][1]}eV": data_vals[i][3:] for i in indices}
    df = pd.DataFrame(data=data)
    if get_indices_map is True:
        return df, {j: list(data.keys())[i] for i,j in enumerate(indices)}
    return df

def parfunc(x: float, a: float, x_0: float):
    return a * (max(x - x_0, 0))**2

@nb.njit
def parfunc_arr(x: np.array, a: float, x_0: float):
    return np.array([a * (max(x_ - x_0, 0))**2 for x_ in x])

@nb.jit(forceobj=True)
def _fit_parabolas(df_arr: np.array, y_max: float, columns: list):
    """Takes array and y_max and returns fitted parabolas array"""
    # Create empty results array
    results = np.array([[0.0]*8]*len(df_arr[0]))
    
    # Fit parabola for each column
    for i,colname in enumerate(columns):
        y_data_par = df_arr[df_arr[:,i] <= y_max, i]
        y_data_lin = df_arr[-10:, i]
        # print(f"[{i}] '{wave_col}' y_data:", y_data) if print_output else None
        x_data_par = np.arange(0,len(y_data_par))
        x_data_lin = np.arange(128-10,128).reshape(-1, 1)
        
        # Parabola fit
        popt, pcov = curve_fit(parfunc_arr, x_data_par, y_data_par, bounds=([-3, -10], [3, 40]))
        reg = LinearRegression().fit(x_data_lin, y_data_lin)
        # check_df = pd.DataFrame(data = {"data": part, "fit": parfunc_arr(x_data, *popt)})
        tot_var = (parfunc_arr(x_data_par, *popt) - y_data_par)**2
        results[i][0:2] = popt
        results[i][2] = len(y_data_par)
        # results[i][3] = np.sum(tot_var)
        # results[i][4] = np.sum(tot_var[int(popt[1]):])
        results[i][3] = np.sum(tot_var) / len(tot_var)
        results[i][4] = np.sum(tot_var[int(popt[1]):]) / len(tot_var[int(popt[1]):])
        results[i][5] = (160 - float([s[s.find("]")+1:s.find("dT")] for s in [colname.replace(" ","")]][0]))/4
        results[i][6] = parfunc(results[i][5], *popt)
        results[i][7] = reg.coef_[0]
        # print(f"[{i}] '{wave_col}' parameters:", results[i], "x0_cutoff_var", tot_var[int(popt[1])]) if print_output else None
    return results

def fit_parabolas(df: pd.DataFrame, y_max: float, **kwargs):
    """Fits parabola to each column in the dataframe up to y-value ``y_max``, returns parameter df"""
    max_col = kwargs.pop("force_max_col", 127)
    if kwargs:
        raise TypeError(f"fit_parabolas() got an unexpected keyword argument '{list(kwargs.keys())[0]}'")
    if len(df.columns) > max_col:
        raise ValueError(f"Up to {max_col} columns can be fitted with this function, exceeded this by {len(df.columns)-max_col}")
    if not isinstance(y_max, float) or not (0 < y_max <= 1):
        raise ValueError(f"y_max parameter must be a float between 0 and 1")

    # Create numpy array from df for fast processing and get the results
    results = _fit_parabolas(df.to_numpy(), y_max, list(df.columns))

    results_df = pd.DataFrame(data=results,
                              columns=["a","x0","x_cutoff","var","x0_cutoff_var","xT","yT_fit","final10_slope"],
                              index=[col[:col.find("]")+1] for col in df.columns])
    return results_df.astype({"x_cutoff": 'int32'})
    # fig2 = wave_df.iplot(asFigure=True)
    # fig2.add_vline(x=(160-float([s[s.find("]")+1:s.find("dT")] for s in [col_for_str(wave_df, str(waveform)).replace(" ","")]][0]))/4, line_color="orange")
    
    # col_for_str = lambda df, col: [c for c in df.columns if col in c][0]

def df_with_fits(df: pd.DataFrame, y_max: float, return_results: bool = False, **kwargs):
    """Fits parabola to each column in the dataframe up to y-value ``y_max``, returns df with fitted columns"""
    results_df = fit_parabolas(df, y_max, **kwargs)

    # Create new dataframe with empty fit columns
    new_df = pd.concat([df]+[pd.DataFrame(data={f"Fit {i}": [np.nan]*len(df)}) for i in results_df.index], axis=1)
    
    # Add found fit values to each existing fit column
    for i in results_df.index:
        new_df[f"Fit {i}"][:results_df["x_cutoff"][i]] = parfunc_arr(df.index.to_numpy(), results_df["a"][i], results_df["x0"][i])[:results_df["x_cutoff"][i]]
    
    if return_results:
        return new_df, results_df
    return new_df


def get_pred_func(results_df: pd.DataFrame, drop_rows: list = [], bounds=([-10000, 1], [10, 20])):
    df = results_df.drop(index=drop_rows)
    popt, pcov = curve_fit(lambda x, m, b: m*x + b, df["a"], df["xT"]-df["x0"], bounds=bounds)
    # print("updated_results:", popt, pcov, "\nx0,y0 - x1,y1:", (0,popt[1]), (0.001, popt[0]/1000+popt[1]))
    pred_func = lambda a, x0: popt[0] * a + popt[1] + x0
    return pred_func

# Functions that plot predictions vs labels
def pred_trace(results_df: pd.DataFrame, drop_rows: list = [], pred_func: Callable = None, return_variance: bool= False):
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
    trace, var = pred_trace(results_df, drop_rows, pred_func, return_variance=True)
    fig = go.Figure(data=trace)
    df = results_df.drop(index=drop_rows)
    min_,max_ = df["xT"].min(),df["xT"].max()
    fig.update_layout(title='Predicted and actual xT',
                      xaxis_title="xT_pred",
                      yaxis_title="xT",
                      shapes = [{'type': 'line', 'yref': 'y', 'xref': 'x', 'y0': min_, 'y1': max_, 'x0': min_, 'x1': max_}])
    print("Variance:", var)
    return fig

def corr_trace(results_df: pd.DataFrame, drop_rows: list = [], c_min_max: tuple = (None, None)):
    return go.Scatter(x=results_df["a"].drop(drop_rows),
                      y=(results_df["xT"]-results_df["x0"]).drop(drop_rows),
                      mode='markers',
                      marker=dict(cmax=c_min_max[1] or results_df["x0_cutoff_var"].max(),
                                  cmin=c_min_max[0] or results_df["x0_cutoff_var"].min(),
                                  color=results_df["x0_cutoff_var"],
                                  colorbar=dict(title="Parabolic<br>fit error"),
                                  colorscale="Viridis"),
                      customdata=results_df[["x0_cutoff_var","xT","x_cutoff","final10_slope"]],
                      hovertemplate = '<i>%{text}</i><br>'+
                                      '<b>a</b>: %{x:.6f} <br>'+
                                      '<b>xT - x0</b>: %{y}<br>'+
                                      '<b>Par. fit error</b>: %{customdata[0]:.8f}<br>'+
                                      '<b>(xT, x_cutoff)</b>: (%{customdata[1]:.2f}, %{customdata[2]:.0f})<br>'+
                                      '<b>Final slope (10)</b>: (%{customdata[3]:.5f})',
                      text = ['Waveform {}'.format(tracename) for tracename in results_df.index])
    
def corr_fig(results_df: pd.DataFrame, drop_rows: list = [], c_min_max: tuple = (0, 3e-5), **kwargs):
    fig = go.Figure(data=corr_trace(results_df, drop_rows, c_min_max))
    fig.update_layout(title='Correlation of fitted parabola and gamma arrival time',
                      xaxis_title="a",
                      yaxis_title="xT - x0",
                     **kwargs)
    return fig

def mlp_reg_trace(data_train: np.array, labels_train: np.array, regressor, return_variance: bool= False, wave_indices = None):
    variance = np.sum(np.square(regressor.predict(data_train) - labels_train) / len(labels_train))
    text = ['Waveform' for i in range(len(labels_train))]
    if wave_indices is not None:
        text = ['Waveform {}'.format(i) for i in wave_indices]
    trace = go.Scatter(x=regressor.predict(data_train),
                       y=labels_train,
                       mode='markers',
                       # customdata=results_df[["rel_x0_cutoff_var","x_cutoff","xT"]],
                       hovertemplate = '<i>%{text}</i><br>'+
                                       '<b>xT_pred</b>: %{x:.6f}<br>'+
                                       '<b>xT</b>: %{y:.6f}',
                       text = text)
    if return_variance:
        return trace, variance
    return trace

def mlp_reg_fig(data_train: np.array, labels_train: np.array, regressor, wave_indices: np.array = None):
    trace, var = mlp_reg_trace(data_train, labels_train, regressor, return_variance=True, wave_indices=wave_indices)
    fig = go.Figure(data=trace)
    min_,max_ = np.min(trace.x),np.max(trace.x)
    fig.update_layout(title='Predicted and actual xT',
                      xaxis_title="xT_pred",
                      yaxis_title="xT",
                      shapes = [{'type': 'line', 'yref': 'y', 'xref': 'x', 'y0': min_, 'y1': max_, 'x0': min_, 'x1': max_}])
    print("Variance:", var)
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