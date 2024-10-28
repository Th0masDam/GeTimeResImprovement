import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import numba as nb
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

# def get_pred_func(results_df: pd.DataFrame, drop_rows: list = [], bounds=([-10000, 1], [10, 20])):
#     df = results_df.drop(index=drop_rows)
#     popt, pcov = curve_fit(lambda x, m, b: m*x + b, df["a"], df["xT"]-df["x0"], bounds=bounds)
#     # print("updated_results:", popt, pcov, "\nx0,y0 - x1,y1:", (0,popt[1]), (0.001, popt[0]/1000+popt[1]))
#     pred_func = lambda a, x0: popt[0] * a + popt[1] + x0
#     return pred_func

@np.vectorize
def combine_and(*conditions):
    """Combines conditions as if using the '&' operator"""
    for condition in conditions:
        if not condition:
            return False
    return True

@np.vectorize
def combine_or(*conditions):
    """Combines conditions as if using the '|' operator"""
    for condition in conditions:
        if condition:
            return True
    return False

def quadratic(x: float, a: float, x0: float):
    return a * (max(x - x0, 0))**2

def exponential(x: float, b: float, c: float):
    return 1-b**(-x+c)

def gaussian(x: float, a: float, x0: float, sigma: float):
    return a/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-x0)**2/(2*sigma**2))

def inverse_quadratic(x: float, a: float, x0: float, sigma: float):
    b = 3.37580799191/(sigma**2)
    return a*np.sqrt(b)/(np.pi*(1+b*(x-x0)**2))

@nb.njit
def quadratic_arr(x: np.array, a: float, x0: float):
    """Same as ``gewapro.functions.quadratic``, but supports numpy ndarrays as input"""
    return np.array([a * (max(x_ - x0, 0))**2 for x_ in x])

@nb.njit
def exponential_arr(x: np.array, b: float, c: float):
    """Same as  ``gewapro.functions.exponential``, but supports numpy ndarrays as input"""
    return np.array([1 - b ** (-x_ + c) for x_ in x])

@nb.njit
def gaussian_arr(x: np.array, a: float, x0: float, sigma: float):
    """Same as  ``gewapro.functions.gaussian``, but supports numpy ndarrays as input"""
    return np.array([a/(sigma*np.sqrt(2*np.pi))*np.exp(-(x_-x0)**2/(2*sigma**2)) for x_ in x])

@nb.njit
def inverse_quadratic_arr(x: np.array, a: float, x0: float, sigma: float):
    """Same as  ``gewapro.functions.inverse_quadratic``, but supports numpy ndarrays as input"""
    b = 3.37580799191/(sigma**2)
    return np.array([a*np.sqrt(b)/(np.pi*(1+b*(x_-x0)**2)) for x_ in x])


@nb.jit(forceobj=True)
def _fit_parabolas(df_arr: np.array, y_max: float, columns: list):
    """Takes array and y_max and returns fitted parabolas array"""
    # Create empty results array
    results = np.array([[0.0]*7]*len(df_arr[0]))
    
    # Fit parabola for each column
    for i,colname in enumerate(columns):
        y_data_par = df_arr[df_arr[:,i] <= y_max, i]
        x_data_par = np.arange(0,len(y_data_par))
        
        # Parabola fit
        popt, _ = curve_fit(quadratic_arr, x_data_par, y_data_par, bounds=([-3, -10], [3, 40]))
        tot_var = (quadratic_arr(x_data_par, *popt) - y_data_par)**2
        results[i][0:2] = popt
        results[i][2] = len(y_data_par)
        results[i][3] = np.sum(tot_var) / len(tot_var)
        results[i][4] = np.sum(tot_var[int(popt[1]):]) / len(tot_var[int(popt[1]):])
        results[i][5] = (160 - float([s[s.find("]")+1:s.find("dT")] for s in [colname.replace(" ","")]][0]))/4
        results[i][6] = quadratic(results[i][5], *popt)
        # print(f"[{i}] '{wave_col}' parameters:", results[i], "x0_cutoff_var", tot_var[int(popt[1])]) if print_output else None
    return results

@nb.jit(forceobj=True)
def _fit_final_slope(df_arr: np.array, columns: list, final_points: int = 10):
    """Takes array and columns and returns fitted final slope array"""
    # Create empty results array
    results = np.array([[0.0]]*len(df_arr[0]))
    
    # Fit final slope for each column
    for i,colname in enumerate(columns):
        x_data_lin = np.arange(128-final_points,128).reshape(-1, 1)
        y_data_lin = df_arr[-final_points:, i]
        reg = LinearRegression().fit(x_data_lin, y_data_lin)
        results[i][0] = reg.coef_[0]

    return results

@nb.jit(forceobj=True)
def _fit_exponents(df_arr: np.array, y_min: float, columns: list):
    """Takes array and columns and returns fitted final slope array"""
    # Create empty results array
    results = np.array([[0.0]*4]*len(df_arr[0]))
    
    # Fit exponent for each column
    for i,colname in enumerate(columns):
        y_data_exp = df_arr[df_arr[:,i] >= y_min, i]
        x_data_exp = np.arange(128-len(y_data_exp),128)
        if len(y_data_exp) == 0:
            results[i][:] = [np.nan, np.nan, 128, np.nan]
            continue
        
        # Exponential fit
        popt, _ = curve_fit(exponential_arr, x_data_exp, y_data_exp, bounds=([1.001, 0], [1.100, 128]))
        tot_var = (exponential_arr(x_data_exp, *popt) - y_data_exp)**2
        results[i][0:2] = popt
        results[i][2] = x_data_exp[0]
        results[i][3] = np.sum(tot_var) / len(tot_var)

    return results

def fit_parabolas(df: pd.DataFrame, y_max: float, **kwargs):
    """Fits parabola to each column in the dataframe up to y-value ``y_max``, returns parameter df"""
    max_col = kwargs.pop("force_max_col", 127)
    include_exp = kwargs.pop("exponent_fit", False)
    y_min = kwargs.pop("y_min",-1) if include_exp else 0.5
    if y_min == -1:
        raise ValueError("Missing keyword argument y_min while exponent_fit is True")
    if kwargs:
        raise TypeError(f"fit_parabolas() got an unexpected keyword argument '{list(kwargs.keys())[0]}'")
    if len(df.columns) > max_col:
        raise ValueError(f"Up to {max_col} columns can be fitted with this function, exceeded this by {len(df.columns)-max_col}")
    if not isinstance(y_max, float) or not (0 < y_max <= 1):
        raise ValueError(f"y_max argument must be a float between 0 and 1")
    if not isinstance(y_min, float) or not (0 < y_min <= 1):
        raise ValueError(f"y_min argument must be a float between 0 and 1")

    # Create numpy array from df for fast processing and get the results
    df_vals = df.to_numpy()
    if include_exp:
        results = np.append(_fit_parabolas(df_vals, y_max, df.columns), _fit_exponents(df_vals, y_min, df.columns), axis=1)
    else:
        results = np.append(_fit_parabolas(df_vals, y_max, df.columns), _fit_final_slope(df_vals, df.columns), axis=1)

    results_df = pd.DataFrame(data=results,
                              columns=["a","x0","x_cutoff","var","x0_cutoff_var","xT","yT_fit"]+
                              (["b","c","exp_start","exp_var"] if include_exp else ["final10_slope"]),
                              index=[col[:col.find("]")+1] for col in df.columns])
    return results_df.astype({"x_cutoff": 'int32', "exp_start": 'int32'} if include_exp else {"x_cutoff": 'int32'})
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
        new_df[f"Fit {i}"][:results_df["x_cutoff"][i]] = quadratic_arr(df.index.to_numpy(), results_df["a"][i], results_df["x0"][i])[:results_df["x_cutoff"][i]]
        if "b" in results_df.columns:   
            new_df[f"Fit {i}"][results_df["exp_start"][i]:] = exponential_arr(df.index.to_numpy(), results_df["b"][i], results_df["c"][i])[results_df["exp_start"][i]:]
    
    if return_results:
        return new_df, results_df
    return new_df
