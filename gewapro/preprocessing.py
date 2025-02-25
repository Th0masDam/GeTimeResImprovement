import pandas as pd
import numpy as np
from typing import Tuple, List, Literal, Callable
from sklearn.model_selection import train_test_split
from gewapro.functions import combine_or, combine_and
from gewapro.util import pandas_string_rep
from gewapro.cache import cache
import os

def train_test_split_cond(*arrays, test_size=None, train_size=None, random_state=None, condition=None, add_removed_to_test=False):
    """Creates a train-test split on `arrays` given a certain `condition`, adds conditionally removed data to test data if `add_removed_to_test` is `True`"""
    conditioned_arrays = arrays
    not_conditioned_arrays = []
    if not condition is None:
        conditioned_arrays = []
        for array in arrays:
            conditioned_arrays.append(array[condition])
            not_conditioned_arrays.append(array[~condition])
    split_arrays = train_test_split(*conditioned_arrays, test_size=test_size, train_size=train_size, random_state=random_state)
    if not_conditioned_arrays and add_removed_to_test:
        if condition is None:
            raise ValueError("Outside condition values cannot be added to test set if there is no condition provided")
        else:
            for i in range(len(split_arrays)):
                if i % 2 == 1:
                    split_arrays[i] = np.append(split_arrays[i], not_conditioned_arrays[(i-1)//2], axis=0)
                    print(f"Appended not conditioned array {i}/{len(not_conditioned_arrays)} to split array index {i}")
    return split_arrays

# def get_split_waveforms(uniform_test_set: list[int] = []):
#     data_df = get_waveforms(source_data=data, get_indices_map=False, include_energy=include_energy, select_energies=select_energies, select_channels=select_channels)
#     if uniform_test_set:
#         if not (isinstance(uniform_test_set, list) and all([isinstance(x,int) for x in uniform_test_set])):
#             raise ValueError("Uniform test set must be a list of integer energy bounds (energy in arb. units)")
#         test_ranges = sorted([(x,y) for x,y in list(zip(uniform_test_set[:-1],uniform_test_set[1:])) if x<y])
#         test_sets = {f"{test_range}": get_waveforms(source_data=data,
#                                                     get_indices_map=False,
#                                                     include_energy=include_energy,
#                                                     select_energies=test_range,
#                                                     select_channels=select_channels) for test_range in test_ranges}
#         smallest_set = min(*[len(df.columns) for df in test_sets.values()])
#         test_sets = {k: df.iloc[:,:smallest_set] for k,df in test_sets.items()}
#         print(f"[MLFlow run] Created uniform test sets of length {smallest_set} in energy (arb. unit) ranges:",test_ranges)

def select_from_source(source_data: pd.DataFrame,
                       select_channels: list[int]|int = [],
                       select_energies: tuple[int, int] = ()):
    """Selects a part of the source data (NOT as a deep copy) based on channels and/or energies"""
    if not isinstance(source_data, pd.DataFrame):
        raise TypeError(f"source_data must be a DataFrame, got {source_data}")
    elif isinstance(source_data, pd.DataFrame) and any([col not in source_data.columns for col in ["Ch","E"]]):
        raise ValueError(f"source_data must be a DataFrame with columns 'Ch' and 'E', but got: {pandas_string_rep(source_data)}")
    if select_channels:
        channel_condition = combine_or(*[source_data["Ch"] == ch for ch in ([select_channels] if isinstance(select_channels,int) else select_channels)])
    else:
        channel_condition = np.array([True]*len(source_data))
    if select_energies:
        energy_condition = (source_data["E"] >= select_energies[0]) & (source_data["E"] < select_energies[1])
    else:
        energy_condition = np.array([True]*len(source_data))
    return source_data[channel_condition & energy_condition]

def get_waveforms(*indices: int,
                  source_data: pd.DataFrame,
                  get_indices_map: bool = False,
                  include_energy: bool = False,
                  select_channels: list[int]|int = [],
                  select_energies: tuple[int, int] = (),
                 ):
    """Gets plottable waveforms dataframe using .iplot(). Indices may be start,stop for slicing, or list of indices
    
    Arguments
    ---------
    *indices : int
        Waveform indices to use. May be a single index, ``[start,stop]`` for slicing, or list of individual indices
    source_data : pd.DataFrame
        Source data to use, that contains the normalized waveforms.
    get_indices_map : bool, default False
        Whether to also output a dict with indices for each of the column names
    include_energy : bool, default False
        Whether to include the energy data scaled by 1/50 000 th in the first row of the waveform
    select_channels : List[int]
        List of which channels to select. If empty or None, gives full data set (default behaviour)
    select_energies : Tuple[int, int], default ()
        (min, max) tuple of which energies to select (right bound exclusive) in the data set. If empty or None, gives
        full data set (default behaviour)
    """
    source_data = select_from_source(source_data=source_data, select_channels=select_channels, select_energies=select_energies)

    if len(indices) == 1 and isinstance(indices[0], (list, int)):
        indices = [indices[0]] if isinstance(indices[0], int) else indices[0]
    elif len(indices) == 2:
        indices = range(indices[0],indices[1])
    elif len(indices) == 0 or (len(indices) == 1 and indices[0] is None):
        indices = range(0, len(source_data))
    else:
        raise ValueError("Invalid indices provided. May be start,stop args for slicing, or list of indices")

    data_vals = source_data.values
    i = {"dT":"-"} | {col:i for i,col in enumerate(source_data.columns) if col in ["Ch","E","Tref","dT","s0"]}
    # print(source_data.columns, i,"\n", source_data.head(n=3))
    if include_energy: # Includes 1/50_000th of the energy in eV (max was ~56e3 eV)
        data = {f"[{j}] Tref {data_vals[j][i['Tref']]}, dT {'-' if i['dT'] == '-' else data_vals[j][i['dT']]}, E {int(data_vals[j][i['E']])}": np.append(data_vals[j][i['E']]*2e-5, data_vals[j][i['s0']:]) for j in indices}
    else:
        data = {f"[{j}] Tref {data_vals[j][i['Tref']]}, dT {'-' if i['dT'] == '-' else data_vals[j][i['dT']]}, E {int(data_vals[j][i['E']])}": data_vals[j][i['s0']:] for j in indices}
    df = pd.DataFrame(data=data)
    if get_indices_map is True:
        return df, {j: list(data.keys())[i] for i,j in enumerate(indices)}
    return df

def smoothen_waveforms(wave_df: pd.DataFrame,
                       smoothing_window: int|tuple[int,int],
                       apply_to_energies: tuple[int, int] = None,
                       in_place: bool|Literal["replace"] = True,
                       normalize: bool|Literal["original"] = "original"):
    """Smoothens waveforms using a uniform rolling window on their derivatives. Window size equal to noise frequency works best
    
    Arguments
    ---------
    wave_df : pd.DataFrame
        The input waveform DataFrame as gotten from the ``get_waveforms`` function
    smoothing_window : int | tuple[int, int]
        The smoothing window to apply. May be *int* (single smoothing) or *int, int* tuple (doubly applied smoothing).
        Found empirically that often ``22,40`` double smoothing gives good results
    apply_to_energies : Tuple[int, int], default None
        Energy range (min,max) to apply the smoothing to (right bound exclusive). If None or empty, applies to all waveforms given
        (default behaviour)
    in_place : bool | Literal["replace"], default False
        Whether to make a copy of the input DataFrame (*False*, default behaviour), or edit the original DataFrame (*True*). If
        "replace", replaces the original columns with the smoothed versions
    normalize : bool | int | Literal["original"], default True
        Normalizes the waveform amplitude (post-smoothing) to this value. If False, does not normalize. By default, normalizes final
        waveform to original amplitude.
    """
    sw = smoothing_window
    length = len(wave_df[list(wave_df.columns)[0]])
    i = 1 if (length in [161, 201]) else 0
    if length not in [160, 161, 200, 201]:
        raise ValueError(f"Unknown waveform length {length}: only 160, 161, 200 and 201 are supported")
    df = wave_df.copy(deep=True) if not in_place else wave_df
    if isinstance(sw, int):
        w0, w1 = sw, None
    elif isinstance(sw, tuple) and len(sw) == 2 and isinstance(sw[0], int) and isinstance(sw[1], int):
        w0, w1 = sw
    else:
        raise ValueError("smoothing_window must be an int (single smoothing) or a length 2 int tuple (double smoothing), e.g. 23 or 22,40")
    if apply_to_energies and not (len(apply_to_energies) == 2 and isinstance(apply_to_energies[0], (int,float)) and isinstance(apply_to_energies[1], (int,float))):
        raise ValueError("apply_to_energies must be a list or tuple of length 2")
    elif isinstance(apply_to_energies, tuple) and apply_to_energies[1] <= apply_to_energies[0]:
        raise ValueError(f"apply_to_energies '{apply_to_energies}' invalid: second number must be higher than first")

    for col in list(df.columns):
        if not apply_to_energies or apply_to_energies[0] <= float(col[col.find(", ")+1:col.find("eV")]) < apply_to_energies[1]:
            amplitude_goal = df[col].max() if str(normalize) == "original" else normalize
            new_col_name = col[:col.find("]")+1]+f" smoothed_{w0}"+(f"_{w1}" if w1 else "")
            if str(in_place).lower() in ["replace","replace_col","replace_cols","replace_columns"]:
                new_col_name = col
            else:
                df[new_col_name] = df[col]
            if w1:
                df[new_col_name][i:] = df[col][i:].diff().rolling(w0,center=True,min_periods=1).mean().rolling(w1,center=True,min_periods=1).mean().cumsum()
            else:
                df[new_col_name][i:] = df[col][i:].diff().rolling(w0,center=True,min_periods=1).mean().cumsum()
            if normalize:
                df[new_col_name][i:] = df[new_col_name][i:] * amplitude_goal/df[new_col_name][i:].max()
    return df

def _post_cache(wrapped: Callable, kwargs: dict, call_results: List[pd.DataFrame]):
    include_energy = kwargs["include_energy"]
    return call_results[0].loc[int(not include_energy):].reset_index().drop(columns=["index"], errors="ignore")

@cache(cache_dir=os.path.join("data","cache"), ignore_args=["include_energy"], post_cache=_post_cache)
def get_and_smoothen_waveforms(source_data_path: os.PathLike|str,
                               include_energy: bool,
                               select_channels: List[int]|int,
                               select_energies: Tuple[int, int] = (),
                               smoothing_window: int|Tuple[int,int] = (),
                               apply_to_energies: Tuple[int, int] = None,
                               in_place: bool|Literal["replace"] = "replace",
                               normalize: int|Literal["original"] = True):
    """Combines the get_waveforms and smoothen_waveforms functions
    
    Arguments
    ---------
    source_data_path : PathLike|str
        Path to the source data for the waveforms
    include_energy : bool
        Whether to include the energy data scaled by 1/50 000 th in the first row of the waveform
    select_channels : List[int]
        List of which channels to select
    select_energies : Tuple[int, int], default ()
        (min, max) tuple of which energies to select (right bound exclusive) in the data set. If empty or None, gives full data set
    smoothing_window : int | Tuple[int, int]
        The smoothing window to apply. May be *int* (single smoothing) or *int, int* tuple (doubly applied smoothing).
        Found empirically that often ``22,40`` double smoothing gives good results
    apply_to_energies : Tuple[int, int], default None
        Energy range to apply the smoothing to. If None or empty, applies to all waveforms given (default)
    in_place : bool | Literal["replace"], default False
        Whether to make a copy of the input DataFrame (*False*, default), or edit the original DataFrame (*True*). If "replace", replaces
        the original columns with the smoothed versions
    normalize : bool | int | Literal["original"], default True
        Normalizes the waveform amplitude (post-smoothing) to this value. If False, does not normalize. By default, normalizes final
        waveform to original amplitude.
    """
    # if not smoothing_window:
    #     raise TypeError("get_and_smoothen_waveforms() missing 1 required positional argument: 'smoothing_window'")
    wave_df = get_waveforms(source_data=pd.read_csv(source_data_path),
                            include_energy=True,
                            select_channels=select_channels,
                            select_energies=select_energies)
    if smoothing_window:
        smoothen_waveforms(wave_df,
                        smoothing_window=smoothing_window,
                        apply_to_energies=apply_to_energies,
                        in_place=in_place,
                        normalize=normalize)
    return wave_df.loc[int(not include_energy):].reset_index().drop(columns=["index"])

# def _process_tup(wrapped: Callable, tup: tuple, cache_dir: str) -> dict:
#     """Default ``pre_cache`` function, returns ``{<file_path>: tup}`` dict"""
#     file_path = f"{os.path.join(cache_dir, f'{wrapped.__name__}_{encode64(tup)}')}.parquet"
#     return {file_path: tup}

# def _process_results(wrapped: Callable, tup: tuple, call_results: List[pd.DataFrame]):
#     """Default ``post_cache`` function, returns first list item of `call_results`"""
#     return call_results[0]
