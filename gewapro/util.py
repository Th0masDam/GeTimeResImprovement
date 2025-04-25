import os
import pandas as pd
import numpy as np
from typing import Iterable, Callable
from functools import partial
import warnings
import re
from gewapro.functions import rmse, var
from contextlib import contextmanager

class strictclassmethod:
    """Behaves as the built-in ``classmethod`` decorator, but will raise an error when called on instances"""
    def __init__(self, method):
        self.method: Callable = method
    def __get__(self, instance, cls):
        if instance:
            raise TypeError(f"The {self.method.__qualname__} method can only be called on the {cls.__name__} class, not on instances of {cls.__name__}")
        def clsmethod(*args, **kwargs):
            return self.method(cls, *args, **kwargs)
        return clsmethod

def print_warnings(warning_format: str = "[WARNING] <TyPe>: <Message>"):
    """Catches all warnings that the function execution raises and prints them according to `warning_format`

    Follows the capitalization of \\<type\\> and \\<message\\>, e.g. "UserWarning: test warning you CANNOT miss!" with format "[\\<TYPE\\>]
    \\<Message\\>" will be printed as "[USERWARNING] Test warning you CANNOT miss!"
    """
    if not isinstance(warning_format, str):
        raise TypeError("warning_format must be a string")
    def warn_decorator(func: Callable, warning_format: str):
        return warn_wrapper(func, warning_format=warning_format)
    return partial(warn_decorator, warning_format=warning_format)

def warn_wrapper(wrapped: Callable, warning_format: str):
    """Warning wrapper for functions, it is advised to use the function decorator ``@print_warnings()`` instead"""
    # Function that replaces type and message in the format with values
    def _replace_from_format(formatter: str, type: str, message: str):
        replacer = {formatter[(sl:=formatter.lower().find("<type>")):sl+6]:str(type),
                    formatter[(sl:=formatter.lower().find("<message>")):sl+9]:str(message)}
        for k,v in replacer.items():
            for stringmethod in ["lower","upper","capitalize"]:
                if k and k[1:-1] == k[1:-1].__getattribute__(stringmethod)():
                    if stringmethod == "capitalize":
                        replacer[k] = v.upper()[:1]+v[1:]
                    else:
                        replacer[k] = v.__getattribute__(stringmethod)()
            formatter = formatter.replace(k,replacer[k]) if k else formatter
        return formatter

    # Create the wrapped function
    def warn_func(*args, **kwargs):
        # Catch all warnings
        with warnings.catch_warnings(record=True) as caught_warnings:
            output = wrapped(*args, **kwargs)
            for warning in caught_warnings:
                # Print each warning as it is caught
                print(_replace_from_format(warning_format,warning.category.__name__,warning.message))
            return output
    return warn_func

@contextmanager
def modify_message(*exceptions: Exception, prepend_msg: str = "", append_msg: str = "", replace: dict[str,str] = {}, notes: dict[str,str] = {}):
    """Context manager that modifies error messages, but keeps all else (e.g. traceback) intact.
    
    Arguments
    ---------
    *exceptions: Exception
        Exceptions to catch. If not specified, catches all Exception types.
    prepend_msg: str
        String to prepend to all errors raised.
    append_msg: str
        String to append to all errors raised.
    replace: dict[str, str]
        Dictionary with key-value pairs that replace key with value if key is found in the error message.
    notes: dict[str, str]
        Dictionary that adds the value as a note if the key is found in the error message (after replacement by replace).
    """
    if any(invalid_excepts := [not (isinstance(excep, Exception) or "Exception" in str(excep)) for excep in exceptions]):
        raise ValueError(f"Got non-ExceptionType in exceptions arguments: {[t[0] for t in zip(exceptions,invalid_excepts) if t[1]==1]}")

    def _replace(string: str, replacer: dict[str,str]):
        for replace in replacer:
            string = string.replace(replace, replacer[replace])
        return string

    def _add_notes(err: Exception, checker: dict[str,str]):
        for string in checker:
            if string in err.args[0]:
                err.add_note(checker[string])
        return err

    if not exceptions:
        exceptions = Exception
    try:
        yield # You can return something if you want, that gets picked up in the 'as'
    except exceptions as err:
        err.args = ((f"{prepend_msg}{_replace(arg,replace)}{append_msg}" if i == 0 else arg) for i,arg in enumerate(err.args))
        raise _add_notes(err, notes)
    finally:
        pass

def _validate_a_b(correct_energy: tuple[float,float]|dict) -> tuple[float,float]:
    if len(correct_energy) == 2 and (isinstance(correct_energy, (list,tuple)) and all([isinstance(i,(float,int)) for i in correct_energy])) or (
        isinstance(correct_energy, dict) and all([isinstance(correct_energy.get(s,None),float) for s in ["a","b"]])):
            a,b = list(correct_energy.values()) if isinstance(correct_energy, dict) else correct_energy
            return a,b
    else:
        raise ValueError("correct_energy must be a list/dict of two floats (a & b) where a is a factor and b is an offset")

def correct_energy(correct_energy: tuple[float,float], energy_labels: pd.Series|np.ndarray):
    a,b = _validate_a_b(correct_energy=correct_energy)
    return a*energy_labels+b

def invert_start_end_step(start: int, end: int, step: int, a: float, b: float) -> tuple[int,int,int]:
    """Gets original start end step from newly corrected start end step"""
    len_ = get_len(start,end,step)
    return round((start-b)/a),round(start/a+len_*round(step/a)-b/a),round(step/a)

def correct_start_end_step(start: int, end: int, step: int, a: float, b: float) -> tuple[int,int,int]:
    """Gets newly corrected start end step"""
    len_ = get_len(start,end,step)
    return round(a*start+b),round(a*start+len_*round(a*step)+b),round(a*step)

def get_len(start: int,end: int,step: int) -> int:
    """Gets the number of periods"""
    if (len_ := (end-start)//step) == (end-start)/step:
        return len_
    raise ValueError("Invalid len, start to end is no integer steps")

def join_strings(iterable: Iterable, bound: int=5, final_connector: str = "or", wrap_str: bool|str="\"", wrap_all: bool|str = False) -> str:
    """Joins all items in ``iterable`` (up to max ``bound``) with ',' and the final with ``final_connector``
    
    Strings will be wrapped with the ``wrap_str`` argument, by default with double quotes (")"""
    bound,show = (bound, bound-2) if isinstance(bound,int) else bound
    fc = " "+final_connector
    wrap_str = str(wrap_str) if wrap_str else ""
    ls = [(wrap_str+i+wrap_str if (isinstance(i,str)) else str(i)) for i in iterable] if wrap_str else [str(i) for i in iterable]
    if wrap_all and isinstance(wrap_all, str):
        ls = [wrap_all+s+wrap_all for s in ls]
    new_list = ls[:show-1]+[f"... <{len(ls)-show} more> ..."]+ls[-1:] if (bound and len(ls) > bound) else ls
    return_str = f"{fc} ".join(new_list).replace(f"...{fc}","...").replace(f"{fc} ",", ",len(ls)-2)
    return return_str

def pandas_string_rep(obj: pd.DataFrame|pd.Series, bound: int|tuple[int,int]=5):
    """Returns compact string representation of Series or DataFrame"""
    if isinstance(obj, pd.DataFrame):
        return f"<DataFrame (columns: {join_strings([f"\"{tup[0]}\" ({tup[1]})" for tup in zip(obj.columns,obj.dtypes)],bound,"&",wrap_str="")})>"
    elif isinstance(obj, pd.Series):
        return "<Series ("+(f"name: \"{obj.name}\", " if obj.name else "")+f"values ({obj.dtype}): {join_strings(obj,bound,"&")})>"
    else:
        raise ValueError("obj must be a Series or DataFrame")

def add_notes(error: Exception, *notes):
    """Adds notes to an Exception if they are non-empty strings"""
    for note in notes:
        error.add_note(note) if (note and isinstance(note,str)) else None
    return error

def stats(df: pd.DataFrame):
    data = {}
    for col in df.columns:
        data[col] = [len(df[col]),df[col].mean(),rmse(df[col]),rmse(df[col],"+"),rmse(df[col],"-"),var(df[col],"+"),var(df[col],"-")]
    return pd.DataFrame(data, index=["N","Mean","RMSE","RMSE+","RMSE-","VAR+","VAR-"])

def remove_all_input_examples_locally():
    """Remove all the input examples from the models in all runs"""
    exp_folders = [ f.path for f in os.scandir("mlruns") if (f.is_dir() and not ".trash" in f.path and not "models" in f.path)]
    i = 0
    for exp_folder in exp_folders:
        run_folders = [ f.path for f in os.scandir(exp_folder) if (f.is_dir() and not "tags" in f.path and not "datasets" in f.path)]
        print("[GeWaPro][util.remove_all_input_examples_locally] Checking folder",exp_folder, "with runs "+", ".join(run_folders)) if run_folders else None
        for run_folder in run_folders:
            model_folder = [f.path for f in os.scandir(os.path.join(run_folder,"artifacts")) if f.is_dir() ]
            for model_f in model_folder:
                # print("[GeWaPro][util.remove_all_input_examples_locally] Found model folder:",model_folder)
                if os.path.isfile(os.path.join(model_f, "input_example.json")):
                    os.remove(os.path.join(model_f, "input_example.json"))
                    print("[GeWaPro][util.remove_all_input_examples_locally] Removed",(os.path.join(model_f, "input_example.json")))
                    i += 1
                else:
                    continue
                if os.path.isfile(os.path.join(model_f, "serving_input_example.json")):
                    os.remove(os.path.join(model_f, "serving_input_example.json"))
                    print("[GeWaPro][util.remove_all_input_examples_locally] Removed",(os.path.join(model_f, "serving_input_example.json")))
                    i += 1
    print("[GeWaPro][util.remove_all_input_examples_locally] Done.", i, "files removed")

def get_column_map(df: pd.DataFrame, valid_columns: dict[str,type]|list[str]) -> dict:
    """Gets a numerical column map of the columns in valid_colums to the DataFrame, ordered as in valid_columns"""
    defaults = dict(zip(valid_columns,["-"]*len(valid_columns)))
    if isinstance(valid_columns, dict):
        return defaults | {col:[i,valid_columns[col]] for i,col in enumerate(df.columns) if col in valid_columns}
    elif isinstance(valid_columns, (list,tuple,set)):
        return defaults | {col:i for i,col in enumerate(df.columns) if col in valid_columns}
    else:
        raise ValueError("valid_columns must be list or dict")

def cols_to_name(j: int, column_map: dict, data_values: np.ndarray) -> str:
    """Creates waveform column name from values"""
    return f"[{j}] "+", ".join([f"{col} {'-' if val == '-' else val[1](data_values[j][val[0]])}" for col,val in column_map.items()])

def name_to_vals(col_name:str) -> dict[str,float]:
    """Converts a column name back into its constituent values"""
    names_list = col_name.split(", ")
    names_list = [name.split(" ") for name in names_list]
    # Remove the waveform index
    names_list[0].pop(0)
    return {k:v if v=="-" else get_int_or_float(v) for [k,v] in names_list}

def get_int_or_float(x: str):
    "Gets int or float from string"
    try:
        return int(x)
    except ValueError:
        return float(x)

def sort(value: Iterable,/, convert_strings_to_values: bool = True) -> list|dict:
    """Smart sorter, sorts e.g. ``['12', 1, '40', 'B', None, 'None', 2.4, 'A', '[13 13]', '[13 1]', np.inf]`` as ``[None, None, 1, 2.4, 12, 40, np.inf, [13 1], [13 13], 'A', 'B']``
    
    Always sorts in the order: None/nans, ints/floats, lists, strings (alphabetically)

    NOTE: Tuples or sets cannot be sorted, dictionaries will be sorted by their keys. Note that convert_strings_to_values does not work for dicts (keys will remain string)
    """
    if (i := 2) and convert_strings_to_values:
        i = 0
    if isinstance(value, dict):
        sorted_lists = _sort(list(value.keys()))
        return {sorted_lists[2][j]:value[k] for j,k in enumerate(sorted_lists[2])}
    return _sort(value)[i]

def isort(value: Iterable) -> list:
    """Returns the indices of the input list to sort according to the smart `sort(...)` function

    NOTE: Tuples or sets cannot be sorted, dictionaries will be sorted by their keys
    """
    if isinstance(value, dict):
        return _sort(list(value.keys()))[1]
    return _sort(value)[1]

def _sort(value: Iterable) -> tuple[list,list,list]:
    """Returns sorted,i_sorted,sorted_original lists tuple"""
    sortable_nans,nans_i = [],[]
    sortable_floats,floats_i = [],[]
    sortable_lists,lists_i = {},[]
    sortable_strings,strings_i = [],[]
    i_mapper = {}
    if isinstance(value, str):
        raise ValueError("Strings are not supported, got string to sort: \""+value+'"')
    elif isinstance(value, dict):
        raise ValueError(f"Dicts are not supported, got dict to sort: {value}")
    for i,v in enumerate(value):
        i_mapper[i] = v
        if isinstance(v, str):
            try:
                if (v_strp := v.strip('" \'')).lower() == "nan":
                    sortable_nans.append(np.nan),nans_i.append(i)
                    i_mapper[i] = np.nan
                elif v_strp.lower() == "none":
                    sortable_nans = [None]+sortable_nans
                    nans_i.append(i)
                    i_mapper[i] = None
                else:
                    sortable_floats.append(get_int_or_float(v_strp)),floats_i.append(i)
                    i_mapper[i] = get_int_or_float(v_strp)
            except ValueError:
                raised = True
                if (v_strp.startswith("[") and v_strp.endswith("]")):
                    try:
                        float_ls = [get_int_or_float(x) for x in re.findall(r"[-+]?(?:\d*\.*\d+)", v_strp[1:-1])]
                        raised = False if float_ls else True
                        sortable_lists.setdefault(len(float_ls),[])
                        sortable_lists[len(float_ls)].append(float_ls)
                    except ValueError:
                        raised = True
                if raised:
                    strings_i.append(i),sortable_strings.append(v)
                else:
                    lists_i.append(i)
                    i_mapper[i] = float_ls
        elif isinstance(v, (int,float)):
            if np.nan is v:
                sortable_nans.append(v),nans_i.append(i)
            else:
                sortable_floats.append(v),floats_i.append(i)
        elif isinstance(v, list):
            sortable_lists.setdefault(len(v),[])
            sortable_lists[len(v)].append(v),lists_i.append(i)
        elif v is None:
            sortable_nans = [None]+sortable_nans
            nans_i.append(i)
        else:
            raise ValueError(f"For sorting, only bools, ints, floats, strings, lists or None are supported. Got {v.__class__.__qualname__}: {v}")
    sorted_lists = []
    if sortable_lists:
        for j in range(max(sortable_lists.keys())):
            sorted_lists.extend(sorted(sortable_lists.get(j+1,[])))
    if len(return_val := sortable_nans+sorted(sortable_floats)+sorted_lists+sorted(sortable_strings)) != len(value):
        raise RuntimeError(f"Got sorted list (length {len(return_val)}) that was not equal in length as input list (length {len(value)})")

    # Get all indices from input to sorted list
    final_i_ls,start_len = [],0
    for ls in [nans_i,floats_i,lists_i,strings_i]:
        ivals = [j for k in range(start_len,start_len+len(ls)) for j in ls if (i_mapper[j] == return_val[k] or i_mapper[j] is return_val[k])]
        start_len += len(ls)
        uivals = []
        for k in ivals:
            uivals.append(k) if k not in uivals else None
        final_i_ls.extend(uivals)
    return return_val,final_i_ls,[value[k] for k in final_i_ls]

def combine_cols_with_errors(df: pd.DataFrame, error_suffix: str = " SD", round: bool|int = False):
    """Combines all columns that have an error column to an error margin (dtype to object), e.g. ``["val","val SD"]:[1.0,0.1]`` -> ``["val"]:["1.0 ± 0.1"]``"""
    paired_columns = [(col,col+error_suffix) if (col+error_suffix in df.columns) else col for col in df.columns if not col.endswith(error_suffix)]
    if round is not False:
        df_new = df.round(round)
        return pd.concat([df[col] if isinstance(col,str) else pd.DataFrame({col[0]:df_new[col[0]].astype(str) + " ± " + df_new[col[1]].astype(str)}) for col in paired_columns],axis=1)
    return pd.concat([df[col] if isinstance(col,str) else pd.DataFrame({col[0]:df[col[0]].astype(str) + " ± " + df[col[1]].astype(str)}) for col in paired_columns],axis=1)
