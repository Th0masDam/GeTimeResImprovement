import os
import pandas as pd
import numpy as np
from typing import Iterable
from gewapro.functions import rmse, var

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
    for note in notes:
        error.add_note(note) if note else None
    return error

def stats(df: pd.DataFrame):
    data = {}
    for col in df.columns:
        data[col] = [len(df[col]),df[col].mean(),rmse(df[col]),rmse(df[col],"+"),rmse(df[col],"-"),var(df[col],"+"),var(df[col],"-")]
    return pd.DataFrame(data, index=["N","Mean","RMSE","RMSE+","RMSE-","VAR+","VAR-"])

def remove_all_input_examples_locally():
    """Remove all the input examples from the models in all runs"""
    exp_folders = [ f.path for f in os.scandir("mlruns") if (f.is_dir() and not ".trash" in f.path and not "models" in f.path)]
    # print(exp_folders)
    i = 0
    for exp_folder in exp_folders:
        run_folders = [ f.path for f in os.scandir(exp_folder) if (f.is_dir() and not "tags" in f.path and not "datasets" in f.path)]
        print("Checking folder",exp_folder, "with runs "+", ".join(run_folders)) if run_folders else None
        for run_folder in run_folders:
            model_folder = [f.path for f in os.scandir(os.path.join(run_folder,"artifacts")) if f.is_dir() ]
            for model_f in model_folder:
                # print("Found model folder:",model_folder)
                if os.path.isfile(os.path.join(model_f, "input_example.json")):
                    os.remove(os.path.join(model_f, "input_example.json"))
                    print("removed",(os.path.join(model_f, "input_example.json")))
                    i += 1
                else:
                    continue
                if os.path.isfile(os.path.join(model_f, "serving_input_example.json")):
                    os.remove(os.path.join(model_f, "serving_input_example.json"))
                    print("removed",(os.path.join(model_f, "serving_input_example.json")))
                    i += 1
    print("Done.", i, "files removed")

