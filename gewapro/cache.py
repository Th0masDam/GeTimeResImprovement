import os
import base64
import hashlib
import inspect
import pandas as pd
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, NamedTuple
import re
from gewapro.util import add_notes

class CacheError(ValueError):
    """Error related to the cache system"""
    def __init__(self, *args):
        super().__init__(*args)

class CacheSetupError(ValueError):
    """Error related to the cache system"""
    def __init__(self, *args):
        super().__init__(*args)

class Empty:
    """Empty instances are only equal to themselves, smaller than all else and may not have attributes"""
    def __setattr__(self, name: str, value: Any) -> None:
        raise ValueError("Empty object may not have attributes")
    def __repr__(self):
        return "Empty"
    def __eq__(self, other):
        if isinstance(other, Empty):
            return True
        return False
    def __lt__(self, other):
        return not self.__eq__(other)
    def __gt__(self, other):
        return False
    def __le__(self, other):
        return True
    def __ge__(self, other):
        return self.__eq__(other)

class CacheInfo(NamedTuple):
    function: Callable
    cache_dir: str
    ignore_args: list
    pre_cache: Callable
    post_cache: Callable
    cache_size: int
    "in kB"
    cached_files: list
    
    def __repr__(self) -> str:
        funcname = self.function.__qualname__
        return f"""CacheInfo for "{funcname}"
=============={'='*len(funcname)}==
Cache directory: {self.cache_dir}
Ignored arguments: {self.ignore_args}
Pre-cache function: {self.pre_cache}
Post-cache function: {self.post_cache}
Cached files ({f'{self.cache_size:.0f} kB' if self.cache_size < 1024 else f'{self.cache_size/1024:.1f} MB'}):
   {f'\n   '.join(self.cached_files) if self.cached_files else '<No files in cache>'}
"""

encode64 = lambda input: f"{base64.urlsafe_b64encode(hashlib.sha3_512(f'{input}'.encode()).digest())}"[2:-1].rstrip("=")
dict_from_tup = lambda tup: dict(zip([t for i,t in enumerate(tup) if i%2==0],[t for i,t in enumerate(tup) if i%2==1], strict=True))

def _func_string(obj: Dict[str, Any]|tuple, limit: int=100):
    if isinstance(obj, tuple):
        obj = dict_from_tup(obj)
    string = '('+', '.join([f"{k}='{v}'" if isinstance(v,str) else f"{k}={v}" for k,v in obj.items()])+')'
    if len(string) > limit:
        return string[:round(limit/3)]+" ... "+string[-round(limit/3):].replace("\n","")
    return string.replace("\n","")

def _process_kwargs(wrapped: Callable, kwargs: Dict[str, Any], ignore_args: List[str], cache_dir: str|os.PathLike) -> dict:
    """Default ``pre_cache`` function, returns ``{<file_path>: <kwargs>}`` dict"""
    tup = tuple([k_or_v for kv_pair in sorted(kwargs.items()) if (not kv_pair[0] in ignore_args) for k_or_v in kv_pair])
    file_path = f"{os.path.join(cache_dir, f'{wrapped.__name__}_{encode64(tup)}')}.parquet"
    return {file_path: kwargs}

def _process_results(wrapped: Callable, kwargs: Dict[str, Any], call_results: List[pd.DataFrame]):
    """Default ``post_cache`` function, returns first list item of `call_results`"""
    return call_results[0]

def _clear_cache(function_to_clear: Callable, cache_dir: str|os.PathLike, verbose: bool = False) -> None:
    def clear_cache(verbose: bool = False):
        """Clears all cache for the function from `cache_dir` directory"""
        if not os.path.exists(cache_dir):
            raise CacheError(f"Cache directory '{cache_dir}' does not exist, no cache to clear")
        file_path_regex = re.compile(f"{function_to_clear.__name__}_.*\\.parquet$")
        for _, _, files in os.walk(cache_dir):
            for file in files:
                if file_path_regex.match(file):
                    try:
                        os.remove(os.path.join(cache_dir,file))
                        if verbose:
                            print(f"Failed to remove file '{file}' from cache directory: {err}")
                    except Exception as err:
                        if verbose:
                            print(f"Remove file '{file}' from cache directory '{cache_dir}'")
    return clear_cache


def cache_info(function_to_inspect: Callable, cache_dir: str|os.PathLike, ignore_args: list, pre_cache: Callable, post_cache: Callable) -> CacheInfo:
    """Returns info on the cached function's cache settings & cache size"""
    file_path_regex = re.compile(f"{function_to_inspect.__name__}_.*\\.parquet$")
    matched_files = []
    for _, _, files in os.walk(cache_dir):
        for file in files:
            if file_path_regex.match(file):
                matched_files.append((file, os.path.getsize(os.path.join(cache_dir,file))/1024))
    cache_size = round(sum([file[1] for file in matched_files]))
    matched_files = [file[0]+(f" ({file[1]:.0f} kB)" if file[1]<1024 else f" ({file[1]/1024:.1f} MB)") for file in matched_files]
    return CacheInfo(function_to_inspect, cache_dir, ignore_args, pre_cache, post_cache, cache_size, matched_files)


def cache(cache_dir: str|os.PathLike = "cache", ignore_args: List[str] = [], pre_cache: Callable = _process_kwargs, post_cache: Callable = _process_results):
    """Stores all function calls to the wrapped function in cache, and retrieves if same arguments were used before
    
    - Cached wrapped function must return a ``pd.DataFrame``, otherwise caching will fail
    - Certain function arguments can be ignored for the calls, with a list of argument names passed to ``ignore_args``
    - Whether arguments or keyword arguments are passed does not matter, automatically infers equal function calls
    - Function argument order does not matter, all arguments are sorted alphabetically before being hashed
    - Calls to cache can be split up into multiple calls using the ``pre_cache`` and ``post_cache`` function arguments

    NOTE: Does not work for functions that have variable args/kwargs. Also, using '/' within a function signature can
    cause erroneous cache calls
    """
    def cache_decorator(func: Callable, cache_dir: str|os.PathLike, ignore_args: List[str], pre_cache: Callable, post_cache: Callable):
        return cache_wrapper(func, cache_dir=cache_dir, ignore_args=ignore_args, pre_cache=pre_cache, post_cache=post_cache)
    return partial(cache_decorator, cache_dir=cache_dir, ignore_args=ignore_args, pre_cache=pre_cache, post_cache=post_cache)

def cache_wrapper(wrapped: Callable, cache_dir: str|os.PathLike, ignore_args: List[str], pre_cache: Callable = _process_kwargs, post_cache: Callable = _process_results):
    """Cache wrapper for functions, it is advised to use the function decorator ``@cache()`` instead"""
    # Check if all args are valid and get annotations with defaults
    _check_validity(wrapped)
    annotations, defaults = _get_annotations(wrapped), _get_defaults(wrapped)
    # Check if args provided can be ignored
    for arg in ignore_args:
        if arg not in annotations.keys():
            raise CacheSetupError(f"Got non-existent arg to ignore in ignore_args: {arg}")
    # Create cache folder if it does not already exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"[CACHE WRAPPER] Created cache directory: {os.path.join(cache_dir,"strip")[:-5]}")

    # Create the cached function
    def cached_func(*args, **kwargs):
        new_kwargs = _check_annotations_get_kwargs(annotations, defaults, *args, **kwargs)
        if isinstance(new_kwargs, dict):
            # Split the kwargs up into multiple cache checks within a dict
            file_paths_dict: dict = pre_cache(wrapped=wrapped, kwargs=new_kwargs, ignore_args=ignore_args, cache_dir=cache_dir)
            if not file_paths_dict:
                raise CacheError(f"Tuple(s) could not be generated within pre-cache function '{pre_cache.__name__}' from initial kwargs {new_kwargs}")
            call_results = []
            # For each file and kwargs, get from cache or perform function call
            for file_path,kwargs_ in file_paths_dict.items():
                call_results.append(_call_or_get_from_cache(wrapped=wrapped, kwargs=kwargs_, file_path=file_path))
            # Return the processed post-cache call_results
            return post_cache(wrapped=wrapped, kwargs=new_kwargs, call_results=call_results)
        raise add_notes(CacheError(f"Tuple could not be generated from arguments and keyword arguments: {new_kwargs}"))
    
    # Add function attributes
    cached_func.cache_info = cache_info(function_to_inspect=wrapped, cache_dir=cache_dir, ignore_args=ignore_args, pre_cache=pre_cache, post_cache=post_cache)
    """Returns info on the cached function's cache settings & cache size"""
    cached_func.clear_cache = _clear_cache(function_to_clear=wrapped, cache_dir=cache_dir)
    """Clear the cache of this function"""
    cached_func.__name__ = wrapped.__name__
    cached_func.__qualname__ = cached_func.__qualname__.replace("cached_func",wrapped.__qualname__)
    return cached_func

def _call_or_get_from_cache(wrapped: Callable, kwargs: dict, file_path: str):
    """Checks if file for current kwargs already exists, otherwise calls `wrapped` with kwargs and saves to `file_path`"""
    if os.path.exists(file_path):
        print(f"[CACHE WRAPPER] Found file in cache for call {wrapped.__name__}{_func_string(kwargs)}: {file_path}")
        return pd.read_parquet(file_path)
    print(f"[CACHE WRAPPER] No cache for call {wrapped.__name__}{_func_string(kwargs)}, running function...")
    save_df = wrapped(**kwargs)
    if isinstance(save_df, pd.DataFrame):
        save_df.to_parquet(file_path)
    else:
        raise TypeError(f"Function {wrapped.__name__} did not output 'pd.DataFrame', but '{save_df.__class__.__name__}'")
    print(f"[CACHE WRAPPER] Saved to file: {file_path}")
    return save_df

def _check_annotations_get_kwargs(annotations: dict, defaults: dict, *args, **kwargs) -> dict|tuple[str,str]:
    """Takes function and passed arguments and creates a kwargs dict with alphabetically ordered key-value pairs e.g. ``{"arg1":12,"Barg1":None,"Carg1":True}``"""
    note = "" # Keep track of invalid args & kwargs
    for kwarg_key in kwargs.keys():
        if not kwarg_key in annotations.keys():
            note += f"Got invalid kwarg: \'{kwarg_key}\'. "
    # Get dict of keyword arguments that are empty, fill with positional arguments
    fill_empty_kwargs_with_args = {k:v for k,v in list(zip(annotations, list(args) + [Empty()]*len(annotations))) if v != Empty()}
    # Fail if kwargs are named that are already filled by positional arguments
    if overwritten := [k for k in kwargs.keys() if k in fill_empty_kwargs_with_args.keys()]:
        note += f"Args overwritten by kwargs: \'{'\', \''.join(overwritten)}\'. "
    # Fill defaults with args, then update with kwargs
    filled_annotations = (defaults | fill_empty_kwargs_with_args) | kwargs
    # Fail if too few/many args are given
    if len(filled_annotations) != len(annotations) or len(args)+len(kwargs) > len(annotations):
        note += f"Expected {len(annotations)} (keyword) arguments, but got {len(args)+len(kwargs)} ({len(args)} args + {len(defaults | kwargs)} kwargs). "
    # Return a detailed output on what failed if invalid args or kwargs were passed
    return note if note else filled_annotations

def _get_defaults(func: Callable) -> dict[str, Any]:
    """Gets dict of the default values of all `func` arguments that have a default"""
    co_varnames = func.__code__.co_varnames[:func.__code__.co_argcount]
    if func.__defaults__ is None:
        return func.__kwdefaults__ or {}
    return dict(zip(co_varnames[-len(func.__defaults__):],func.__defaults__)) | (func.__kwdefaults__ or {})

def _get_annotations(func: Callable) -> dict:
    """Gets dict of annotations of all `func` arguments, even if not annotated"""
    valid_varnames = func.__code__.co_varnames[:func.__code__.co_argcount+func.__code__.co_kwonlyargcount]
    inspected_annotations: dict = {a:Empty for a in valid_varnames} | func.__annotations__
    inspected_annotations.pop("return",None)
    return inspected_annotations

def _check_validity(func: Callable) -> None:
    """Checks if there are no pos-only arguments or variable (keyword) argument(s) in ``func``"""
    if not [None,None] == (varargs := [inspect.getfullargspec(func)[1],inspect.getfullargspec(func)[2]]):
        raise CacheSetupError(f"Got variable (keyword) argument(s) in function '{func.__name__}': '{'\' & \''.join([v for v in varargs if v is not None])}\'")
    elif func.__code__.co_posonlyargcount > 0:
        raise CacheSetupError("Function signature containing '/' not supported")