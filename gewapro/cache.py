import os
import base64
import hashlib
import pandas as pd
from functools import partial
from typing import Any, Callable, Dict, List, Tuple


class Empty:
    """Empty instances are only equal to themselves, smaller than all else and may have no attributes"""
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


encode64 = lambda input: f"{base64.urlsafe_b64encode(hashlib.sha3_512(f'{input}'.encode()).digest())}"[2:-1].rstrip("=")
dict_from_tup = lambda tup: dict(zip([t for i,t in enumerate(tup) if i%2==0],[t for i,t in enumerate(tup) if i%2==1], strict=True))

def _func_string(obj: Dict[str, Any]|tuple):
    if isinstance(obj, tuple):
        obj = dict_from_tup(obj)
    return '('+', '.join([f"{k}='{v}'" if isinstance(v,str) else f"{k}={v}" for k,v in obj.items()])+')'

def _process_kwargs(wrapped: Callable, kwargs: Dict[str, Any], ignore_args: List[str], cache_dir: str|os.PathLike) -> dict:
    """Default ``pre_cache`` function, returns ``{<file_path>: tup}`` dict"""
    tup = tuple([k_or_v for kv_pair in sorted(kwargs.items()) if (not kv_pair[0] in ignore_args) for k_or_v in kv_pair])
    file_path = f"{os.path.join(cache_dir, f'{wrapped.__name__}_{encode64(tup)}')}.parquet"
    return {file_path: kwargs}

def _process_results(wrapped: Callable, kwargs: Dict[str, Any], call_results: List[pd.DataFrame]):
    """Default ``post_cache`` function, returns first list item of `call_results`"""
    return call_results[0]

def cache(cache_dir: str|os.PathLike = "cache", ignore_args: List[str] = [], pre_cache: Callable = _process_kwargs, post_cache: Callable = _process_results):
    """Stores all function calls to the wrapped function in cache, and retrieves if same arguments were used before
    
    - Cached wrapped function must return a ``pd.DataFrame``, otherwise caching will fail
    - Certain function arguments can be ignored for the calls, with a list of argument names passed to ``ignore_args``
    - Whether arguments or keyword arguments are passed does not matter, automatically infers equal function calls
    - Function argument order does not matter, all arguments are sorted alphabetically before being hashed
    - Calls to cache can be split up into multiple calls using the ``pre_cache`` and ``post_cache`` function arguments
    """
    def cache_decorator(func: Callable, cache_dir: str|os.PathLike, ignore_args: List[str], pre_cache: Callable, post_cache: Callable):
        return cache_wrapper(func, cache_dir=cache_dir, ignore_args=ignore_args, pre_cache=pre_cache, post_cache=post_cache)
    return partial(cache_decorator, cache_dir=cache_dir, ignore_args=ignore_args, pre_cache=pre_cache, post_cache=post_cache)

def cache_wrapper(wrapped: Callable, cache_dir: str|os.PathLike, ignore_args: List[str], pre_cache: Callable = _process_kwargs, post_cache: Callable = _process_results):
    """Cache wrapper for functions, it is advised to use the function decorator ``@cache()`` instead"""
    annotations, defaults = wrapped.__annotations__, _get_defaults(wrapped)
    # Check if args provided can be ignored
    for arg in ignore_args:
        if arg not in annotations.keys():
            raise ValueError(f"Got non-existent arg to ignore in ignore_args: {arg}")
    # Create cache folder if it does not already exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Created cache directory: {os.path.join(cache_dir,"strip")[:-5]}")

    # Create the cached function
    def cached_func(*args, **kwargs):
        new_kwargs = _check_annotations_get_kwargs(annotations, defaults, *args, **kwargs)
        if isinstance(new_kwargs, dict):
            # Split the kwargs up into multiple cache checks within a dict
            file_paths_dict: dict = pre_cache(wrapped=wrapped, kwargs=new_kwargs, ignore_args=ignore_args, cache_dir=cache_dir)
            if not file_paths_dict:
                raise ValueError(f"Tuple(s) could not be generated within pre-cache function '{pre_cache.__name__}' from initial kwargs {new_kwargs}")
            call_results = []
            # For each file and kwargs, get from cache or perform function call
            for file_path,kwargs_ in file_paths_dict.items():
                call_results.append(_call_or_get_from_cache(wrapped=wrapped, kwargs=kwargs_, file_path=file_path))
            # Return the processed post-cache call_results
            return post_cache(wrapped=wrapped, kwargs=new_kwargs, call_results=call_results)
        raise ValueError(f"Tuple could not be generated from arguments {args} and keyword arguments {kwargs}: {new_kwargs}")
    return cached_func

def _call_or_get_from_cache(wrapped: Callable, kwargs: dict, file_path: str):
    """Checks if file for current kwargs already exists, otherwise calls `wrapped` with kwargs and saves to `file_path`"""
    if os.path.exists(file_path):
        print(f"Found file in cache for call {wrapped.__name__}{_func_string(kwargs)}: {file_path}")
        return pd.read_parquet(file_path)
    print(f"No cache for call {wrapped.__name__}{_func_string(kwargs)}, running function...")
    save_df = wrapped(**kwargs)
    if isinstance(save_df, pd.DataFrame):
        save_df.to_parquet(file_path)
    else:
        raise TypeError(f"Function {wrapped.__name__} did not output 'pd.DataFrame', but '{save_df.__class__.__name__}'")
    print(f"Saved to file: {file_path}")
    return save_df

def _check_annotations_get_kwargs(annotations: dict, defaults: dict, *args, **kwargs) -> dict|str:
    """Takes function and passed arguments and creates a kwargs dict with alphabetically ordered key-value pairs e.g. ``{"arg1":12,"Barg1":None,"Carg1":True}``"""
    annotations.pop("return", None)
    invalid_args_kwargs = False  # Keep track of invalid args & kwargs
    for kwarg_key in kwargs.keys():
        if not kwarg_key in annotations.keys():
            invalid_args_kwargs = 1  # Invalid kwarg name
    if len(args)+len(defaults | kwargs) != len(annotations.keys()):
        invalid_args_kwargs += 2  # Invalid arg, kwarg total length
    # Fill annotations with defaults, then update with kwargs
    filled_annotations = ({k:Empty() for k in annotations.keys()} | defaults) | kwargs
    extended_args = list(args) + [Empty()]*len(annotations)
    # Get dict of keyword arguments that are empty, fill with positional arguments
    fill_empty_kwargs_with_args = {k:extended_args[i] for i,(k,v) in enumerate(filled_annotations.items()) if v == Empty()}
    # Get list of keyword arguments that are non-empty
    filled_kwargs = [v for v in filled_annotations.values() if v != Empty()]
    if not [Empty()]*len(fill_empty_kwargs_with_args)+filled_kwargs == list(filled_annotations.values()):
        invalid_args_kwargs += 4  # Invalid arg, kwarg order
    # Return a detailed output on what failed if invalid args or kwargs were passed
    if invalid_args_kwargs:
        return f"Input failed with fail code sum {invalid_args_kwargs} (1: unknown kwarg, 2: too few/many args/kwargs, 4: invalid arg-kwarg order)"
    # Return filled annotations, where empty kwargs are filled with args
    return filled_annotations | fill_empty_kwargs_with_args

def _get_defaults(func: Callable) -> Dict[str, Any]:
    """Gets dict of the default values of all `func` arguments"""
    co_varnames = func.__code__.co_varnames[:func.__code__.co_argcount]
    if func.__defaults__ is None:
        return func.__kwdefaults__ or {}
    return dict(zip(co_varnames[-len(func.__defaults__):],func.__defaults__)) | (func.__kwdefaults__ or {})

