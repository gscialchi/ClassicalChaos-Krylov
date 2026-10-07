"""
This library provides a decorator that fulfills a similar but simpler
functionality to that of the `doer` library. This is mainly intended to be
used to store frequently used things that may be slow to calculate, like
Hamiltonians, propagators, etc., for large dimensions. In practice, I use it
quite indiscriminately.
"""

import os
import inspect, functools, glob

import numpy as np
import scipy as sp


# duplicating this from doer, ugh
def _get_params_partial(partial_func):
    """
    Get the total arguments that are set from the function itself and from
    its partial.
    """
    f_spec = inspect.getfullargspec(partial_func.func)
    p_spec = inspect.getfullargspec(partial_func)

    p_spec_def = p_spec.defaults
    if p_spec_def is None:
        p_defaults = {}
    else:
        p_defaults = dict(zip(p_spec.args[-len(p_spec_def):], p_spec_def))

    p_spec_kwdef = p_spec.kwonlydefaults
    if p_spec_kwdef is None:
        p_kwdefaults = {}
    else:
        p_kwdefaults = p_spec_kwdef

    # these are not the real defaults, but those defaulted by partial
    f_defaults = dict(zip(f_spec.args[:len(partial_func.args)],
                          partial_func.args))

    p_params = {**f_defaults, **p_defaults, **p_kwdefaults}
    return p_params


def store(path, overwrite_name=False):
    def decorator(func):
        if overwrite_name is False:
            func_name = func.__name__
        else:
            func_name = overwrite_name
        argspec = inspect.getfullargspec(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # generate filename

            if argspec.defaults is not None:
                kwarg_defaults = {**dict(zip(argspec.args[-len(argspec.defaults):],
                                              argspec.defaults))}
            else:
                kwarg_defaults = {}
            params = {**kwarg_defaults, **dict(zip(argspec.args, args)),
                      **kwargs,}
            # important that actual args come after defaults

            for k, v in params.copy().items():
                if isinstance(v, dict):
                    d = params.pop(k)
                    params.update(d)
                if callable(v): # if argument is a function
                    if isinstance(v, functools.partial):
                        params_partial = _get_params_partial(v)
                        params[k] = v.func.__name__ + ' '
                        params[k] = '--' + params[k]
                        params[k] += ' '.join(f"{kk}={vv}" for kk, vv in params_partial.items())
                        params[k] += '--'
                    else:
                        params[k] = v.__name__
            params = dict(params.items())
            params = {k:params[k] for k in sorted(params.keys())} # sort params

            with np.printoptions(legacy='1.25'): # avoid np.float64() in values
                filename = path + f'{func_name} '
                filename += " ".join(f"{k}={v}" for k, v in params.items())

            # handle different saving formats
            if ('sparse' in params.keys()) and (params['sparse']):
                load_func = sp.sparse.load_npz
                save_func = sp.sparse.save_npz
                format = 'npz'
            else:
                load_func = np.load
                save_func = np.save
                format = 'npy'
            filename += f'.{format}'

            # if the file already exists, load it
            files = list(glob.iglob(path+f'/*.{format}'))
            if filename in files:
                out = load_func(filename)
                print(f'Loaded: {filename}')
                return out
            else:
                print(f'Not found: {filename}')

            out = func(*args, **kwargs)

            if not os.path.isdir(path): # make path before saving
                os.makedirs(path)
            save_func(filename, out)
            print(f'Saved: {filename}')
            return out
        return wrapper
    return decorator
