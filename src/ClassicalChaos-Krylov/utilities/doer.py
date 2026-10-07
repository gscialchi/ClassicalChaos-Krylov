"""
This library provides a class that allows for automatically checking whether a
given calculation has already been made and its results stored. If it has, then
simply load the data. If it hasn't, proceed to calculate and store.

The basic idea is to keep track of every method and argument that yields a
final result. The information about the whole pipeline is contained in an
`infostring`.

Files are stored named with a hash code. A look-up table is used to link a
given calculation (infostring) to the correct hash.
"""

import os
import json
import inspect
import hashlib
from copy import deepcopy
import functools

import numpy as np


def get_compatible_infostring(doer, dic):
    """
    Looks up infostrings in dic with everything the same except the value of
    iterable. Return the one with the largest value of iterable, and
    whether it is larger than the one sought.
    """
    infostring = doer.get_infostring()
    iterable = doer.iterable

    ifsplit1 = infostring.split(f' {iterable}=')
    ifsplit2 = ifsplit1[-1].split(' ')

    left = ifsplit1[0]
    iterable_value = float(ifsplit2[0])
    right = ' '.join(ifsplit2[1:])

    found = False
    strings = []
    values = []
    for string in dic.keys():
        dic_ifsplit1 = string.split(f' {iterable}=')
        dic_ifsplit2 = dic_ifsplit1[-1].split(' ')
        dic_left = dic_ifsplit1[0]
        dic_right = ' '.join(dic_ifsplit2[1:])

        if (left == dic_left) and (right == dic_right):
            value = float(dic_ifsplit2[0])
            strings.append(string)
            values.append(value)
    if len(strings) == 0:
        return found, None
    else:
        if np.max(values) >= iterable_value:
            found = True
        else:
            found = False
        return found, strings[np.argmax(values)]


def get_hash(doer, infostring):
    try:
        with open(doer.path+'table.json', 'r') as table:
            dic = json.load(table)
        hash = dic[infostring]
    except (FileNotFoundError, KeyError):
        hash = None
    return hash


def search_data(doer):
    try:
        with open(doer.path+'table.json', 'r') as table:
            dic = json.load(table)

        if not doer.iterable is None:
            found, infostring = get_compatible_infostring(doer, dic)
        else:
            found, infostring = True, doer.get_infostring()

        hash = get_hash(doer, infostring)
        if hash is None:
            found = False

    except FileNotFoundError:
        found = False
        infostring = None
        hash = None
    return found, infostring, hash


def load_data(doer, found_infostring, found_hash):
    filename = f'{doer.path}{found_hash}.npy'

    data_memmap = np.load(filename, mmap_mode='r')
    data_shape = data_memmap.shape

    if not doer.slice_return == slice(None):
        data = data_memmap[doer.slice_return] # load only what will be returned
    else:
        data = data_memmap
    del data_memmap

    slices = [slice(None)]*len(data.shape)
    if not doer.iterable is None:
        slices[doer.iterable_index] = slice(0, doer.args[doer.iterable]+1)
    data = data[tuple(slices)]
    return data


def delete_entry(doer, entry):
    try:
        with open(doer.path+'table.json', 'r+') as table:
            dic = json.load(table)
            del dic[entry]
        with open(doer.path+'table.json', 'w') as table:
            json.dump(dic, table, indent=4, separators=(', ', ': '))
            print(f'Deleted entry from table: {entry}')
    except (FileNotFoundError, KeyError):
        ...


def delete_data(doer, hash):
    filename = f'{doer.path}{hash}.npy'
    try:
        os.remove(filename)
        print(f'Removed data: {filename}')
    except FileNotFoundError:
        ...


def save_data(doer, data, infostring=None):
    if not doer.slice_save == slice(None):
        data = data[doer.slice_save]

    if infostring is None:
        infostring = doer.get_infostring()

    hash = hashlib.sha256(infostring.encode('UTF-8')).hexdigest()

    filename = f'{doer.path}{hash}.npy'
    if not os.path.isdir(doer.path): # make path before saving
        os.makedirs(doer.path)
    np.save(filename, data)

    try:
        with open(doer.path+'table.json', 'r+') as table:
            dic = json.load(table)
            dic[infostring] = hash
            table.seek(0)
            json.dump(dic, table, indent=4, separators=(', ', ': '))
    except FileNotFoundError:
        with open(doer.path+'table.json', 'w') as table:
            dic = {}
            dic[infostring] = hash
            json.dump(dic, table, indent=4, separators=(', ', ': '))
    print(f'Data saved: {infostring}')


def replace_data(doer, data, new_entry, old_entry=None, old_hash=None):
    if old_entry is None:
        old_entry = new_entry
    if old_hash is None:
        old_hash = get_hash(doer, old_entry)
    delete_entry(doer, old_entry)
    delete_data(doer, old_hash)
    save_data(doer, data, new_entry)


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


class Doer:
    def __init__(self, func, alias=None,
                 load=True, save=True, replace=False,
                 args=None, fake_args=None, ignore_args=None,
                 iterable=None, iterable_index=None,
                 iterable_continue=True, continue_arg=None,
                 slice_save=slice(None), slice_return=slice(None),
                 except_if_not_found=False, provides=None,
                 path=None, disabled=False):
        if args is None:
            args = {}
        if fake_args is None:
            fake_args = {}
        if ignore_args is None:
            ignore_args = []
        elif not isinstance(ignore_args, list):
            ignore_args = [ignore_args]
        if provides is None:
            provides = []
        elif not isinstance(provides, list):
            provides = [provides]

        self.func = func
        if hasattr(func, '__wrapped__'):
            # argspec is lost when function is wrapped
            self.spec = inspect.getfullargspec(getattr(func, '__wrapped__'))
        else:
            self.spec = inspect.getfullargspec(func)
        self.name = func.__name__
        if alias is None:
            self.alias = self.name
        else:
            self.alias = alias

        self.load = load
        self.save = save
        self.replace = replace

        self.args = args
        self.fake_args = fake_args # are not passed, only appear in infostring
        self.ignore_args = ignore_args # are passed, don't appear in infostring

        self.slice_save = slice_save # apply slice before saving data
        self.slice_return = slice_return # apply slice before returning data

        self.path = path

        self.iterable = iterable # func parameter over which is iterated
        self.iterable_index = iterable_index # index in output's shape that corresponds to iterable
        self.iterable_continue = iterable_continue # continue iteration from stored data if available
        self.continue_arg = continue_arg # which arg to put the loaded data into

        self.except_if_not_found = except_if_not_found
        self.provides = provides

        self.disabled = disabled

    def set_args(self, dic={}, **kwargs):
        self.args = {**self.args, **dic, **kwargs}

    def set_fakeargs(self, **kwargs):
        self.fake_args = {**self.fake_args, **kwargs}

    def set_ignoreargs(self, args):
        if isinstance(args, list):
            self.ignore_args += args
        else:
            self.ignore_args += [args]

    def get_args(self):
        argspec = self.spec

        if argspec.defaults is not None:
            kwarg_defaults = {**dict(zip(argspec.args[-len(argspec.defaults):],
                                          argspec.defaults))}
        else:
            kwarg_defaults = {}
        params = {**kwarg_defaults, **self.args, **self.fake_args}
        # NOTE: important that actual args come after defaults
        return params

    def get_infostring(self):
        params = self.get_args()

        for p in self.ignore_args:
            params.pop(p)

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
            if isinstance(v, Doer): # if argument is a Doer
                params[k] = '--' + v.get_infostring() + '--'
        params = dict(params.items())
        params = {k:params[k] for k in sorted(params.keys())} # sort params

        with np.printoptions(legacy='1.25'): # avoid np.float64() in values
            infostring = f'{self.alias} '
            infostring += " ".join(f"{k}={v}" for k, v in params.items())
        return infostring

    def getit(self):
        return self.func(**self.args)

    def _doit(self):
        doit_args = self.args.copy()
        for k, v in doit_args.items():
            if isinstance(v, Doer):
                #  doit_args[k] = v.getit()
                doit_args[k] = v._doit()
        return self.func(**doit_args)

    def doit(self):
        if not self.disabled:
            wanted_infostring = self.get_infostring()
            if self.load:
                found, compat_infostring, compat_hash = search_data(self)
                no_compat_found = (compat_infostring is None) or (compat_hash is None)

                if found:
                    print(f'Data found: {compat_infostring}')
                    data = load_data(self, compat_infostring, compat_hash)
                elif self.except_if_not_found and no_compat_found:
                    raise FileNotFoundError(f'Data NOT found: {wanted_infostring}')
                else:
                    print(f'Data NOT found: {wanted_infostring}')

                    if no_compat_found or (not self.iterable_continue):
                        data = self._doit()
                    elif not self.continue_arg is None:
                        print(f'Continue from: {compat_infostring}')
                        compat_data = load_data(self, compat_infostring, compat_hash)
                        self.set_args(dic={self.continue_arg:compat_data})
                        self.set_ignoreargs(self.continue_arg)
                        data = self._doit()
                    else:
                        raise Exception(f"Cannot do continue: `continue_arg` not provided.")

                    if self.save:
                        if no_compat_found:
                            save_data(self, data, wanted_infostring)
                        elif not self.iterable is None:
                            replace_data(self, data, wanted_infostring,
                                         compat_infostring, compat_hash)

                if not self.slice_return == slice(None):
                    data = data[self.slice_return]
            else:
                data = self._doit()
                already_exists, _, _ = search_data(self)
                if self.save and already_exists and self.replace:
                    replace_data(self, data, wanted_infostring)
                elif self.save and already_exists and (not self.replace):
                    print(f'Not saving (replace is set to False): {wanted_infostring}')
                elif self.save and not already_exists:
                    save_data(self, data, wanted_infostring)
                if not self.slice_return == slice(None):
                    data = data[self.slice_return]
            return data
        else:
            return self._doit()

    def copy(self):
        return deepcopy(self)
