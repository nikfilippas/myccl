import numpy as np


def _remove_char(str, chars=[]):
    # Given an input string, remove all characters in the `chars` list.
    for char in chars:
        str = str.replace(char, "")
    return str


def _get_array_info(str):
    # Return the array name, array dimensions, and array data.
    name_dim, numbers = str.split("=")                 # split at equal sign
    data = [float(num) for num in numbers.split(",")]  # convert str to numbers
    name, *dims = name_dim.split("[")                  # extract variable name
    dims = [int(dim.strip("]")) for dim in dims]       # extract ndim
    return name, dims, data


def _read_emu_data():
    with open("emu22_params.h") as f:
        readfile = f.read()

    pars = readfile.split("static double")[1:]
    pars = [_remove_char(par, ["\n", "{", "}", ";", " "]) for par in pars]
    raw_arrays = [_get_array_info(par) for par in pars]

    for name, dims, data in raw_arrays:
        data = np.asarray(data).reshape(dims)
        global _emu_data
        _emu_data[name] = data


def get_emu_data():
    # Run only at import. Load the emulator data.
    if not _emu_data:
        _read_emu_data()
    return _emu_data.copy()


_emu_data = {}
