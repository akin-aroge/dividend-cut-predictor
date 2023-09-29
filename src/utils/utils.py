""" General utilities for the project."""

import pathlib
import pickle

def get_proj_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent.parent

def save_value(value, fname):
    with open(fname, "wb") as f:
        pickle.dump(value, f)


def load_value(fname):
    with open(fname, "rb") as f:
        value = pickle.load(f)
    return value