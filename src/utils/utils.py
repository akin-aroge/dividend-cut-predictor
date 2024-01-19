""" General utilities for the project."""

import pathlib
import pickle
import nbformat
import configparser
from nbconvert.preprocessors import ExecutePreprocessor


def get_proj_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent.parent


def save_value(value, fname):
    with open(fname, "wb") as f:
        pickle.dump(value, f)


def load_value(fname):
    with open(fname, "rb") as f:
        value = pickle.load(f)
    return value


def run_notebook(notebook_path: pathlib.Path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)

    ep = ExecutePreprocessor(timeout=100000, kernel_name="python3")
    ep.preprocess(nb)

    fname_no_extension = notebook_path.stem
    output_f_path = notebook_path.parent.joinpath(
        f"rough/{fname_no_extension}_exec.ipynb"
    )
    with open(output_f_path, "w", encoding="utf-8") as f:
        nbformat.write(nb=nb, fp=f)

def get_config(config_rel_path:pathlib.Path, interpolation=None):

    proj_root = get_proj_root()

    config = configparser.ConfigParser(interpolation=interpolation)
    config.read(proj_root.joinpath(config_rel_path))   

    return config

def get_full_path(rel_path:pathlib.Path):

    proj_root = get_proj_root()

    full_path = proj_root.joinpath(rel_path)

    return full_path