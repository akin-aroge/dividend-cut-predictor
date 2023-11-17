""" General utilities for the project."""

import pathlib
import pickle
import nbformat
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

def run_notebook(notebook_path:pathlib.Path):

    with open(notebook_path) as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)

    ep = ExecutePreprocessor(timeout=100000, kernel_name='python3')
    ep.preprocess(nb)

    fname_no_extension = notebook_path.stem
    output_f_path = notebook_path.parent.joinpath(f'rough/{fname_no_extension}_exec.ipynb')
    with open(output_f_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb=nb, fp=f)

    # return nb_out

# def run_notebook(notebook_path):
#     execute_notebook(
#         notebook_path,
#         output_path=notebook_path,
#         kernel_name='python3',  # Change the kernel name if necessary
#     )