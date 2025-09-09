# Build a list of models, load all solvers_ori... manage files.

import os
import importlib
import platform
import shutil
import importlib
import os
import importlib.util

from utils.generic_model import GenericModel
from typing import List


def reset():
    """
    Function that reset the experience environment.
    """
    for folder in ["saved_models", "saved_value_functions", "results"]:
        try:
            shutil.rmtree(os.path.join(os.getcwd(), folder))
        except FileNotFoundError:
            pass


def load_module(file_path):
    spec = importlib.util.spec_from_file_location("__temp_module__", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_models(folder_path=os.path.join(os.getcwd(), "models")):
    result_dict = {}

    for root, dirs, files in os.walk(folder_path):
        # Exclude "__pycache__" directory
        if "__pycache__" in dirs:
            dirs.remove("__pycache__")

        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)

                try:
                    module = load_module(file_path)
                except Exception as e:
                    print(f"Error loading module from {file_path}: {e}")
                    continue

                # Assuming Model is the class you want to instantiate
                if hasattr(module, "Model") and hasattr(module, "param_list"):
                    model_class = module.Model
                    param_list = module.param_list

                    if not len(param_list):
                        continue

                    # Instantiate Model with each element from param_list
                    model_instances = [model_class(element) for element in param_list]

                    result_dict[file_name.split(".")[0]] = model_instances

    return result_dict


def build_model_list(folder_path: str = os.path.join(os.getcwd(), "models")) -> list:
    models_dict = build_models(folder_path)
    models = [mod for model_list in list(models_dict.values()) for mod in model_list]

    models.sort(key=lambda model: model.state_dim)
    return models


def linux() -> bool:
    """
    To check if the system runs on Linux.
    Output : True if Linux, False otherwise
    """
    return platform.system() == "Linux"


def remove_unused_solver_files(solver_name_list: list) -> list:
    for file_name in ["__pycache__", "_pyMarmoteMDP.so", "pyMarmoteMDP.py"]:
        try:
            solver_name_list.remove(file_name)
        except ValueError:
            pass
    return solver_name_list


def build_solver_list(solver_path: str = os.path.join(os.getcwd(), "solvers")) -> List:
    """
    Build a list of available solvers_ori.
    """
    solver_name_list = os.listdir(solver_path)
    solver_name_list = remove_unused_solver_files(solver_name_list)

    possible_solvers = [
        "marmote",
        "gurobi",
        "slicing",
        "mdptoolbox",
        "personal_implementations",
        "bertsekas",
        "chen",
    ]
    wanted_solvers = possible_solvers

    force_remove_marmote = "marmote" not in wanted_solvers
    force_remove_gurobi = "gurobi" not in wanted_solvers
    force_remove_slicing = "slicing" not in wanted_solvers
    force_remove_mdptoolbox = "mdptoolbox" not in wanted_solvers
    force_remove_personal_implementations = (
        "personal_implementations" not in wanted_solvers
    )
    force_remove_bertsekas = "bertsekas" not in wanted_solvers
    force_remove_chen = "chen" not in wanted_solvers

    if not linux() or force_remove_marmote:
        solver_name_list = [
            solver_name
            for solver_name in solver_name_list
            if "marmote" not in solver_name
        ]

    if force_remove_gurobi:
        solver_name_list = [
            solver_name
            for solver_name in solver_name_list
            if "gurobi" not in solver_name
        ]

    if force_remove_slicing:
        solver_name_list = [
            solver_name
            for solver_name in solver_name_list
            if "slicing" not in solver_name
        ]

    if force_remove_mdptoolbox:
        solver_name_list = [
            solver_name
            for solver_name in solver_name_list
            if "mdptoolbox" not in solver_name
        ]

    if force_remove_personal_implementations:
        solver_name_list = [
            solver_name
            for solver_name in solver_name_list
            if "personal" not in solver_name
        ]

    if force_remove_bertsekas:
        solver_name_list = [
            solver_name
            for solver_name in solver_name_list
            if "bertsekas" not in solver_name
        ]

    if force_remove_chen:
        solver_name_list = [
            solver_name for solver_name in solver_name_list if "chen" not in solver_name
        ]

    # solver_name_list = []
    # solver_name_list.append("value_successive_slicing.py")
    # solver_name_list.append("pi_successive_slicing.py")
    # solver_name_list.append("q_value_successive_slicing.py")
    # solver_name_list.append("personal_policy_iteration.py")
    # solver_name_list.append("personal_value_iteration.py")
    # solver_name_list.append("pi_bertsekas_slicing.py")

    # Remove .py extension
    solver_name_list = [file[:-3] for file in solver_name_list]

    # Import python objects from names
    solver_list = []
    for solver_file in solver_name_list:
        solver_file_import = importlib.import_module("solvers." + solver_file)
        solver_class_from_file = getattr(solver_file_import, "Solver")
        solver_list.append(solver_class_from_file)

    return solver_list
