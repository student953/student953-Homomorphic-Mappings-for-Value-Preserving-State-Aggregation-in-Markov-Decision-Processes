# Generate the exact solution of a given model.

from mdptoolbox.mdp import (
    RelativeValueIteration,
    ValueIteration,
    PolicyIterationModified,
)
from utils.generic_model import GenericModel
from utils.pickle_my import pickle_load, pickle_save
from typing import Optional, Callable
import numpy as np
import os


av_str, dis_str, tot_str = "average", "discounted", "total"
all_method_set = {av_str, dis_str, tot_str}


def build_value_function_name(
    model: GenericModel, method: str, discount: float = 1.0
) -> str:
    """
    The name of the file we are going to use to save the optimal value function.
    """

    assert method in all_method_set, "Method {} not recognized.".format(method)
    return "{}_{}_{}".format(method, discount, model.name)


def compute_exact_value(
    model: GenericModel,
    method: str,
    discount: Optional[float] = None,
    warning: bool = True,
    epsilon_exact_value: float = 1e-6,
    max_iter_exact_value: int = int(1e8),
) -> np.ndarray:
    """
    Given a model, we compute the exact value function
    up to a given precision.
    """
    assert (
        not warning
    ), "The function should not be used in itself, use get_exact_value function instead."

    if method == av_str:
        solver = RelativeValueIteration(
            transitions=model.transition_matrix,
            reward=model.reward_matrix,
            epsilon=epsilon_exact_value,
            max_iter=max_iter_exact_value,
        )
    elif method == dis_str:
        solver = ValueIteration(
            transitions=model.transition_matrix,
            reward=model.reward_matrix,
            discount=discount,
            epsilon=epsilon_exact_value,
            max_iter=max_iter_exact_value,
        )
    elif method == tot_str:
        solver = ValueIteration(
            transitions=model.transition_matrix,
            reward=model.reward_matrix,
            discount=1.0,
            epsilon=epsilon_exact_value,
            max_iter=max_iter_exact_value,
        )
    solver.run()
    return np.array(solver.V)


def check_model_method_discount(model: GenericModel, method: str, discount: float):
    """
    Check if given model, method and discount are logical inputs.
    """
    assert 0 < discount <= 1.0, "The discount should be between 0 and 1."
    assert hasattr(model, "transition_matrix") and hasattr(
        model, "reward_matrix"
    ), "Model should have been created using model.create_model()."

    if method == tot_str or method == av_str:
        assert (
            abs(discount - 1.0) < 1e-9
        ), "The discount should be equal to 1 in the total or average case."
    elif method == dis_str:
        assert discount < 1.0, "The discount should be smaller than 1."


def get_exact_value(
    model: GenericModel, criterion: str, discount: float = 1.0
) -> np.ndarray:
    """
    For a given model, either we fetch the pickle
    saved optimal value or we compute it.
    Parameters :
    criterion : total, discounted or average
    """
    check_model_method_discount(model, criterion, discount)

    saving_folder = os.path.join(os.getcwd(), "saved_value_functions")
    try:
        os.mkdir(saving_folder)
    except FileExistsError:
        pass

    exact_value_function_pickle_file = "{}.pkl".format(
        build_value_function_name(model, criterion, discount)
    )

    solving_function = lambda: compute_exact_value(
        model, criterion, discount, warning=False
    )

    return get_saved_object(
        saving_folder, solving_function, exact_value_function_pickle_file
    )


def get_saved_object(
    folder_path: str, function_to_compute_it: Callable, file_name: str
):
    """
    If file_name already exists, return the pickle load of it.
    Else, compute a result with function_to_compute_it and save it using pickle.
    """
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        pass

    if ".pkl" not in file_name:
        file_name += ".pkl"

    file_full_path = os.path.join(folder_path, file_name)

    if os.path.isfile(file_full_path):
        return pickle_load(file_full_path)
    else:
        obj_to_save = function_to_compute_it()
        pickle_save(obj_to_save, file_full_path)
        return obj_to_save
