import os
from utils.pickle_my import pickle_load, pickle_save
import scipy
import numpy as np
from typing import Optional


class GenericModel:
    def __init__(self, params: list = []) -> None:
        # Parsing params
        self.name: str
        self.state_dim: int
        self.action_dim: int
        self.transition_matrix: list
        self.reward_matrix: np.ndarray
        self.params: dict

    def create_model(self):
        # Saving files process
        self.pickle_file_name = "{}.pkl".format(self.name)
        if "saved_models" not in os.listdir(os.getcwd()):
            os.mkdir("saved_models")

        if hasattr(self, "transition_matrix") and hasattr(self, "reward_matrix"):
            return
        elif self.pickle_file_name not in os.listdir("saved_models"):
            self._build_model()
            self._normalize_reward_matrix()
            self._test_model()

            model = (
                self.state_dim,
                self.action_dim,
                self.transition_matrix,
                self.reward_matrix,
            )
            pickle_save(model, os.path.join("saved_models", self.pickle_file_name))
        else:
            (
                self.state_dim,
                self.action_dim,
                self.transition_matrix,
                self.reward_matrix,
            ) = pickle_load(os.path.join("saved_models", self.pickle_file_name))

    def _build_model(self):
        # Define it in the specific model.
        GENERIC_METHOD_USAGE_MESSAGE = "This 'build_model' method is the generic one, build a new one in the specific model."
        assert False, GENERIC_METHOD_USAGE_MESSAGE

    def lighten_model(self):
        """
        Use it to remove heavy matrices during computations.
        """
        try:
            del self.transition_matrix
            del self.reward_matrix
        except AttributeError:
            pass

    def _test_model(self):
        assert self.reward_matrix.shape == (
            self.state_dim,
            self.action_dim,
        ), "The shape of the reward matrix is not correct."

        assert (
            len(self.transition_matrix) == self.action_dim
        ), "The transition matrix does not contain action_dim elements."
        for aa in range(self.action_dim):
            assert self.transition_matrix[aa].shape == (
                self.state_dim,
                self.state_dim,
            ), "The shape of the {}-th transition matrix is not correct".format(aa)

        for aa in range(self.action_dim):
            for ss1 in range(self.state_dim):
                assert (
                    abs(self.transition_matrix[aa][[ss1], :].sum() - 1) < 1e-6
                ), "The transition matrix is not stochastic. Problem with state {}, action {}, transition {}.".format(
                    ss1, aa, self.transition_matrix[aa][[ss1], :].sum()
                )

    def _normalize_reward_matrix(self):
        self.reward_matrix -= np.min(self.reward_matrix)
        self.reward_matrix /= np.max(self.reward_matrix)


class GenericModelGym(GenericModel):
    def __init__(self, env):
        super().__init__(env)
