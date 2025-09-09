import numpy as np
from utils.generic_model import GenericModel
import time


class Solver:
    def __init__(
        self,
        env: GenericModel,
        discount: float,
        epsilon: float = 1e-3,
    ):
        self.env = env
        self.discount = discount
        assert (
            self.discount < 1.0
        ), "Value Iteration shloud be applied to discounted problems."
        self.epsilon = epsilon
        self.name = "personal_value_iteration"

        self.value: np.ndarray = None
        self.policy: np.ndarray = None
        self.runtime: float

    def bellman_operator(self, value):
        q_value = np.zeros((self.env.state_dim, self.env.action_dim))

        for aa in range(self.env.action_dim):
            q_value[:, aa] = (
                self.env.reward_matrix[:, aa]
                + self.discount * self.env.transition_matrix[aa] @ value
            )

        return q_value.max(axis=1)

    def run(self):
        start_time = time.time()
        self.value = np.zeros((self.env.state_dim))

        while True:
            new_value = self.bellman_operator(self.value)
            bellman_residual = np.linalg.norm(new_value - self.value, ord=np.inf)
            if bellman_residual < self.epsilon * (1 - self.discount):
                self.value = new_value
                break
            self.value = new_value

        self.runtime = time.time() - start_time
