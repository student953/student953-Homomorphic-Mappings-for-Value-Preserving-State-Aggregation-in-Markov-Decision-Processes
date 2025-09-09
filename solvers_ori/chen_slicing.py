"""
This code allows to slice a state space progressively 
along an evolving value function.
"""

import numpy as np
from time import time
from utils.generic_model import GenericModel
from utils.value_partition_class import ValuePartition
from utils.calculus import optimal_bellman_operator, norminf


class Solver:
    def __init__(
            self,
            model: GenericModel,
            discount: float,
            epsilon: float = 1e-3,  # Final Precision
            verbose: bool = False,
            max_iter_loop: int = 1000,
            iter_agg: int = 1,
            iter_bellman: int = 100,
            lr: float = 0.1,
            max_iter_evaluation: int = int(1e3),
            max_iter_policy_update: int = int(1e4),
    ):
        # Class arguments
        self.model = model
        self.discount = discount
        self.epsilon = epsilon
        self.name = "chen_slicing"
        self.verbose = verbose
        self.lr = lr
        self.max_iter_evaluation = max_iter_evaluation
        self.max_iter_policy_update = max_iter_policy_update

        self.max_iter_loop, self.iter_agg, self.iter_bellman = (
            max_iter_loop,
            iter_agg,
            iter_bellman,
        )

        # Variables
        self.partition = ValuePartition(self.model, self.discount)
        self.contracted_value: np.ndarray

        self.alpha_function = lambda t: 1 / (t + 1) ** 2

    def run(self):
        start_time = time()
        self.value = np.zeros((self.model.state_dim))

        n = 0

        for i_episode in range(self.max_iter_policy_update):
            n += 1
            for _ in range(self.iter_bellman):
                self.new_value = optimal_bellman_operator(
                    self.model, self.value, self.discount
                )
                if norminf(self.new_value - self.value) < self.epsilon * (
                        1 - self.discount
                ):
                    self.runtime = time() - start_time
                    return
                self.value = self.new_value

            self.partition = ValuePartition(self.model, self.discount)
            self.partition.divide_regions_along_tv(
                self.value,
                self.epsilon,
                np.zeros((len(self.partition.states_in_region))),
            )
            self.partition._build_weights()
            self.contracted_value = self.partition.weights @ self.value
            alpha = self.alpha_function(n * (self.iter_agg + self.iter_bellman))

            for _ in range(self.iter_agg):
                self.extended_value = (
                        self.partition._partial_phi() @ self.contracted_value
                )
                for k in range(len(self.partition.states_in_region)):
                    ss = np.random.choice(self.partition.states_in_region[k])
                    tsv_action = np.zeros((self.model.action_dim))
                    for aa in range(self.model.action_dim):
                        tsv_action[aa] = (
                                self.model.reward_matrix[ss, aa]
                                + self.discount
                                * self.model.transition_matrix[aa].getrow(ss)
                                @ self.extended_value
                        )
                    self.contracted_value[k] = (1 - alpha) * self.contracted_value[
                        k
                    ] + alpha * max(tsv_action)

            self.value = self.partition._partial_phi().dot(self.contracted_value)
            if (norminf(self.value - optimal_bellman_operator(self.model, self.value, self.discount))
                    > self.epsilon * (1 - self.discount)):
                break

        assert False, "Finished without convergence."
