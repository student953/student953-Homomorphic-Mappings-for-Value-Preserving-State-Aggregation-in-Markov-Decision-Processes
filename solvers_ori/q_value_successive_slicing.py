"""
Implementation of Q-Value Iteration. 
Here, the Q-value is updated by blocks 
following an iteratively refined partition.
"""

import numpy as np
from utils.generic_model import GenericModel
from time import time
from utils.generic_model import GenericModel
from utils.q_value_partition_class import QValuePartition
from utils.calculus import norminf, q_optimal_bellman_operator
from utils.calculus_projected import apply_poqbo_until_var_small


class Solver:
    def __init__(
        self,
        env: GenericModel,
        discount: float,
        epsilon: float = 1e-3,  # Final Precision
    ):
        # Class arguments
        self.env = env
        self.discount = discount
        self.epsilon = epsilon
        self.name = "q_value_successive_slicing"

        # Variables
        self.partition = QValuePartition(self.env, self.discount)
        self.contracted_q_value: np.ndarray = None

        self.epsilon_pbr = self.epsilon * (1 - self.discount) / 2
        self.epsilon_span = self.epsilon * (1 - self.discount) / 2

    def run(self):
        start_time = time()
        self.contracted_q_value = np.zeros(
            (len(self.partition.states_in_region), self.env.action_dim)
        )

        while True:
            self.partition._compute_aggregate_transition_reward()

            self.contracted_q_value, pbr_value = apply_poqbo_until_var_small(
                self.env,
                self.discount,
                self.partition.aggregate_transition_matrix,
                self.partition.aggregate_reward_matrix,
                self.epsilon_pbr,
                self.contracted_q_value,
            )

            q_value = self.partition._partial_phi().dot(self.contracted_q_value)

            bellman_of_q_value = q_optimal_bellman_operator(
                self.env, q_value, self.discount
            )
            maximum_span = np.max(
                self._get_spans_q_value_on_each_region(bellman_of_q_value)
            )

            if maximum_span > self.epsilon_span:
                # If maximum span is greater than the wanted bound

                self.contracted_q_value = self.partition.divide_region_along_tq(
                    bellman_of_q_value,
                    self.epsilon_span,
                    self.contracted_q_value,
                )

                projected_q_bellman_value = self.partition.projected_q_bellman_operator(
                    self.contracted_q_value
                )
                pbr_value = norminf(projected_q_bellman_value - self.contracted_q_value)

            if maximum_span < self.epsilon_span and pbr_value < self.epsilon_pbr:
                # print("Stopped for precision.")
                self.runtime = time() - start_time
                self.value = self.partition._partial_phi().dot(
                    self.contracted_q_value.max(axis=1)
                )
                break

    def _get_spans_q_value_on_each_region(
        self, full_q_value_function: np.ndarray
    ) -> list:
        """
        Return a list :
        [max(value[region])-min(value[region]) for region in regions]
        """
        return [
            np.ptp(full_q_value_function[region, :], axis=0)
            for region in self.partition.states_in_region
        ]
