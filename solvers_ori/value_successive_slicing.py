"""
This code allows to slice a state space progressively 
along an evolving value function.
"""


import numpy as np
from utils.generic_model import GenericModel
from time import time
from utils.generic_model import GenericModel
from utils.value_partition_class import ValuePartition
from utils.calculus import optimal_bellman_operator, norminf
from utils.calculus_projected import (
    apply_pobo_until_var_small,
    projected_optimal_bellman_operator,
)
from utils.calculus_partition import span_by_region


class Solver:
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        epsilon: float = 1e-3,  # Final Precision
        verbose: bool = False,
    ):
        # Class arguments
        self.env = model
        self.discount = discount
        self.epsilon = epsilon
        self.name = "value_successive_slicing"
        self.verbose = verbose

        # Variables
        self.partition = ValuePartition(self.env, self.discount)
        self.contracted_value: np.ndarray

        self.epsilon_pbr = (
            self.epsilon * (1 - self.discount) / 2
        )  # Bound on the projected bellman residual
        self.epsilon_span = (
            self.epsilon * (1 - self.discount) / 2
        )  # Bound on the span of bellman(V)

    def run(self):
        start_time = time()
        self.contracted_value = np.zeros((len(self.partition.states_in_region)))

        while True:
            self.partition._compute_aggregate_transition_reward()

            self.contracted_value, pbr_value = apply_pobo_until_var_small(
                self.env,
                self.discount,
                self.partition.aggregate_transition_matrix,
                self.partition.aggregate_reward_matrix,
                self.partition.weights,
                self.epsilon_pbr,
                self.contracted_value,
            )

            # Once we sufficiently applied PB operator
            # We compute maximum span to eventually change partition
            value = self.partition._partial_phi().dot(self.contracted_value)
            bellman_of_value = optimal_bellman_operator(self.env, value, self.discount)
            maximum_span = max(
                span_by_region(bellman_of_value, self.partition.states_in_region)
            )

            if self.verbose:
                print("maximum_span", maximum_span)
                print("K", self.partition.region_number)

            if maximum_span > self.epsilon_span:
                # If maximum span is greater than the wanted bound
                self.contracted_value = self.partition.divide_regions_along_tv(
                    bellman_of_value,
                    self.epsilon_span,
                    self.contracted_value,
                )

                self.partition._compute_aggregate_transition_reward()

                pbellman_value = projected_optimal_bellman_operator(
                    self.env,
                    self.discount,
                    self.contracted_value,
                    self.partition.aggregate_transition_matrix,
                    self.partition.aggregate_reward_matrix,
                    self.partition.weights,
                )

                pbr_value = norminf(pbellman_value - self.contracted_value)

            if maximum_span < self.epsilon_span and pbr_value < self.epsilon_pbr:
                # print("Stopped for precision.")
                self.runtime = time() - start_time
                self.value = self.partition._partial_phi().dot(self.contracted_value)
                break
