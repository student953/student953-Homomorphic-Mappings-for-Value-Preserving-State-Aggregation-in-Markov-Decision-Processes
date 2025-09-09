"""
Implementation of the Policy Iteration algorithm given page 8 of 
"Adaptive Aggregation Methods for Infinite Horizon 
Dynamic Programming" - Bertsekas, Castanon.
"""

import numpy as np
from utils.generic_model import GenericModel
import time
from utils.generic_model import GenericModel
from utils.fixed_value_partition_class import FixedValuePartition
from utils.calculus import (
    norminf,
    bellman_operator,
    value_span_on_regions,
    iterative_policy_evaluation,
    compute_transition_reward_policy,
    inv_approximate,
)


# def inv_approximate(matrix: np.ndarray, tolerance=1e-3):
#     res = np.zeros((matrix.shape[0], matrix.shape[0]))
#     x = np.eye(matrix.shape[0]) - matrix
#     step = 0
#     while True:
#         val = x**step
#         res += val
#         if np.linalg.norm(val) < tolerance:
#             break
#         step += 1
#     return res


def bellman_policy_operator(
    value: np.ndarray,
    discount: float,
    transition_policy: np.ndarray,
    reward_policy: np.ndarray,
) -> np.ndarray:
    """
    Returns R^pi + gamma . T^pi @ value
    """
    return reward_policy + discount * transition_policy @ value


class Solver:
    def __init__(
        self,
        env: GenericModel,
        discount: float,
        epsilon_policy_evaluation: float = 1e-3,
        beta_1: float = 0.02,
        beta_2: float = 0.05,
        step_approx_y: int = 50,
        epsilon_final_policy_evaluation: float = 1e-3,
        verbose: bool = False,
    ):
        # Class arguments
        self.env = env
        self.discount = discount
        self.epsilon_policy_evaluation = (
            epsilon_policy_evaluation  # * (1 - self.discount)
        )
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.name = "pi_bertsekas_slicing"
        self.verbose = verbose
        self.epsilon_final_policy_evaluation = epsilon_final_policy_evaluation
        self.step_approx_y = step_approx_y

        self.fixed_partition = FixedValuePartition(self.env, self.discount)

    def span_state_space(self, value: np.ndarray) -> float:
        return value.max() - value.min()

    def run(self):
        start_time = time.time()
        self.policy = np.zeros((self.env.state_dim))

        while True:
            # Policy Evaluation
            self.value = self._policy_evaluation(
                self.policy,
            )

            # Policy Update
            new_policy = self.bellman_no_max(self.value).argmax(axis=1)

            if np.all(new_policy == self.policy):
                if self.verbose:
                    print("Optimal Policy Reached.")
                transition_policy, reward_policy = self._compute_transition_reward_pi(
                    self.policy
                )
                self.value = iterative_policy_evaluation(
                    transition_policy,
                    reward_policy,
                    self.discount,
                    self.epsilon_final_policy_evaluation,
                )

                self.runtime = time.time() - start_time

                break
            else:
                self.policy = new_policy

    def _policy_evaluation(self, policy) -> np.ndarray:
        transition_policy, reward_policy = compute_transition_reward_policy(
            self.env, policy
        )
        self.omega_1 = np.inf
        self.omega_2 = np.inf
        self.value = np.zeros((self.env.state_dim))

        while True:
            bellman_value = bellman_policy_operator(
                self.value, self.discount, transition_policy, reward_policy
            )
            span_tv_minus_v = self.span_state_space(bellman_value - self.value)

            if span_tv_minus_v < self.epsilon_policy_evaluation:
                self.value += self.discount / 2 / (1 - self.discount) * span_tv_minus_v
                return self.value
            else:
                if span_tv_minus_v <= self.omega_1 and span_tv_minus_v >= self.omega_2:
                    self.omega_1 = self.beta_1 * span_tv_minus_v
                else:
                    self.omega_2 = self.beta_2 * span_tv_minus_v
                    self.value = bellman_value
                    continue

                self.fixed_partition.build_partition_along_value(
                    bellman_value - self.value
                )
                self.fixed_partition._build_weights()
                Q = self.fixed_partition.weights
                W = self.fixed_partition._partial_phi()
                right_y = Q @ (bellman_value - self.value)

                left_y_before_inv = (
                    np.eye(self.fixed_partition.region_number)
                    - self.discount * Q @ transition_policy @ W
                )

                left_y = inv_approximate(left_y_before_inv)

                y = left_y @ right_y
                value_1 = self.value + W @ y
                self.value = value_1
                self.omega_2 = np.inf

    def bellman_no_max(self, value):
        return bellman_operator(self.env, value, self.discount)

    def _compute_transition_reward_pi(self, policy):
        transition_policy = np.empty((self.env.state_dim, self.env.state_dim))
        reward_policy = np.zeros(self.env.state_dim)
        for aa in range(self.env.action_dim):
            ind = (policy == aa).nonzero()[0]
            # if no rows use action a, then no need to assign this
            if ind.size > 0:
                transition_policy[ind, :] = self.env.transition_matrix[aa][
                    ind, :
                ].todense()
                reward_policy[ind] = self.env.reward_matrix[ind, aa]

        return transition_policy, reward_policy
