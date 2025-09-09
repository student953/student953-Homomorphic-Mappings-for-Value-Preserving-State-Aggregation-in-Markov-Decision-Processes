"""
Implementation of modified Policy Iteration 
with a changed Policy Evaluation step.

The new Policy Evaluation successively slices 
the state space to update by block the current 
value function V^pi.
"""

import numpy as np
from scipy.special import softmax

from utils.generic_model import GenericModel
import time
from utils.generic_model import GenericModel
from utils.pi_partition_class import PiPartition
from utils.calculus import (
    norminf,
    bellman_operator,
    value_span_on_regions,
    bellman_policy_operator,
    compute_transition_reward_policy,
)
from utils.calculus_projected import (
    projected_policy_bellman_operator,
    apply_ppbo_until_var_small,
)
from typing import List, Tuple


class Solver:
    def __init__(
            self,
            env: GenericModel,
            discount: float,
            lr: float = 0.1,
            max_iter_evaluation: int = int(1e3),
            max_iter_policy_update: int = int(1e4),
            epsilon_policy_evaluation: float = 1e-3,  # Policy Evaluation Precision
            epsilon_final_policy_evaluation: float = 1e-3,  # Final Precision
            epsilon_variation: float = 1e-3,
            max_time: float = 3e3,
            verbose: bool = False,
    ):
        # Class arguments
        self.env = env
        self.discount = discount
        self.lr = lr
        self.max_time = max_time
        self.name = "pim_successive_slicing"
        self.max_iter_evaluation = max_iter_evaluation
        self.max_iter_policy_update = max_iter_policy_update
        self.epsilon_policy_evaluation = epsilon_policy_evaluation
        self.epsilon_final_policy_evaluation = epsilon_final_policy_evaluation
        self.epsilon_variation = epsilon_variation
        self.verbose = verbose
        self.policy_theta = np.ones((self.env.state_dim, self.env.action_dim))
        self.policy = softmax(self.policy_theta, axis=1)
        self.pi_partition = PiPartition(self.env, self.discount)
        self.value = np.zeros((self.env.state_dim))

    def run(self):
        start_time = time.time()
        value_list = []
        print_dict = {}
        i_episode = 0
        while True:
            # Policy Evaluation
            self.value = self._policy_evaluation(
                self.policy,
                self.epsilon_policy_evaluation,
                self.value,
                self.max_iter_evaluation
            )
            tem_value = self._policy_evaluation_stand(self.policy, self.max_iter_evaluation)
            tem_print = np.mean(tem_value)
            if i_episode == 0:
                end_time = time.time()
                print_dict.update({'{}'.format(i_episode): [tem_print, end_time - start_time]})
                print("Episode: {0:<5} Time :{2:5.2f} | Value {1:5.2f}".format(0, tem_print, end_time - start_time))

            # Policy Update
            q_value = bellman_operator(self.env, self.value, self.discount)
            new_value = q_value.max(axis=1)
            self.value = new_value
            self.policy_theta = (self.policy_theta + self.lr * q_value)
            self.policy = softmax(self.policy_theta, axis=1)

            # calculate real value
            value_list.append(tem_print)
            if (i_episode + 1) % 1 == 0:
                end_time = time.time()
                print("Episode: {0:<5} Time :{2:5.2f} | Value {1:5.2f}".format(i_episode, tem_print,end_time-start_time))
                print_dict.update({'{}'.format(i_episode): [tem_print, end_time-start_time]})
                value_list = []
            i_episode += 1
            end_time = time.time() - start_time
            if end_time > self.max_time:
                break
        return print_dict


    def _apply_pbo_until_pbr_small(
            self, contracted_value: np.ndarray, epsilon_pbr: float
    ) -> tuple[np.ndarray, float]:
        """
        While the PBR is too big, apply the PBO.
        """
        self.pi_partition._compute_aggregate_transition_reward_policy(True)
        # We compute Pi T V_ and the Projected Bellman Residual
        new_contracted_value = projected_policy_bellman_operator(
            self.discount,
            contracted_value,
            self.pi_partition.aggregate_transition_policy,
            self.pi_partition.aggregate_reward_policy,
        )
        pbr_value = norminf(new_contracted_value - contracted_value)
        contracted_value = new_contracted_value

        while pbr_value > epsilon_pbr:
            # While the Projected Bellman Residual is too big,
            # apply the Projected Bellman Operator
            new_contracted_value = projected_policy_bellman_operator(
                self.discount,
                contracted_value,
                self.pi_partition.aggregate_transition_policy,
                self.pi_partition.aggregate_reward_policy,
            )

            pbr_value = norminf(new_contracted_value - contracted_value)
            contracted_value = new_contracted_value

        return contracted_value, pbr_value

    def _get_maximum_span_and_bellman_value(
            self,
            contracted_value: np.ndarray,
            transition_policy: np.ndarray,
            reward_policy: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        partial_phi = self.pi_partition._partial_phi()
        value = partial_phi.dot(contracted_value)
        bellman_of_value = bellman_policy_operator(
            value, self.discount, transition_policy, reward_policy
        )
        return (
            max(
                value_span_on_regions(
                    bellman_of_value, self.pi_partition.states_in_region
                )
            ),
            bellman_of_value,
        )

    def _policy_evaluation(
            self,
            policy: np.ndarray,
            epsilon_policy_evaluation: float,
            initial_full_value: np.ndarray,
            max_iteration_evaluation,
    ) -> np.ndarray:
        epsilon_pbr = (1 - self.discount) * epsilon_policy_evaluation / 2
        epsilon_span = (1 - self.discount) * epsilon_policy_evaluation / 2

        self.pi_partition._build_weights()
        contracted_value = self.pi_partition.weights.dot(initial_full_value)

        transition_policy, reward_policy = compute_transition_reward_policy(
            self.env, policy
        )

        self.pi_partition.update_transition_reward_policy(
            transition_policy, reward_policy
        )

        for i in range(max_iteration_evaluation):
            self.pi_partition._compute_aggregate_transition_reward_policy(True)
            pbr_value, contracted_value = apply_ppbo_until_var_small(
                self.discount,
                self.pi_partition.aggregate_transition_policy,
                self.pi_partition.aggregate_reward_policy,
                epsilon_pbr,
                contracted_value,
            )

            # Compute span
            maximum_span, bellman_of_value = self._get_maximum_span_and_bellman_value(
                contracted_value, transition_policy, reward_policy
            )

            self.pi_partition.update_transition_reward_policy(
                transition_policy, reward_policy
            )
            self.aggregate_reward_policy = None
            self.aggregate_transition_policy = None
            self.pi_partition._compute_aggregate_transition_reward_policy()

            # If span and pbr small, break
            if maximum_span < epsilon_span and pbr_value < epsilon_pbr:
                self.value = self.pi_partition._partial_phi().dot(contracted_value)
                return self.value

            # Else, the span should be big (normally), we divide partition
            contracted_value = self.pi_partition.divide_regions_along_tv(
                bellman_of_value,
                epsilon_span,
                contracted_value,
            )
        return self.pi_partition._partial_phi().dot(contracted_value)

    def _policy_evaluation_stand(
            self,
            policy: np.ndarray,
            max_iteration_evaluation: int,
    ) -> np.ndarray:
        eval_iter = 0
        transition_policy, reward_policy = self._compute_transition_reward_pi(policy)
        value = np.zeros((self.env.state_dim,))
        for i in range(max_iteration_evaluation):
            eval_iter += 1
            new_value = reward_policy + self.discount * transition_policy @ value
            value = new_value
        return new_value

    def _compute_transition_reward_pi(self, policy):
        policy_t = policy.T[:, :, np.newaxis]
        matrix_all = []
        for a in self.env.transition_matrix:
            matrix_all.append(a.A[np.newaxis])
        matrix_all = np.vstack(matrix_all)
        transition_policy = (policy_t * matrix_all).sum(0)
        reward_policy = (policy * self.env.reward_matrix).sum(1)
        return transition_policy, reward_policy
