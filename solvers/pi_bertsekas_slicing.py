"""
Implementation of the Policy Iteration algorithm given page 8 of 
"Adaptive Aggregation Methods for Infinite Horizon 
Dynamic Programming" - Bertsekas, Castanon.
"""

import numpy as np
from tqdm import tqdm
from scipy.special import softmax

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
            lr: float = 0.1,
            beta_1: float = 0.02,
            beta_2: float = 0.05,
            step_approx_y: int = 50,
            epsilon_final_policy_evaluation: float = 1e-3,
            max_iter_evaluation: int = int(1e2),
            max_iter_policy_update: int = int(1e4),
            max_time: float = float(3e3),
            verbose: bool = False,
    ):
        # Class arguments
        self.env = env
        self.discount = discount
        self.lr = lr
        self.epsilon_policy_evaluation = (
            epsilon_policy_evaluation  # * (1 - self.discount)
        )
        self.max_time = max_time
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.name = "pi_bertsekas_slicing"
        self.verbose = verbose
        self.epsilon_final_policy_evaluation = epsilon_final_policy_evaluation
        self.step_approx_y = step_approx_y
        self.max_iter_evaluation = max_iter_evaluation
        self.max_iter_policy_update = max_iter_policy_update
        self.fixed_partition = FixedValuePartition(self.env, self.discount)
        self.policy_theta = np.ones((self.env.state_dim, self.env.action_dim))
        self.policy = softmax(self.policy_theta, axis=1)

    def span_state_space(self, value: np.ndarray) -> float:
        return value.max() - value.min()

    def run(self):
        start_time = time.time()
        value_list = []
        print_dict = {}

        i_episode = 0
        while True:
            # Policy Evaluation
            self.value = self._policy_evaluation(
                self.policy,
            )
            tem_value = self._policy_evaluation_stand(self.policy, self.max_iter_evaluation)
            tem_print = np.mean(tem_value)
            if i_episode == 0:
                end_time = time.time()
                print_dict.update({'{}'.format(i_episode): [tem_print, end_time - start_time]})
                print("Episode: {0:<5} Time :{2:5.2f} | Value {1:5.2f}".format(0, tem_print, end_time - start_time))
            # Policy Update
            q_value = self.bellman_no_max(self.value)
            self.policy_theta = (self.policy_theta + self.lr * q_value)
            self.policy = softmax(self.policy_theta, axis=1)
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

    def _policy_evaluation(self, policy) -> np.ndarray:
        transition_policy, reward_policy = compute_transition_reward_policy(
            self.env, policy
        )
        self.omega_1 = np.inf
        self.omega_2 = np.inf
        self.value = np.zeros((self.env.state_dim))

        for i in range(self.max_iter_evaluation):
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
        return self.value + self.discount / 2 / (1 - self.discount) * span_tv_minus_v

    def bellman_no_max(self, value):
        return bellman_operator(self.env, value, self.discount)

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
