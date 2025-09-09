"""
This code allows to slice a state space progressively 
along an evolving value function.
"""

import numpy as np
import time

from scipy.special import softmax

from utils.generic_model import GenericModel
from utils.value_partition_class import ValuePartition
from utils.calculus import optimal_bellman_operator, norminf


class Solver:
    def __init__(
            self,
            env: GenericModel,
            discount: float,
            epsilon: float = 1e-3,  # Final Precision
            verbose: bool = False,
            max_iter_loop: int = 1000,
            iter_agg: int = 1,
            iter_bellman: int = 100,
            lr: float = 0.1,
            max_time: float = 3e3,
            max_iter_evaluation: int = int(1e3),
            max_iter_policy_update: int = int(1e4),
    ):
        # Class arguments
        self.env = env
        self.discount = discount
        self.epsilon = epsilon
        self.name = "chen_slicing"
        self.verbose = verbose
        self.lr = lr
        self.max_iter_evaluation = max_iter_evaluation
        self.max_iter_policy_update = max_iter_policy_update

        self.iter_agg = iter_agg
        self.max_time = max_time
        # Variables
        self.partition = ValuePartition(self.env, self.discount)
        self.contracted_value: np.ndarray

        self.alpha_function = lambda t: 1 / (t + 1) ** 2

        self.policy_theta = np.ones((self.env.state_dim, self.env.action_dim))
        self.policy = softmax(self.policy_theta, axis=1)

    def run(self):
        start_time = time.time()
        self.value = np.zeros((self.env.state_dim))
        print_dict = {}
        n = 0
        value_list = []
        i_episode = 0
        while True:
            n += 1
            for _ in range(self.max_iter_evaluation):
                self.new_value = optimal_bellman_operator(
                    self.env, self.value, self.discount
                )
                self.value = self.new_value

            self.partition = ValuePartition(self.env, self.discount)
            self.partition.divide_regions_along_tv(
                self.value,
                self.epsilon,
                np.zeros((len(self.partition.states_in_region))),
            )
            self.partition._build_weights()
            self.contracted_value = self.partition.weights @ self.value
            alpha = self.alpha_function(n * (self.iter_agg + self.max_iter_evaluation))

            for _ in range(self.iter_agg):
                self.extended_value = (
                        self.partition._partial_phi() @ self.contracted_value
                )
                for k in range(len(self.partition.states_in_region)):
                    ss = np.random.choice(self.partition.states_in_region[k])
                    tsv_action = np.zeros((self.env.action_dim))
                    for aa in range(self.env.action_dim):
                        tsv_action[aa] = (
                                self.env.reward_matrix[ss, aa]
                                + self.discount
                                * self.env.transition_matrix[aa].getrow(ss)
                                @ self.extended_value
                        )
                    self.contracted_value[k] = (1 - alpha) * self.contracted_value[
                        k
                    ] + alpha * max(tsv_action)

            self.value = self.partition._partial_phi().dot(self.contracted_value)
            tem_value = self._policy_evaluation_stand(self.policy, self.max_iter_evaluation)
            tem_print = np.mean(tem_value)
            if i_episode == 0:
                end_time = time.time()
                print_dict.update({'{}'.format(i_episode): [tem_print, end_time-start_time]})
                print("Episode: {0:<5} Time :{2:5.2f} | Value {1:5.2f}".format(0, tem_print,end_time-start_time))

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

    def bellman_no_max(self, value):
        q_value = np.zeros((self.env.state_dim, self.env.action_dim))

        for aa in range(self.env.action_dim):
            q_value[:, aa] = (
                    self.env.reward_matrix[:, aa]
                    + self.discount * self.env.transition_matrix[aa] @ value
            )

        return q_value

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
