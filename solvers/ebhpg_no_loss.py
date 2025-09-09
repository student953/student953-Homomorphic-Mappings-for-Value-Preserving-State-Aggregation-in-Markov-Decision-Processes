"""
-------------------------------------------------
# @Project  :state_space_disaggregation-main
# @File     :covering_pi
# @Date     :2024/12/29 18:29
# @Author   :
-------------------------------------------------
"""
import copy

import numpy as np
import time

from scipy.special import softmax

from utils.NMF import nmf_multiplicative_update
from utils.generic_model import GenericModel


class Solver:
    def __init__(
            self,
            env: GenericModel,
            discount: float,
            lr: float = 0.1,
            covering_rate: float = 0.4,
            max_iter_evaluation: int = int(1e3),
            max_iter_policy_update: int = int(1e4),
            max_time: float = 3e3,
            epsilon_policy_evaluation: float = 1e-3,
    ):
        # Class arguments
        self.env = env
        self.lr = lr
        self.covering_state_num = int(self.env.state_dim * covering_rate)
        self.tau_matrix = np.abs(np.random.rand(self.covering_state_num, self.env.state_dim))
        self.tau_matrix = np.abs(2 * np.random.rand(1, self.env.state_dim)) * self.tau_matrix
        self.tau_matrix /= np.sum(self.tau_matrix, axis=1, keepdims=True)
        self.u_matrix = np.mean(self.tau_matrix, axis=0, keepdims=True).T * self.env.state_dim
        self.max_time = max_time

        self.discount = discount
        self.max_iter_evaluation = max_iter_evaluation
        self.max_iter_policy_update = max_iter_policy_update
        self.name = "pi_EBHPG"
        self.epsilon_policy_evaluation = epsilon_policy_evaluation
        self.policy_theta = np.ones((self.env.state_dim, self.env.action_dim))
        self.policy_theta /= np.sum(self.policy_theta, axis=1, keepdims=True)
        self.policy = softmax(self.policy_theta, axis=1)

        self.value = np.zeros((self.env.state_dim, self.env.action_dim))
        # calculate trans_tau and reward_tau
        # self.trans_tau_a = [self.tau_matrix @ tran @ tran for tran in self.env.transition_matrix]
        # self.reward_tau_a = self.tau_matrix @ self.env.reward_matrix

    def run(self):
        start_time = time.time()
        policy_update_iter = 0
        print_dict = {}
        value_list = []
        i_episode = 0
        while True:
            self.value = self._policy_evaluation(
                self.policy, self.epsilon_policy_evaluation, self.max_iter_evaluation
            )
            tem_value = self._policy_evaluation_stand(self.policy, self.max_iter_evaluation)
            tem_print = np.mean(tem_value)
            if i_episode == 0:
                print_dict.update({'{}'.format(i_episode): tem_print})
                print("Episode: {0:<5} | Value {1:5.2f}".format(0, tem_print))
            q_value = self.bellman_no_max(self.value)
            self.update_tau(self.value)
            self.policy_theta = (self.policy_theta + self.lr * q_value * self.u_matrix)
            self.policy = softmax(self.policy_theta, axis=1)
            value_list.append(tem_print)
            end_time = time.time()
            time_gap = end_time - start_time
            if (i_episode + 1) % 1 == 0:
                print("Episode: {0:<5} | Value {1:5.2f} | Time {2:5.2f}".format(i_episode, np.mean(value_list), time_gap))
                print_dict.update({'{}'.format(i_episode): [tem_print, time_gap]})
                value_list = []
            i_episode += 1
            if time_gap > self.max_time:
                break

        return print_dict

    def update_tau(self, v_value):
        v_value_ = copy.deepcopy(v_value).reshape(-1, 1)
        self.u_matrix = softmax(v_value_, 1) * self.env.state_dim

    def _policy_evaluation(
            self,
            policy: np.ndarray,
            epsilon_policy_evaluation: float,
            max_iteration_evaluation: int,
    ) -> np.ndarray:
        eval_iter = 0
        transition_policy, reward_policy = self._compute_transition_reward_pi(policy)
        test1 = np.linalg.eig(transition_policy)
        test2 = test1.eigenvalues
        test3 = np.abs(test2)
        test4 = np.sort(test3)
        value = np.zeros((self.env.state_dim,))
        for i in range(max_iteration_evaluation):
            eval_iter += 1
            new_value = reward_policy + self.discount * transition_policy @ value
            variation = np.absolute(new_value - value).max()
            if ((variation < ((1 - self.discount) / self.discount) * epsilon_policy_evaluation) or
                    eval_iter >= max_iteration_evaluation):
                return new_value
            else:
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
        return transition_policy @ transition_policy, reward_policy

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
        transition_policy, reward_policy = self._compute_transition_reward_pi_stand(policy)
        value = np.zeros((self.env.state_dim,))
        for i in range(max_iteration_evaluation):
            eval_iter += 1
            new_value = reward_policy + self.discount * transition_policy @ value
            value = new_value
        return new_value

    def _compute_transition_reward_pi_stand(self, policy):
        policy_t = policy.T[:, :, np.newaxis]
        matrix_all = []
        for a in self.env.transition_matrix:
            matrix_all.append(a.A[np.newaxis])
        matrix_all = np.vstack(matrix_all)
        transition_policy = (policy_t * matrix_all).sum(0)
        reward_policy = (policy * self.env.reward_matrix).sum(1)
        return transition_policy, reward_policy
