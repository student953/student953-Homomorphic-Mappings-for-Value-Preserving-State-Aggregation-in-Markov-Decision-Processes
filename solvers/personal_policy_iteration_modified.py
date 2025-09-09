import time

import numpy as np
from scipy.special import softmax

from utils.generic_model import GenericModel


class Solver:
    def __init__(
            self,
            env: GenericModel,
            discount: float,
            lr: float = 0.1,
            max_iter_evaluation: int = int(1e3),
            max_iter_policy_update: int = int(1e4),
            max_time: float = 3e3,
            epsilon_policy_evaluation: float = 1e-3,
    ):
        # Class arguments
        self.env = env
        self.discount = discount
        self.lr = lr
        self.max_time = max_time
        self.max_iter_evaluation = max_iter_evaluation
        self.max_iter_policy_update = max_iter_policy_update
        self.name = "personal_policy_iteration_modified"
        self.epsilon_policy_evaluation = epsilon_policy_evaluation
        self.policy_theta = np.ones((self.env.state_dim, self.env.action_dim))
        self.policy = softmax(self.policy_theta, axis=1)

    def run(self):
        start_time = time.time()
        value_list = []
        print_dict = {}
        i_episode = 0
        while True:
            self.value = self._policy_evaluation(
                self.policy, self.epsilon_policy_evaluation, self.max_iter_evaluation
            )

            if i_episode == 0:
                end_time = time.time()
                print_dict.update({'{}'.format(i_episode): [np.mean(self.value), end_time - start_time]})
                print("Episode: {0:<5} Time :{2:5.2f} | Value {1:5.2f}".format(0, np.mean(self.value),
                                                                               end_time - start_time))
            q_value = self.bellman_no_max(self.value)
            self.policy_theta = (self.policy_theta + self.lr * q_value)
            self.policy = softmax(self.policy_theta, axis=1)
            value_list.append(np.mean(self.value))
            if (i_episode + 1) % 1 == 0:
                end_time = time.time()
                print("Episode: {0:<5} Time :{2:5.2f} | Value {1:5.2f} Time {2:5.2f}"
                      "".format(i_episode, np.mean(value_list), end_time - start_time))
                print_dict.update({'{}'.format(i_episode): [np.mean(self.value), end_time - start_time]})
                value_list = []
            end_time = time.time() - start_time
            i_episode += 1
            if i_episode > self.max_time:
                break
        return print_dict

    def _policy_evaluation(
            self,
            policy: np.ndarray,
            epsilon_policy_evaluation: float,
            max_iteration_evaluation: int,
    ) -> np.ndarray:
        eval_iter = 0
        transition_policy, reward_policy = self._compute_transition_reward_pi(policy)
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
        return transition_policy, reward_policy

    def bellman_no_max(self, value):
        q_value = np.zeros((self.env.state_dim, self.env.action_dim))

        for aa in range(self.env.action_dim):
            q_value[:, aa] = (
                    self.env.reward_matrix[:, aa]
                    + self.discount * self.env.transition_matrix[aa] @ value
            )

        return q_value
