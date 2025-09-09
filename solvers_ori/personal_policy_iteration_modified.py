import numpy as np
from utils.generic_model import GenericModel
from time import time
from utils.generic_model import GenericModel
from utils.q_value_partition_class import QValuePartition


class Solver:
    def __init__(
        self,
        env: GenericModel,
        discount: float,
        max_iter_evaluation: int = int(1e8),
        max_iter_policy_update: int = int(1e8),
        epsilon_policy_evaluation: float = 1e-3,
    ):
        # Class arguments
        self.env = env
        self.discount = discount
        self.max_iter_evaluation = max_iter_evaluation
        self.max_iter_policy_update = max_iter_policy_update
        self.name = "personal_policy_iteration_modified"
        self.epsilon_policy_evaluation = epsilon_policy_evaluation

    def run(self):
        start_time = time()
        self.policy = np.zeros((self.env.state_dim))
        policy_update_iter = 0
        while True:
            self.value = self._policy_evaluation(
                self.policy, self.epsilon_policy_evaluation, self.max_iter_evaluation
            )
            new_policy = self.bellman_no_max(self.value).argmax(axis=1)
            if np.all(new_policy == self.policy) or (
                policy_update_iter == self.max_iter_policy_update
            ):
                self.runtime = time() - start_time
                break
            else:
                self.policy = new_policy

    def _policy_evaluation(
        self,
        policy: np.ndarray,
        epsilon_policy_evaluation: float,
        max_iteration_evaluation: int,
    ) -> np.ndarray:
        eval_iter = 0
        transition_policy, reward_policy = self._compute_transition_reward_pi(policy)
        value = np.zeros((self.env.state_dim))
        while True:
            eval_iter += 1
            new_value = reward_policy + self.discount * transition_policy @ value
            variation = np.absolute(new_value - value).max()
            if (
                variation
                < ((1 - self.discount) / self.discount) * epsilon_policy_evaluation
            ) or eval_iter == max_iteration_evaluation:
                return new_value
            else:
                value = new_value

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

    def bellman_no_max(self, value):
        q_value = np.zeros((self.env.state_dim, self.env.action_dim))

        for aa in range(self.env.action_dim):
            q_value[:, aa] = (
                self.env.reward_matrix[:, aa]
                + self.discount * self.env.transition_matrix[aa] @ value
            )

        return q_value
