import numpy as np
from scipy.sparse import random
from utils.generic_model import GenericModel

param_list = [
    {
        "state_dim": sdim,
        "action_dim": action,
        "sparsity_transition": sparsity_transition,
        "sparsity_reward": 1.0,
        "min_reward": 0.0,
        "max_reward": 1.0,
    }
    for sdim in [100, 200, 500, 1000, 2000]
    for sparsity_transition in [0.3]
    for action in [10]
]
default_params = param_list[0]


class Model(GenericModel):
    def __init__(self, params: dict = default_params):
        self.state_dim = params["state_dim"]
        self.action_dim = params["action_dim"]
        self.sparsity_transition = params["sparsity_transition"]
        self.sparsity_reward = params["sparsity_reward"]
        self.min_reward = params["min_reward"]
        self.max_reward = params["max_reward"]

        self.name = "{}_{}_garnet_{}_{}_{}_{}".format(
            self.state_dim,
            self.action_dim,
            self.sparsity_transition,
            self.sparsity_reward,
            self.min_reward,
            self.max_reward,
        )

    def _build_model(self):
        random_generator = np.random.default_rng(seed=0)
        self.reward_matrix = (
            random_generator.random(size=(self.state_dim, self.action_dim))
            * (self.max_reward - self.min_reward)
            + self.min_reward
        )
        u_dim = 0.8 * int(self.state_dim)
        self.transition_matrix = [
            random(
                self.state_dim,
                self.state_dim,
                density=self.sparsity_transition,
                format="lil",
                data_rvs=np.random.rand,
            )
            for _ in range(self.action_dim)
        ]
        for aa in range(self.action_dim):
            for ss in range(self.state_dim):
                self.transition_matrix[aa][ss, 0] += 1e-6
                self.transition_matrix[aa][ss] *= (
                    1 / self.transition_matrix[aa][ss].sum()
                )

                self.transition_matrix[aa][ss, 0] += (
                    1 - self.transition_matrix[aa][ss].sum()
                )

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
