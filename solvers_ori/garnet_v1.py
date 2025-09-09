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
        "max_reward": 0.3,
        "independent_rate": 0.8,
    }
    for sdim in [500]
    for sparsity_transition in [0.7]
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
        self.independent_rate = params["independent_rate"]
        self.independent_dim = int(self.independent_rate * self.state_dim)

        self.name = "{}_{}_garnet_{}_{}_{}_{}_{}".format(
            self.state_dim,
            self.action_dim,
            self.sparsity_transition,
            self.sparsity_reward,
            self.min_reward,
            self.max_reward,
            self.independent_rate,
        )

    def _build_model(self):
        random_generator = np.random.default_rng(seed=0)
        self.reward_matrix = (
                random_generator.random(size=(self.state_dim, self.action_dim))
                * (self.max_reward - self.min_reward)
                + self.min_reward
        )
        self.transition_matrix = []
        base_matrix = random(
            self.independent_dim,
            self.state_dim,
            density=self.sparsity_transition,
            format="lil",
            data_rvs=np.random.rand,
        )

        for _ in range(self.action_dim):
            a_matrix = random(
                self.state_dim,
                self.independent_dim,
                density=self.sparsity_transition,
                format="lil",
                data_rvs=np.random.rand,
            )
            self.transition_matrix.append(
                a_matrix @ base_matrix
            )

        self.transition_matrix = [i for i in self.transition_matrix]
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
