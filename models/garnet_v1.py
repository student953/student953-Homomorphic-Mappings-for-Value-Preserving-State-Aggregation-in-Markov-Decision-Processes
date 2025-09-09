import numpy as np
from scipy.sparse import random, csr_matrix

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
    for sdim in [100]
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
        self.name = "{}_{}_garnet_{}_{}_{}_{}".format(
            self.state_dim,
            self.action_dim,
            self.sparsity_transition,
            self.sparsity_reward,
            self.min_reward,
            self.max_reward,
        )
        self.transition_matrix = []

    def _build_model(self):
        random_generator = np.random.default_rng(seed=0)
        self.reward_matrix = (
            random_generator.random(size=(self.state_dim, self.action_dim))
            * (self.max_reward - self.min_reward)
            + self.min_reward
        )
        u_dim = int(0.8 * self.state_dim)

        base_ = np.random.rand(u_dim, self.state_dim)
        data = np.random.choice([0, 1], size=(u_dim, self.state_dim),
                                p=[1-self.sparsity_transition, self.sparsity_transition])
        base_ = base_ * data + 1e-6
        base_ = base_ / base_.sum(-1, keepdims=True)
        self.transition_matrix = []
        for _ in range(self.action_dim):
            a_ = np.random.rand(self.state_dim, u_dim)
            a_ = a_ / a_.sum(-1, keepdims=True)
            self.transition_matrix.append(a_ @ base_)
        self.transition_matrix = [csr_matrix(matrix) for matrix in self.transition_matrix]


# import numpy as np
# from scipy.sparse import vstack
# my_env = Model(param_list[0])
# my_env._build_model()
# big_csr = vstack(my_env.transition_matrix)
# p_nv = maximal_independent_rows_numpy(big_csr)
# print(len(p_nv))
