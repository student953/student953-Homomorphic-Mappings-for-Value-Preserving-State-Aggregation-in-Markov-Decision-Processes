# Garnet MDP - Weakly coupled MDP:
# - Divide the state space in n blocks
# - Ensure a random evolution through each block
# - Connect blocks with sparse non zeros transitions

import numpy as np
from scipy.sparse import random, lil_matrix, csr_matrix
from utils.generic_model import GenericModel

rnd_gen = np.random.default_rng(seed=0)

initial_params = {
    "block_dim": 20,
    "action_dim": 10,
    "sparsity_block": 0.5,
    "block_number": 10,
    "connection_ratio": 0.5,
}

param_list = [
    {
        "block_dim": sdim,
        "action_dim": action,
        "sparsity_block": sparsity_transition,
        "block_number": block_number,
        "connection_ratio": connection_ratio,
    }
    for sdim in [360]
    for sparsity_transition in [0.8]
    for action in [10]
    for block_number in [10]
    for connection_ratio in [0.8]
]


class Model(GenericModel):
    def __init__(self, params: dict = initial_params):
        self.block_dim = params["block_dim"]
        self.action_dim = params["action_dim"]
        self.sparsity_block = params["sparsity_block"]
        self.block_number = params["block_number"]
        self.state_dim = self.block_dim * self.block_number
        self.connection_ratio = params["connection_ratio"]

        self.name = "{}_{}_block_{}_{}".format(
            self.state_dim,
            self.action_dim,
            self.sparsity_block,
            self.block_number,
        )

    def _build_model(self):
        random_generator = np.random.default_rng(seed=0)

        self.reward_matrix = random_generator.random(
            size=(self.state_dim, self.action_dim)
        )

        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]

        self.add_block_transition()
        self.add_connections(self.connection_ratio)
        self.normalize()

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]

    def build_random_block(self, size):
        return random(
            size,
            size,
            density=self.sparsity_block,
            format="lil",
            data_rvs=np.random.rand,
        )

    def add_block_transition(self):
        for aa in range(self.action_dim):
            for block_index in range(self.block_number):
                start_index = block_index * self.block_dim
                end_index = start_index + self.block_dim
                self.transition_matrix[aa][
                    start_index:end_index, start_index:end_index
                ] = self.build_random_block(self.block_dim)

    def _update_transition_matrix(self, number_connection, aa, block_1, block_2):
        if block_1 != block_2:
            start_state = block_1 * self.block_dim
            end_state = start_state + self.block_dim
            initial_states = rnd_gen.choice(
                range(start_state, end_state),
                size=number_connection,
            )

            start_state = block_2 * self.block_dim
            end_state = (block_2 + 1) * self.block_dim

            final_states = rnd_gen.choice(
                range(start_state, end_state),
                size=number_connection,
            )

            for ss1 in initial_states:
                for ss2 in final_states:
                    self.transition_matrix[aa][ss1, ss2] = rnd_gen.random()

    def add_connections(self, connection_ratio: float):
        number_connection = int(connection_ratio * self.block_dim)
        for aa in range(self.action_dim):
            for block_1 in range(self.block_number):
                for block_2 in range(self.block_number):
                    self._update_transition_matrix(
                        number_connection, aa, block_1, block_2
                    )

    def normalize(self):
        for aa in range(self.action_dim):
            for ss in range(self.state_dim):
                self.transition_matrix[aa][ss, 0] += 1e-6
                self.transition_matrix[aa][ss] *= (
                    1 / self.transition_matrix[aa][ss].sum()
                )

                self.transition_matrix[aa][ss, 0] += (
                    1 - self.transition_matrix[aa][ss].sum()
                )
