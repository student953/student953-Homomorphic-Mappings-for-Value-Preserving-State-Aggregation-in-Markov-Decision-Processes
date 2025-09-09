#  Tournaire - Factored reinforcement learning for auto-scaling in tandem queues

import numpy as np
from utils.generic_model import GenericModel
from scipy.sparse import dok_array
import itertools


def space_dictionary(list_of_tuples: list):
    ranges = (range(*dimension_tuple) for dimension_tuple in list_of_tuples)
    list_of_ranges = tuple(ranges)
    return {elem: i for i, elem in enumerate(itertools.product(*list_of_ranges))}


param_list = [
    {
        "actions_per_server": aa,
        "lambda": 0.6,
        "mu_1": 0.2,
        "mu_2": 0.2,
        "B1": k1 + 6,
        "B2": k1 + 6,
        "K1": k1,
        "K2": k1,
        "CA": 1,
        "CD": 1,
        "CH": 1,
        "CS": 1,
        "CR": 1,
    }
    for k1 in [6]
    for aa in [3]
]
default_params = param_list[0]


class Model(GenericModel):
    def __init__(self, params=default_params):
        self.params = params
        self.actions_per_server = params["actions_per_server"]
        assert self.actions_per_server % 2 > 0, "The number of actions should be odd."
        self.arrival_rate_lambda = params["lambda"]
        self.mu_1 = params["mu_1"]
        self.mu_2 = params["mu_2"]
        self.B1 = params["B1"]
        self.B2 = params["B2"]
        self.K1 = params["K1"]
        self.K2 = params["K2"]
        self.CA = params["CA"]
        self.CD = params["CD"]
        self.CH = params["CH"]
        self.CS = params["CS"]
        self.CR = params["CR"]

        self.gamma = 0.99

        self.lambda_tilde = (
            self.arrival_rate_lambda + self.K1 * self.mu_1 + self.K2 * self.mu_2
        )

        self.state_dim = np.product([self.B1 + 1, self.B2 + 1, self.K1, self.K2])
        self.action_dim = self.actions_per_server**2

        self.name = "{}_{}_tandem_queue_tournaire".format(
            self.state_dim, self.action_dim
        )
        for param in params.values():
            self.name += "_{}".format(param)

    def _build_model(self):
        self.state_dim = np.product([self.B1 + 1, self.B2 + 1, self.K1, self.K2])
        self.action_dim = self.actions_per_server**2

        self.state_space_tuples = [
            (0, self.B1 + 1),
            (0, self.B2 + 1),
            (1, self.K1 + 1),
            (1, self.K2 + 1),
        ]

        self.state_encoding = space_dictionary(self.state_space_tuples)
        self.state_decoding = list(self.state_encoding.keys())

        min_action = int(-0.5 * self.actions_per_server)
        max_action = int(0.5 * self.actions_per_server)
        self.action_space_tuples = [
            (min_action, max_action + 1),
            (min_action, max_action + 1),
        ]

        self.action_encoding = space_dictionary(self.action_space_tuples)
        self.action_decoding = list(self.action_encoding.keys())

        self.transition_matrix: list = [
            dok_array((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.build_p()
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))
        self.build_r()

    def n1(self, k):
        return min(max(1, k), self.K1)

    def n2(self, k):
        return min(max(1, k), self.K2)

    def lambda_function(self, s, a):
        m1, m2, k1, k2 = self.state_decoding[s]
        a1, a2 = self.action_decoding[a]
        return (
            self.arrival_rate_lambda
            + self.mu_1 * min(m1, self.n1(k1 + a1))
            + self.mu_2 * min(m2, self.n2(k2 + a2))
        )

    def c1(self, s, a):
        m1, _, k1, _ = self.state_decoding[s]
        a1, _ = self.action_decoding[a]
        return self.n1(k1 + a1) * self.CS + m1 * self.CH

    def c2(self, s, a):
        _, m2, _, k2 = self.state_decoding[s]
        _, a2 = self.action_decoding[a]
        return self.n2(k2 + a2) * self.CS + m2 * self.CH

    def h1(self, s, a):
        m1, _, _, _ = self.state_decoding[s]
        a1, _ = self.action_decoding[a]
        return (
            self.CA * int(a1 == 1)
            + self.CD * int(a1 == -1)
            + self.arrival_rate_lambda
            * self.CR
            * int(m1 == self.B1 - 1)
            / (self.lambda_function(s, a) + self.gamma)
        )

    def h2(self, s, a):
        m1, m2, k1, _ = self.state_decoding[s]
        a1, a2 = self.action_decoding[a]
        return (
            self.CA * int(a2 == 1)
            + self.CD * int(a2 == -1)
            + min(m1, self.n1(k1 + a1))
            * self.mu_1
            * self.CR
            * int(m2 == self.B2 - 1)
            / (self.lambda_function(s, a) + self.gamma)
        )

    def s1p(self, s, a):
        m1, m2, k1, k2 = self.state_decoding[s]
        a1, a2 = self.action_decoding[a]
        return self.state_encoding[
            (min(m1 + 1, self.B1), m2, self.n1(k1 + a1), self.n2(k2 + a2))
        ]

    def s2p(self, s, a):
        m1, m2, k1, k2 = self.state_decoding[s]
        a1, a2 = self.action_decoding[a]
        return self.state_encoding[
            (max(m1 - 1, 0), min(m2 + 1, self.B2), self.n1(k1 + a1), self.n2(k2 + a2))
        ]

    def s3p(self, s, a):
        m1, m2, k1, k2 = self.state_decoding[s]
        a1, a2 = self.action_decoding[a]
        return self.state_encoding[
            (m1, max(m2 - 1, 0), self.n1(k1 + a1), self.n2(k2 + a2))
        ]

    def Reward(self, s, a):
        return (
            (self.c1(s, a) + self.c2(s, a)) / (self.lambda_function(s, a) + self.gamma)
            + self.h1(s, a)
            + self.h2(s, a)
        )

    def Reward_tilde(self, s, a):
        return (
            self.Reward(s, a)
            * (self.lambda_function(s, a) + self.gamma)
            / (self.lambda_tilde + self.gamma)
        )

    def build_p(self):
        for a in range(self.action_dim):
            for s in range(self.state_dim):
                state, action = self.state_decoding[s], self.action_decoding[a]
                m1, m2, k1, k2 = state
                a1, a2 = action
                if m1 == self.B1 and self.n1(k1 + a1) == a1 and self.n2(k2 + a2) == a2:
                    self.transition_matrix[a][s, s] += self.arrival_rate_lambda
                else:
                    self.transition_matrix[a][
                        s, self.s1p(s, a)
                    ] += self.arrival_rate_lambda
                    self.transition_matrix[a][s, self.s2p(s, a)] += self.mu_1 * min(
                        m1, self.n1(k1 + a1)
                    )
                    self.transition_matrix[a][s, self.s3p(s, a)] += self.mu_2 * min(
                        m2, self.n2(k2 + a2)
                    )
                    self.transition_matrix[a][
                        s, s
                    ] += self.lambda_tilde - self.lambda_function(s, a)
        for a in range(self.action_dim):
            self.transition_matrix[a] /= self.lambda_tilde

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]

    def build_r(self):
        for a in range(self.action_dim):
            for s in range(self.state_dim):
                self.reward_matrix[s, a] = self.Reward_tilde(s, a)

    def reformat_regions(self):
        self.regions = [[int(state) for state in region] for region in self.regions]
