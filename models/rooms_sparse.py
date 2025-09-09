# The model can be found described in Hengst - Hierarchical approaches

import numpy as np
from utils.generic_model import GenericModel
from scipy.sparse import dok_array

param_list: list = [
    {"size_of_one_room": room_size, "number_of_doors": 1}
    for room_size in [5]
]
default_params = param_list[0]


class Model(GenericModel):
    def __init__(self, params: dict = default_params):
        self.size_of_one_room, self.n_doors = (
            params["size_of_one_room"],
            params["number_of_doors"],
        )
        self.params = params

        self.action_dim = 4
        self.state_dim = 4 * self.size_of_one_room**2

        self.name = "{}_{}_rooms_sparse_{}_{}".format(
            self.state_dim,
            self.action_dim,
            params["size_of_one_room"],
            params["number_of_doors"],
        )

    def _build_model(self):
        self.action_dim = 4
        self.state_dim = 4 * self.size_of_one_room**2
        self.side_size = self.size_of_one_room * 2

        self.door_step = self.size_of_one_room / (self.n_doors + 1)

        self.doors_indices = [
            int(i * self.door_step) for i in range(1, self.n_doors + 1)
        ] + [
            self.size_of_one_room + int(i * self.door_step)
            for i in range(1, self.n_doors + 1)
        ]

        self.reward_matrix = -np.ones((self.state_dim, self.action_dim))
        self.transition_matrix: list = [
            dok_array((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.transition_matrix = self._add_displacements(
            self.transition_matrix, self.side_size
        )
        self.transition_matrix = self._add_walls(self.transition_matrix, self.side_size)
        self.transition_matrix, self.reward_matrix = self._add_exit(
            self.transition_matrix, self.reward_matrix, self.side_size
        )

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]

    def _coord(self, s, size):
        assert s < size**2
        return s // size, s % size

    def _state(self, i, j, size):
        return i * size + j

    def _next_state(self, i, j, direction):
        assert isinstance(direction, int) and 0 <= direction < 4
        if direction == 0:
            return i - 1, j
        elif direction == 1:
            return i + 1, j
        elif direction == 2:
            return i, j + 1
        else:
            return i, j - 1

    def _is_in_grid(self, i: int, j: int, size: int):
        return 0 <= i < size and 0 <= j < size

    def _add_displacements(self, transition_matrix: np.ndarray, size: int):
        for a in range(self.action_dim):
            for s in range(self.state_dim):
                i, j = self._coord(s, size)
                next_coord = self._next_state(i, j, a)
                if self._is_in_grid(*next_coord, size):
                    transition_matrix[a][s, s] = 0.2
                    transition_matrix[a][s, self._state(*next_coord, size)] = 0.8
                else:
                    transition_matrix[a][s, s] = 1.0
        return transition_matrix

    def _add_wall_element(
        self, s1: int, s2: int, transition_matrix: np.ndarray, direction: str, size: int
    ):
        assert direction in ["horizontal", "vertical"]
        i, j = self._coord(s1, size)
        k, l = self._coord(s2, size)
        assert (i - k) ** 2 + (j - l) ** 2 == 1
        if direction == "vertical":
            assert j < l
            transition_matrix[2][s1, s1] = 1.0
            transition_matrix[2][s1, s2] = 0.0
            transition_matrix[3][s2, s2] = 1.0
            transition_matrix[3][s2, s1] = 0.0
        else:
            assert i < k
            transition_matrix[1][s1, s1] = 1.0
            transition_matrix[1][s1, s2] = 0.0
            transition_matrix[0][s2, s2] = 1.0
            transition_matrix[0][s2, s1] = 0.0
        return transition_matrix

    def _add_walls(self, transition_matrix: np.ndarray, size: int):
        # Horizontal wall
        direction = "horizontal"
        i, k = size // 2 - 1, size // 2
        for index in range(size):
            if index not in self.doors_indices:
                j, l = index, index
                s1, s2 = self._state(i, j, size), self._state(k, l, size)
                transition_matrix = self._add_wall_element(
                    s1, s2, transition_matrix, direction, size
                )

        # Vertical wall
        direction = "vertical"
        j, l = size // 2 - 1, size // 2
        for index in range(size):
            if index not in self.doors_indices:
                i, k = index, index
                s1, s2 = self._state(i, j, size), self._state(k, l, size)
                transition_matrix = self._add_wall_element(
                    s1, s2, transition_matrix, direction, size
                )

        return transition_matrix

    def _add_exit(
        self, transition_matrix: np.ndarray, reward_matrix: np.ndarray, size: int
    ):
        i, j = 0, size // 4
        s = self._state(i, j, size)
        for a in range(self.action_dim):
            transition_matrix[a][s, :] = 0
            transition_matrix[a][s, s] = 1
        reward_matrix[s, :] = 0

        return transition_matrix, reward_matrix
