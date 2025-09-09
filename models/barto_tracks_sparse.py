#################################################################
#  Barto - Learning to act using real-time dynamic programming  #
#################################################################

import numpy as np
from utils.generic_model import GenericModel
from scipy.sparse import dok_matrix
from typing import Optional

# fmt: off

# 0 : Hors circuit
# 1 : Intérieur du circuit
# 2 : Départ
# 3 : Arrivée

track_L = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])

track_R = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3],
])

minitrack = np.array([
    [0, 0, 0, 3, 3],
    [0, 0, 0, 1, 1],
    [2, 1, 1, 1, 1],
    [2, 1, 1, 1, 1]
])

# fmt: on


def make_track_bigger(track: np.ndarray):
    """
    Transform a track of dimension (n, m)
    into a track of dimension (n+1, m+1).
    """
    height, width = track.shape
    new_row = np.ones((1, width))
    new_row[0, 0] = 2
    new_col = np.ones((height + 1, 1))
    new_col[0, 0] = 3
    track = np.concatenate((track, new_row), axis=0)
    track = np.concatenate((track, new_col), axis=1)
    return track


max_track_size = 4
probabilities_fail_action = [0.1]
maximum_speed = [3]
tracks_with_names = [
    # ("track_L", track_L),
    # ("track_R", track_R),
    ("track_1", minitrack),
]


for track_index in range(max_track_size):
    minitrack = make_track_bigger(minitrack)
    tracks_with_names += [("track_{}".format(track_index), minitrack)]


param_list = [
    {"p": p, "track": track, "max_speed": max_speed, "track_name": track_name}
    for p in probabilities_fail_action
    for track_name, track in tracks_with_names
    for max_speed in maximum_speed
]

default_params = {
    "p": 0.1,
    "track": minitrack,
    "max_speed": 3,
    "track_name": "minitrack",
}


class Model(GenericModel):
    def __init__(self, params: dict = default_params):
        self.params = params

        self.proba_fail_action: float = self.params["p"]
        self.track: np.ndarray = self.params["track"]
        self.max_speed = self.params["max_speed"]
        self.track_name = self.params["track_name"]
        self.X, self.Y = self.track.shape

        self.build_state_space()
        self.build_action_space()

        self.name = "{}_{}_barto_{}_{}_{}".format(
            self.state_dim,
            self.action_dim,
            self.track_name,
            self.max_speed,
            self.proba_fail_action,
        )

    def _update_transition_matrix(
        self, aa: int, ax: int, ay: int, ss1: int, x: int, y: int, sx: int, sy: int
    ):
        if self.track[x, y] == 3:
            # Finish line
            self.reward_matrix[ss1, aa] = 0.0
            self.transition_matrix[aa][ss1, ss1] = 1.0
        else:
            # Not a finish line
            new_sx, new_sy = self.compute_new_speed(sx, sy, ax, ay)
            new_x, new_y, evolved = self.compute_new_position(x, y, sx, sy)

            # Si on sort du circuit, on revient au départ
            if not evolved:
                for s2 in self.start_states:
                    self.transition_matrix[aa][ss1, s2] = self.proba_start_states
            else:
                # Avec probabilité p, la vitesse n'est pas modifiée
                s2_non_modified = self.state_space_index[(new_x, new_y, sx, sy)]
                self.transition_matrix[aa][
                    ss1, s2_non_modified
                ] += self.proba_fail_action
                # Avec probabilité 1-p, la vitesse est modifiée selon l'action
                s2_modified = self.state_space_index[(new_x, new_y, new_sx, new_sy)]
                self.transition_matrix[aa][ss1, s2_modified] += (
                    1 - self.proba_fail_action
                )

    def _build_model(self):
        self._init_matrices()

        for a, (ax, ay) in enumerate(self.action_space):
            for s1, (x, y, sx, sy) in enumerate(self.state_space):
                self._update_transition_matrix(a, ax, ay, s1, x, y, sx, sy)

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]

    def build_state_space(self):
        # A state is made of coordinate and speed : (x, y, sx, sy)
        self.state_space = [
            (x, y, sx, sy)
            for x in range(self.X)
            for y in range(self.Y)
            for sx in range(-self.max_speed, self.max_speed + 1)
            for sy in range(-self.max_speed, self.max_speed + 1)
            if self.check_on_track(x, y)
        ]

        self.state_space_index = {
            state: state_index for state_index, state in enumerate(self.state_space)
        }

        # state_dim is an integer
        self.state_dim = len(self.state_space)

    def build_action_space(self):
        # An action is made of two increments (ax, ay) between -1 and +1
        self.action_space = [(ax, ay) for ax in range(-1, 2) for ay in range(-1, 2)]

        self.action_space_index = {
            action: action_index
            for action_index, action in enumerate(self.action_space)
        }

        self.action_dim = len(self.action_space)

    def _init_matrices(self):
        self.build_state_space()
        self.build_action_space()
        self.transition_matrix: list = [
            dok_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = -1 * np.ones(
            (self.state_dim, self.action_dim), dtype=np.float16
        )

        # Liste des états de départ
        self.start_states = [
            state
            for state in range(self.state_dim)
            if (
                self.track[self.state_space[state][:2]] == 2  # Etat de départ
                and self.state_space[state][2:] == (0, 0)  # La vitesse est nulle
            )
        ]
        # Proba d'être dans un état de départ
        self.proba_start_states = 1 / len(self.start_states)

    def check_on_track(self, x, y):
        """
        Checks if the point of coordinates (x, y) is on the track
        """
        return (0 <= x < self.X and 0 <= y < self.Y) and bool(self.track[x, y])

    def compute_new_speed(self, sx, sy, ax, ay):
        """
        From the speed (sx, sy), compute the next
        speed using the action (ax, ay)
        """
        new_sx = max(0, min(self.max_speed, sx + ax))
        new_sy = max(0, min(self.max_speed, sy + ay))
        return new_sx, new_sy

    def evolve_x(self, x, y, action):
        assert self.check_on_track(x, y)
        if self.check_on_track(x + action, y):
            return x + action, True
        else:
            return x, False

    def evolve_y(self, x, y, action):
        assert self.check_on_track(x, y)
        if self.check_on_track(x, y + action):
            return y + action, True
        else:
            return y, False

    def compute_new_position(self, x, y, sx, sy):
        assert self.check_on_track(x, y)
        speed_x, speed_y = sx, sy
        evolved_x, evolved_y = True, True
        for _ in range(abs(sx) + abs(sy)):
            if speed_x > 0:
                x, evolved_x = self.evolve_x(x, y, 1)
                speed_x -= 1

            elif speed_x < 0:
                x, evolved_x = self.evolve_x(x, y, -1)
                speed_x += 1

            if speed_y > 0:
                y, evolved_y = self.evolve_y(x, y, 1)
                speed_y -= 1

            if speed_y != 0:
                y, evolved_y = self.evolve_y(x, y, -1)
                speed_y += 1

        return x, y, evolved_x and evolved_y
